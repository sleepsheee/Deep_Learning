import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import json
from open_clip.factory import create_model_and_transforms
from torchvision import datasets, transforms
import torchvision.models as models
from open_clip import get_tokenizer, get_model_config, get_cast_dtype, build_zero_shot_classifier
from torch.utils.data import DataLoader, Dataset

import random

class DataLoad():
    def __init__(self):
        self.train_loader = None
        self.test_loader = None
        self.catalog = None
        self.all_templates = None
        self.all_labels = None

    def load_metadata(self, metadir="clipeval"):
        with open(os.path.join(metadir, 'dataset_catalog.json')) as f:
            self.catalog = json.load(f)

        with open(os.path.join(metadir, 'templates.json')) as f:
            self.all_templates = json.load(f)

        with open(os.path.join(metadir, 'labels.json')) as f:
            self.all_labels = json.load(f)
    
    def get_cifar100_dataloaders(self, batch_size=128):
        train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                    (0.26862954, 0.26130258, 0.27577711)),
            ])

        train_set = datasets.CIFAR100(root="./data", download=True, train=True, transform=train_transform)
        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                (0.26862954, 0.26130258, 0.27577711)),
        ])

        test_set = datasets.CIFAR100(root="./data", download=True, train=False, transform=test_transform)
        self.test_loader = DataLoader(test_set, batch_size=batch_size // 2, shuffle=False)



    '''
    model.encode_image(image: Tensor)
Given a batch of images, returns the image features encoded by the vision portion of the CLIP model.

model.encode_text(text: Tensor)
Given a batch of text tokens, returns the text features encoded by the language portion of the CLIP model.

model(image: Tensor, text: Tensor)
Given a batch of images and a batch of text tokens, returns two Tensors, containing the logit scores corresponding to each image and text input. The values are cosine similarities between the corresponding image and text features, times 100.
    '''

    def load_clip_model_from_local(self, model_name, weights_path, device):
        # model_cfg = get_model_config(model_name)
        # dtype = get_cast_dtype("fp32")
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            model_name=model_name,
            pretrained=None,
            device=device)
        '''
        当你传入 "openai" 或 "laion400m_e32" 时，函数会 自动下载并加载对应预训练权重
        如果传入 None，则 只初始化模型结构，不会加载任何权重, 模型的参数会按 PyTorch 默认初始化方式 初始化（通常是随机初始化）
        如果你想用本地 checkpoint（.pt 或 .bin），直接传文件路径不一定生效。更通用的方式是： 然后再手动 load_state_dict。
        '''
        state_dict = torch.load(weights_path, map_location=device)
        # 当使用 torch.nn.DataParallel 或多GPU训练时，PyTorch会自动给参数key添加"module."前缀单GPU模型没有这个前缀，所以需要移除才能正确加载
        if any(key.startswith("module.") for key in state_dict.keys()):
            state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()} 
        model.load_state_dict(state_dict)
        # 禁用dropout、batch norm使用运行时的统计量
        model.eval()
        # model: 加载好权重的CLIP模型
        # preprocess_val: 验证/推理用的图像预处理流程
        return model, preprocess_val
    
    def run(self):
        self.load_metadata()
        self.get_cifar100_dataloaders()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)


        print(f"train batches num: {len(self.train_loader)}")
        print(f"test batches num: {len(self.test_loader)}")

        self.load_metadata(metadir="clipeval")
        templates = self.all_templates['CIFAR100']
        classnames = self.all_labels['CIFAR100']
        num_classes = 100

        
        local_weights_path = "clip_models\DFN2B-CLIP-ViT-B-16\open_clip_pytorch_model_1.bin"
        teacher_model, preprocess = self.load_clip_model_from_local("ViT-B-16", local_weights_path, device=device)
        tokenizer = get_tokenizer("ViT-B-16")

        student_model = ResNet18FeatureExtractor(weights=None)
        student_model.to(device)
        teacher_model.to(device)
        student_model.eval()
        teacher_model.eval()
        
        for i, (images, targets) in enumerate(self.train_loader):
            if i == 0:
                # [batch_size, channel_num, height, weight]
                print(f"i: {i}, images shape: {images.shape}, targets shape: {targets.shape}")
                target = targets[0]
                image = images[0]
                image = image.to(device)
                print(f"target: {target}, image shape: {image.shape}")
                # targets_text = random.choice(templates).format(classnames[target])
                # print(f"targets_text:", targets_text)
                # text_inputs = tokenizer(targets_text, context_length=77)
                # print(f"text_inputs.shape: {text_inputs.shape}")
                # print(f"text_inputs: {text_inputs}")

                # 使用unsqueeze函数在第0个维度上添加一个大小为1的维度
                image = image.unsqueeze(0)
                student_image_features = student_model(image)
                classifier_head = ClassificationHead(512, num_classes).to(device)
                student_logits_classification = classifier_head(student_image_features)
                teacher_image_features = teacher_model.encode_image(image)
                teacher_logits_classification = classifier_head(teacher_image_features)

                print(f"student_image_features shape: {student_image_features.shape}")
                print(f"student_logits_classification shape: {student_logits_classification.shape}")
                print(f"teacher_image_features: {teacher_image_features.shape}")
                print(f"teacher_logits_classification shape: {teacher_logits_classification.shape}")

                # 使用Softmax转换为概率
                student_probabilities = torch.softmax(student_logits_classification, dim=1)
                teacher_probabilities = torch.softmax(teacher_logits_classification, dim=1)          
                # 获取最大值的索引
                student_index = torch.argmax(student_probabilities)
                teacher_index = torch.argmax(teacher_probabilities)
                # print("概率之和:", logits_classification.sum(dim=1))  
                print(f"student最大值索引:", student_index.item())
                print(f"teacher最大值索引:", teacher_index.item())
                break



class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, weights=None):
        super(ResNet18FeatureExtractor, self).__init__()

        # 加载 ResNet18 模型
        self.resnet18 = models.resnet18(weights=weights)

        # 去掉 ResNet18 的最后一个全连接分类层
        # 这里我们只保留卷积部分和全局平均池化层
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-1])  # 去掉最后的全连接层

        # 全局平均池化后，输出特征的大小是 (batch_size, 512)
        self.fc = nn.Linear(512, 512)  # 保证最终输出维度为 512

    def forward(self, x):
        x = self.resnet18(x)  # 特征提取部分
        x = torch.flatten(x, 1)  # 将输出展平成(batch_size, 512)
        x = self.fc(x)  # 映射到512维
        return x

class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.linear(x)
        out = self.classifier(x)
        return out
    

if __name__ == "__main__":
    dataload = DataLoad()
    dataload.run()
    