
import torch
import torch.nn.functional as F
import numpy as np

def softmax():
    logits = torch.tensor([[2.0, 1.0, 0.1],
                           [0.5, 2.5, 1.5]])
    print("原始logits:", logits)
    print("\n=== PyTorch计算 ===")
    probs = F.softmax(logits, dim=1)
    print("softmax后的结果:", probs)
    print(probs.sum(dim=1))  # 每一行的和=1

    print("\n=== NumPy计算 ===")
    # NumPy计算 - 需要将tensor转换为numpy
    logits_np = logits.numpy()
    # 手动计算softmax
    exp_logits = np.exp(logits_np)
    sum_exp = np.sum(exp_logits, axis=1, keepdims=True)
    softmax_np = exp_logits / sum_exp
    print("NumPy手动计算softmax:", softmax_np)
    print("每一行的和:", np.sum(softmax_np, axis=1))
    
'''
log_softmax = log(softmax(x))
'''
def log_softmax():
    logits = torch.tensor([[2.0, 1.0, 0.1],
                           [0.5, 2.5, 1.5]])
    print("原始logits:", logits)
    log_probs = F.log_softmax(logits, dim=1)
    print("log_softmax后的结果:", log_probs)
    
    softmax_probs = F.softmax(logits, dim=1)
    print("softmax后的结果:", softmax_probs)
    log_softmax_manual = torch.log(softmax_probs)
    print("手动计算log(softmax)的结果:", log_softmax_manual)

# 归一化
def normalize():
    x = torch.tensor([[3.0, 4.0], [6.0, 8.0]])
    # L2范数归一化, 归一化后向量的 欧几里得长度（模）为 1，向量落在单位圆上（2维）
    # 方便计算余弦相似度=点积
    # 3 * 3 + 4 * 4 = 5 * 5
    # 3/5, 4/5
    x_normalized_2 = F.normalize(x, p=2, dim=1)
    print(x_normalized_2)
    # L1范数归一化, 归一化后向量的 元素绝对值之和为 1
    # 3 * 3 + 4 * 4 = 5 * 5
    # 3/7, 4/7
    x_normalized_1 = F.normalize(x, p=1, dim=1)
    print(x_normalized_1)


# 点积
def dot_product():
    '''
    [1, 2
     3, 4]
    [5, 6
     7, 8]

     1*5 + 2*7 = 19
     ...

    矩阵点积的几何意义：多组行向量与列向量的点积，L2归一化后点积结果=cosine(角度)，直接表示方向相似度


    '''
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # shape (2,2)
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])  # shape (2,2)

    c = a @ b
    # c1 = torch.matmul(a, b)
    print(c)
    # print(c1)

    labels = torch.arange(2)
    print("labels:", labels)
    loss = F.cross_entropy(c, labels)
    print("loss:", loss)

    # 自己手动计算CE loss
    probs = F.softmax(c, dim=1)
    print("probs:", probs)
    loss_manual = (- torch.log(probs[0][0]) - torch.log(probs[1][1]) ) / 2
    print("loss_manual:", loss_manual)

if __name__ == "__main__":
    # softmax()
    log_softmax()
    # normalize()
    # dot_product()