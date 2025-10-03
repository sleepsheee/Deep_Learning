
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

# 范数、归一化
def normalize():

    print("\n=== 1D向量范数计算 ===")
    vector = torch.tensor([3.0, 4.0, 0.0])
    print(f"向量: {vector}")
    
    # L1范数 (曼哈顿距离)
    l1_norm = torch.norm(vector, p=1)
    print(f"L1范数 (p=1): {l1_norm} = |3| + |4| + |0| = {3+4+0}")
    
    # L2范数 (欧几里得距离)
    l2_norm = torch.norm(vector, p=2)
    print(f"L2范数 (p=2): {l2_norm:.4f} = √(3² + 4² + 0²) = {torch.sqrt(torch.tensor(9+16))}")
    

    print("\n=== 2D矩阵范数计算 ===")
    '''
    两个向量规范化得到单位长度后，点积表示它们夹角的余弦
    '''
    matrix = torch.tensor([[1.0, 2.0],
                          [3.0, 4.0]])
    print(f"\n矩阵:\n{matrix}")
    
    # Frobenius范数 (默认)
    fro_norm = torch.norm(matrix)
    print(f"Frobenius范数 (默认): {fro_norm:.4f} = √(1²+2²+3²+4²)")
    

    print("\n=== 3D张量范数计算 ===")
    # 创建3D张量 (batch_size, features, timesteps)
    tensor_3d = torch.randn(2, 3, 4)  # 2个样本, 3个特征, 4个时间步
    print("3D张量:\n", tensor_3d)
    print(f"3D张量形状: {tensor_3d.shape}")
    
    print("\n--- 不同维度的范数计算 ---")
    
    # 计算每个样本的范数 (保持特征和时间维度)
    norm_dim_none = torch.norm(tensor_3d, p=2, dim=None)
    print(f"整个张量的Frobenius范数 (dim=None): {norm_dim_none:.4f}")
    
    # 沿特征维度计算 (对每个样本的每个时间步)
    norm_dim_1 = torch.norm(tensor_3d, p=2, dim=1)
    print(f"沿特征维度 (dim=1) 形状: {norm_dim_1.shape}")
    
    # 沿时间维度计算 (对每个样本的每个特征)
    norm_dim_2 = torch.norm(tensor_3d, p=2, dim=2)
    print(f"沿时间维度 (dim=2) 形状: {norm_dim_2.shape}")
    
    # 沿多个维度计算
    norm_dims_1_2 = torch.norm(tensor_3d, p=2, dim=(1, 2))
    print(f"沿特征和时间维度 (dim=(1,2)) 形状: {norm_dims_1_2.shape}")


    print("\n=== 归一化 ===")
    # L2范数归一化, 归一化后向量的 欧几里得长度（模）为 1，向量落在单位圆上（2维）
    # 方便计算余弦相似度=点积
    # 3 * 3 + 4 * 4 = 5 * 5
    # 3/5, 4/5
    # dim=0：列操作，每列成为一个单位向量
    # dim=1：行操作，每行成为一个单位向量

    x = torch.tensor([[3.0, 4.0],
                      [1.0, 2.0]])
    x_normalized_2 = F.normalize(x, p=2, dim=-1)
    print(f"\nL2归一化沿dim1: ", x_normalized_2) # default p=2
    print(f"\nL2归一化沿dim0", F.normalize(x, p=2, dim=0))

    # # L1范数归一化, 归一化后向量的 元素绝对值之和为 1
    # # 3 * 3 + 4 * 4 = 5 * 5
    # # 3/7, 4/7
    x_normalized_1 = F.normalize(x, p=1, dim=1)
    print(f"\nL1归一化:", x_normalized_1)


# 点积
def dot_product():
    print("=== 向量点积 ===")
    
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    
    print(f"向量 a: {a}, shape: {a.shape}")
    print(f"向量 b: {b}, shape: {b.shape}")
    
    print(f"torch.dot(a, b): {torch.dot(a, b)}")
    print(f"torch.matmul(a, b): {torch.matmul(a, b)}")

   
    print("\n=== 矩阵-向量点积 ===")
    
    # 矩阵 (2x3)
    A = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]])
    
    # 向量 (3,)
    # 在数学表示上，一维张量既可以看作是行向量也可以看作是列向量，取决于具体的运算上下文
    v = torch.tensor([1.0, 2.0, 3.0])
    
    print(f"矩阵 A (2x3):\n{A}")
    print(f"向量（一维张量） v (3,): {v}")
    
    # 矩阵 × 向量
    result = torch.matmul(A, v)
    print(f"A × v = {result}, shape: {result.shape}")
    

    '''
    a = [1, 2
        3, 4]
    b = [5, 6
        7, 8]

    1*5 + 2*7 = 19
    1*6 + 2*8 = 22
     ...

    矩阵点积的几何意义：多组行向量与列向量的点积，L2归一化后点积结果=cosine(角度)，直接表示方向相似度
    '''
    print("\n=== 矩阵点积 ===")
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # shape (2,2)
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])  # shape (2,2)

    c1 = a @ b
    c2 = torch.matmul(a, b)
    print(f"@:", c1)
    print(f"torch.matmul:", c2)

    # labels = torch.arange(2)
    # print("labels:", labels)
    # loss = F.cross_entropy(c, labels)
    # print("loss:", loss)

    # 自己手动计算CE loss
    # probs = F.softmax(c, dim=1)
    # print("probs:", probs)
    # loss_manual = (- torch.log(probs[0][0]) - torch.log(probs[1][1]) ) / 2
    # print("loss_manual:", loss_manual)

if __name__ == "__main__":
    # softmax()
    # log_softmax()
    # normalize()
    dot_product()
