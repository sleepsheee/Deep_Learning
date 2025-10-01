import torch
import torch.nn.functional as F
import torch.nn as nn

# https://zhuanlan.zhihu.com/p/1950257135775642370

'''
KD with T v1
'''
class DistillKL(nn.Module):
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)

        if isinstance(y_t, list):
            p_t_list = [F.softmax(y / self.T, dim=1) for y in y_t]
            p_t_tensor = torch.stack(p_t_list)
            p_t = p_t_tensor.mean(0)
        else:
            p_t = F.softmax(y_t / self.T, dim=1)

        loss = F.kl_div(p_s, p_t, reduction='sum') * \
               (self.T ** 2) / y_s.shape[0]
        return loss


'''
KD with T v2
'''
class DistillKL1(nn.Module):

    def __init__(self, T):
        super(DistillKL1, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)

        if isinstance(y_t, list):
            p_t_list = [F.softmax(y / self.T, dim=1) for y in y_t]
            p_t_tensor = torch.stack(p_t_list)
            p_t = p_t_tensor.mean(0)
        else:
            p_t = F.softmax(y_t / self.T, dim=1)
        # the sum of the output will be divided by the batchsize 
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T ** 2)
        return loss  
    

class ManualDistillKL(nn.Module):
    def __init__(self, T):
        super(ManualDistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.log_softmax(y_t / self.T, dim=1)
        # 给Loss * T^2, loss不变, 但使得T增大时, softmax的梯度不会变小
        kl = (F.softmax(y_t, dim=1) * (p_t - p_s)).sum(dim=1).mean() * (self.T ** 2)
        '''
        P:真实概率分布, p_t  Q:近似概率分布, p_s
        KL(P||Q) = Σ P(x) * log(P(x)/Q(x)) = Σ P(x) * (log(P(x)) - log(Q(x)))
        '''
        print("手动计算 KL(P||Q):", kl.item())
        return kl

if __name__ == "__main__":
    # input
    logits = torch.tensor([[0.3, 0.7]])
    log_logits = F.log_softmax(logits / 1.0, dim=1)
    
    # target
    target = torch.tensor([[0.4, 0.6]])
    target_probs = F.softmax(target / 1.0, dim=1)
    print("log_logits:", log_logits)
    print("target_probs:", target_probs)

    distill_kl = DistillKL(T=1.0)
    distill_kl1 = DistillKL1(T=1.0)
    loss = distill_kl(logits, target)
    print("loss:", loss)  
    loss_1 = distill_kl1(logits, target)
    print("loss1:", loss_1)

    '''
    pytorch官方类KLDivLoss, 不能自定义T
    'none': no reduction will be applied 
    'batchmean': the sum of the output will be divided by the batchsize 
    'sum': the output will be summed 
    'mean': the output will be divided by the number of elements in the output Default: 'mean'
    '''
    # input should be a distribution in the log space
    kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=False) 
    KLDivLoss_loss = kl_loss(log_logits, target_probs)
    print("KLDivLoss Loss:", KLDivLoss_loss)
    
    '''
    manual calculation of KL divergence
    '''
    manual_loss = ManualDistillKL(T=1.0)(logits, target)


