from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import torch


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t, target):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        logit_t = y_t.detach()
        #
        #
        random_logits1 = torch.randn(logit_t.shape[0], logit_t.shape[1]).cuda()
        random_logits1 = random_logits1 * torch.sqrt(torch.tensor(1))
        synthetic_teacher1 = random_logits1*0.1 + logit_t*0.9
        synthetic_teacher1 = F.softmax(synthetic_teacher1 / self.T, dim=1)
        #
        random_logits2 = torch.randn(logit_t.shape[0], logit_t.shape[1]).cuda()
        random_logits2 = random_logits2 * torch.sqrt(torch.tensor(1))
        synthetic_teacher2 = random_logits2*0.1 + logit_t*0.9
        synthetic_teacher2 = F.softmax(synthetic_teacher2 / self.T, dim=1)
        #
        #
        random_logits3 = torch.randn(logit_t.shape[0], logit_t.shape[1]).cuda()
        random_logits3 = random_logits3 * torch.sqrt(torch.tensor(1))
        synthetic_teacher3 = random_logits3*0.1 + logit_t*0.9
        synthetic_teacher3 = F.softmax(synthetic_teacher3 / self.T, dim=1)
        # print("Varian 0.555555555555555555")
        #
        # random_logits4 = torch.randn(logit_t.shape[0], logit_t.shape[1]).cuda()
        # synthetic_teacher4 = random_logits4*0.1 + logit_t*0.9
        # synthetic_teacher4 = F.softmax(synthetic_teacher4 / self.T, dim=1)
        # #
        # random_logits5 = torch.randn(logit_t.shape[0], logit_t.shape[1]).cuda()
        # synthetic_teacher5 = random_logits5*0.1 + logit_t*0.9
        # synthetic_teacher5 = F.softmax(synthetic_teacher5 / self.T, dim=1)
        #
        # random_logits6 = torch.randn(logit_t.shape[0], logit_t.shape[1]).cuda()
        # synthetic_teacher6 = random_logits6*0.1 + logit_t*0.9
        # synthetic_teacher6 = F.softmax(synthetic_teacher6 / self.T, dim=1)
        #
        # random_logits7 = torch.randn(logit_t.shape[0], logit_t.shape[1]).cuda()
        # synthetic_teacher7 = random_logits7*0.1 + logit_t*0.9
        # synthetic_teacher7 = F.softmax(synthetic_teacher7 / self.T, dim=1)

        # random_logits8 = torch.randn(logit_t.shape[0], logit_t.shape[1]).cuda()
        # synthetic_teacher8 = random_logits8*0.1 + logit_t*0.9
        # synthetic_teacher8 = F.softmax(synthetic_teacher8 / self.T, dim=1)
        #
        #
        #
        # random_logits9 = torch.randn(logit_t.shape[0], logit_t.shape[1]).cuda()
        # synthetic_teacher9 = random_logits9*0.1 + logit_t*0.9
        # synthetic_teacher9 = F.softmax(synthetic_teacher8 / self.T, dim=1)

        # random_logits10 = torch.randn(logit_t.shape[0], logit_t.shape[1]).cuda()
        # synthetic_teacher10 = random_logits10*0.1 + logit_t*0.9
        # synthetic_teacher10 = F.softmax(synthetic_teacher8 / self.T, dim=1)

        # print("Teacher ten Burger.........")
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0] + 0.8*(F.kl_div(p_s, synthetic_teacher1, size_average=False) * (self.T**2) / y_s.shape[0] + F.kl_div(p_s, synthetic_teacher2, size_average=False) * (self.T**2) / y_s.shape[0] + F.kl_div(p_s, synthetic_teacher3, size_average=False) * (self.T**2) / y_s.shape[0])

        # loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0] + 0.8*(F.kl_div(p_s, synthetic_teacher1, size_average=False) * (self.T**2) / y_s.shape[0] + F.kl_div(p_s, synthetic_teacher2, size_average=False) * (self.T**2) / y_s.shape[0] + F.kl_div(p_s, synthetic_teacher3, size_average=False) * (self.T**2) / y_s.shape[0] + F.kl_div(p_s, synthetic_teacher4, size_average=False) * (self.T**2) / y_s.shape[0]+ F.kl_div(p_s, synthetic_teacher5, size_average=False) * (self.T**2) / y_s.shape[0]+ F.kl_div(p_s, synthetic_teacher6, size_average=False) * (self.T**2) / y_s.shape[0] + F.kl_div(p_s, synthetic_teacher7, size_average=False) * (self.T**2) / y_s.shape[0])
        # loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / y_s.shape[0]
        return loss
