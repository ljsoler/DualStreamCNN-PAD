import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BinaryAngularLoss(nn.Module):
    def __init__(self, in_features, out_features, m=None, eps=1e-7):
        super(BinaryAngularLoss, self).__init__()
        
        self.m = 0.5 if not m else m
        self.eps = eps
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.fc = nn.Linear(in_features, out_features, bias=False)


    def forward(self, x, labels):
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)
        wf = self.fc(x)
        theta = wf
        theta = torch.clamp(theta, -1.+self.eps, 1-self.eps)
        theta = torch.acos(theta)
        pos = nn.Sigmoid()(torch.cos(theta+self.m))
        neg = nn.Sigmoid()(torch.cos(theta))
        labels = torch.unsqueeze(labels, 1)
        L = labels*torch.log(pos) + (1-labels)*torch.log(1-neg)
        L = torch.mean(L)
#         print(f"---------------------sigmoid----------------------------------")
#         print(nn.Sigmoid()(wf)[0], labels[0])
#         print(nn.Sigmoid()(wf)[1], labels[1])
#         print(nn.Sigmoid()(wf)[2], labels[2])
#         print(nn.Sigmoid()(wf)[3], labels[3])
#         print(nn.Sigmoid()(wf)[4], labels[4])
#         print(nn.Sigmoid()(wf)[5], labels[5])
        return -L, nn.Sigmoid()(wf)
        
class BinaryMarginAngularLoss(nn.Module):
    def __init__(self, m=0.5, std = 0.05):
        super(BinaryMarginAngularLoss, self).__init__()
        
        self.m = m
        self.std = std


    def forward(self, x, labels):
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0

        theta = x

        index = torch.where(labels != 0)[0]
        m_hot = torch.zeros(theta.size()[0], theta.size()[1], device=theta.device)
        margin = torch.normal(mean=self.m, std=self.std, size=labels[index, None].size(), device=theta.device)

        m_hot[index] += margin

        theta = torch.clamp(theta, -1., 1)
        theta = torch.acos(theta)
        theta_pos = theta + m_hot
        pos = nn.Sigmoid()(torch.cos(theta_pos))
        neg = nn.Sigmoid()(torch.cos(theta))
        labels = torch.unsqueeze(labels, 1)
        L = labels*torch.log(pos) + (1-labels)*torch.log(1-neg)
        L = torch.mean(L)
#         print(f"---------------------sigmoid----------------------------------")
#         print(nn.Sigmoid()(wf)[0], labels[0])
#         print(nn.Sigmoid()(wf)[1], labels[1])
#         print(nn.Sigmoid()(wf)[2], labels[2])
#         print(nn.Sigmoid()(wf)[3], labels[3])
#         print(nn.Sigmoid()(wf)[4], labels[4])
#         print(nn.Sigmoid()(wf)[5], labels[5])
        return -L
#     def forward(self, x, labels):
#         assert len(x) == len(labels)
#         assert torch.min(labels) >= 0

#         x = F.normalize(x, p = 2, dim= 1)

#         kernel_norm = F.normalize(self.kernel, p = 2, dim=0)

#         wf = torch.mm(x, kernel_norm)

#         theta = wf

#         theta = torch.clamp(theta, -1., 1)

#         index = torch.where(labels != 0)[0]

#         m_hot = torch.zeros(theta.size()[0], theta.size()[1], device=theta.device)

#         margin = torch.normal(mean=self.m, std=self.std, size=labels[index, None].size(), device=theta.device)

#         theta = torch.acos(theta)

#         m_hot[index] += margin
        
#         pos = nn.Sigmoid()(torch.cos(theta) + m_hot)
#         neg = nn.Sigmoid()(torch.cos(theta))
#         labels = torch.unsqueeze(labels, 1)
#         L = labels*torch.log(pos) + (1-labels)*torch.log(1-neg)
#         L = torch.mean(L)
# #         print(f"---------------------sigmoid----------------------------------")
# #         print(nn.Sigmoid()(wf)[0], labels[0])
# #         print(nn.Sigmoid()(wf)[1], labels[1])
# #         print(nn.Sigmoid()(wf)[2], labels[2])
# #         print(nn.Sigmoid()(wf)[3], labels[3])
# #         print(nn.Sigmoid()(wf)[4], labels[4])
# #         print(nn.Sigmoid()(wf)[5], labels[5])
#         return -L, nn.Sigmoid(wf)