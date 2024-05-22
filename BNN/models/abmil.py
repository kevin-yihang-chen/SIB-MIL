import torch
import torch.nn as nn
import torch.nn.functional as F
from .Base import BaseModel

class BClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(BClassifier, self).__init__()
        self.L = input_size
        self.D = input_size
        self.K = 1

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.ReLU(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.D, num_classes)
        )
        
    def forward(self, x):
        H = x
        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, H)  # KxL
        Y_prob = self.classifier(M)
        return Y_prob

class ABMIL(BaseModel):
    def __init__(self,input_size, num_classes, layer_type='HS', priors=None, activation_type="relu"):
        super(ABMIL, self).__init__(layer_type, priors, activation_type)
        self.L = input_size
        self.D = input_size
        self.K = 1
        self.fc_1 = self.get_fc_layer(self.L, self.D)
        self.fc_2 = self.get_fc_layer(self.D, self.K)
        # self.attention = nn.Sequential(
        #     self.get_fc_layer(self.L, self.D),
        #     self.act,
        #     self.get_fc_layer(self.D, self.K)
        # )

        self.classifier = nn.Sequential(
            self.get_fc_layer(self.D, num_classes)
        )

    def kl_loss(self):
        modules = [m for (name, m) in self.named_modules() if m != self and hasattr(m, 'kl_loss')]
        kl = [m.kl_loss() for m in modules]
        kl = [float(k) for k in kl]
        kl = torch.Tensor(kl)
        kl = torch.nan_to_num(kl, neginf=0)
        kl = torch.nanmean(torch.Tensor(kl))

        return kl

    def analytic_update(self):
        modules = [m for (name, m) in self.named_modules() if m != self and hasattr(m, 'analytic_update')]
        for m in modules:
            m.analytic_update()

    def forward(self, x, Train_flag=True, train_sample=1 ,test_sample=1):
        H = x
        A = self.fc_1(H,n_samples=train_sample if Train_flag else test_sample)
        A = torch.mean(A, dim=0)
        A = self.act(A)
        A = self.fc_2(A,n_samples=train_sample if Train_flag else test_sample)
        A = torch.mean(A, dim=0)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        M = torch.mm(A, H)
        Y_prob = self.classifier(M)
        return Y_prob



