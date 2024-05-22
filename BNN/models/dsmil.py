import torch
import torch.nn as nn
import torch.nn.functional as F
from .Base import BaseModel

class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))
        
    def forward(self, feats):
        x = self.fc(feats)
        return feats, x


class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()
        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(feature_size, output_class)
        
    def forward(self, x):
        device = x.device
        feats = self.feature_extractor(x)  # N x K
        c = self.fc(feats.view(feats.shape[0], -1))  # N x C
        return feats.view(feats.shape[0], -1), c


class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0):  # K, L, N
        super(BClassifier, self).__init__()
        self.q = nn.Linear(input_size, 128)
        self.v = nn.Sequential(
            nn.Dropout(dropout_v),
            nn.Linear(input_size, input_size)
        )
        
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)
        
    def forward(self, feats, c):  # N x K, N x C
        device = feats.device
        V = self.v(feats)  # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1)  # N x Q, unsorted
        
        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True)  # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :])  # select critical instances, m_feats in shape C x K
        q_max = self.q(m_feats)  # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1))  # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax(A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0)  # normalize attention scores, A in shape N x C,
        B = torch.mm(A.transpose(0, 1), V)  # compute bag representation, B in shape C x V
                
        B = B.view(1, B.shape[0], B.shape[1])  # 1 x C x V
        C = self.fcc(B)  # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B

  
class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier
        
    def forward(self, x):
        feats, classes = self.i_classifier(x)
        prediction_bag, A, B = self.b_classifier(feats, classes)
        return classes, prediction_bag, A, B

class DSMIL(BaseModel):
    def __init__(self, input_size, num_classes, layer_type='HS', priors=None, activation_type="relu", dropout_v=0.0):
        super(DSMIL, self).__init__(layer_type, priors, activation_type)
        self.i_classifier = self.get_fc_layer(input_size, num_classes)
        self.q = self.get_fc_layer(input_size, 128)
        self.drop_out = nn.Dropout(dropout_v)
        self.v = self.get_fc_layer(input_size, input_size)
        self.fcc = self.get_conv_layer(num_classes, num_classes, kernel=input_size, conv_type="conv1d")

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
        classes = self.i_classifier(x,n_samples=train_sample if Train_flag else test_sample)
        classes = torch.mean(classes, dim=0)
        device = x.device
        classes = self.drop_out(classes)
        V = self.v(x, n_samples=train_sample if Train_flag else test_sample)
        V = torch.mean(V, dim=0)
        Q = self.q(x, n_samples=train_sample if Train_flag else test_sample)
        Q = torch.mean(Q, dim=0)
        Q = Q.view(x.shape[0], -1)
        _, m_indices = torch.sort(classes, 0, descending=True)
        m_feats = torch.index_select(x, dim=0, index=m_indices[0, :])
        q_max = self.q(m_feats, n_samples=train_sample if Train_flag else test_sample)
        q_max = torch.mean(q_max, dim=0)
        A = torch.mm(Q, q_max.transpose(0, 1))
        A = F.softmax(A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0)
        B = torch.mm(A.transpose(0, 1), V)

        B = B.view(1, B.shape[0], B.shape[1])
        C = self.fcc(B, n_samples=train_sample if Train_flag else test_sample)
        C = torch.mean(C, dim=0)
        C = C.view(1, -1)
        return classes, C, A, B

