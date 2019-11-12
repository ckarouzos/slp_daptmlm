import torch.nn as nn
from torch.autograd import Function

from slp.modules.feedforward import FF

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)

class DALoss(nn.Module):
    def __init__(self, loss_fn_cl, loss_fn_d):
        super(DALoss, self).__init__()
        self.loss_fn_cl = loss_fn_cl
        self.loss_fn_d = loss_fn_d

    def forward(self, pred, targets, domain_pred, domain_targets):
        loss_cl = self.loss_fn_cl(pred, targets)
        loss_d = self.loss_fn_d(domain_pred, domain_targets)
        return loss_cl + loss_d #NOTSURE

class DAClassifier(nn.Module):
    def __init__(self, encoder, encoded_features, num_classes, num_domains):
        super(DAClassifier, self).__init__()
        self.encoder = encoder
        self.clf = FF(encoded_features, num_classes,
                      activation='none', layer_norm=False,
                      dropout=0.)
        self.da = FF(encoded_features, num_domains,
                      activation='none', layer_norm=False,
                      dropout=0.)

    def forward(self, x, lengths):
        x = self.encoder(x, lengths)
        y = grad_reverse(x)
        return self.clf(x), self.da(y)
