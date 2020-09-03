import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from slp.modules.regularization import GaussianNoise
import numpy as np
import math

from slp.modules.attention import Attention
from slp.modules.embed import Embed
from slp.modules.helpers import PackSequence, PadPackedSequence

from slp.modules.util import pad_mask
from slp.modules.rnn import RNN
from slp.modules.feedforward import FF

from torch.autograd import Function
from torch.utils.data.sampler import Sampler

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)

class ConditionalEntropyLoss(nn.Module):
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        return -1.0 * b.mean(dim=0)

def switch_attr(m):
    if hasattr(m, 'track_running_stats'):
        m.track_running_stats ^= True

def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d

class VAT(nn.Module):

    def __init__(self, model, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VAT, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip
        self.model = model

    def forward(self, x, lengths):
        return x

class VADALoss(nn.Module):
    def __init__(self, loss_fn_cl, loss_fn_d, loss_fn_ce, loss_fn_vat, a, b, c, max_epochs, steps):
        super(VADALoss, self).__init__()
        self.loss_fn_cl = loss_fn_cl
        self.loss_fn_d = loss_fn_d
        self.loss_fn_ce = loss_fn_ce
        self.loss_fn_vat = loss_fn_vat
        self.a = a
        self.b = b
        self.c = c
        self.max_epochs = max_epochs
        self.steps = steps
    
    def calc_a(self, epoch, step):
        t = (epoch-1)*self.steps+step
        p = t/(self.max_epochs * self.steps)
        a = 2/(1+math.exp(-10*p)) - 1
        return a

    def forward(self, pred, tar, domain_pred, domain_targets, inputs, epoch=1, step=1600):
        if 0 in domain_targets:
            s_predictions = torch.stack([p for p,d in zip (pred, domain_targets) if d==0])
            s_targets = torch.stack([t for t,d in zip(tar,domain_targets) if d==0])
            s_inputs = torch.stack([i for i,d in zip (inputs, domain_targets) if d==0])
            #s_lengths = torch.stack([l for l,d in zip (lengths, domain_targets) if d==0])
            loss_cl = self.loss_fn_cl(s_predictions, s_targets)
            #loss_vat_s = self.loss_fn_vat(s_inputs, s_lengths) if self.c>0 else 0
        else:
            loss_cl = 0
            loss_vat_s = 0
        if 1 in domain_targets:
            t_predictions = torch.stack([p for p,d in zip(pred, domain_targets) if d==1])
            t_inputs = torch.stack([i for i,d in zip (inputs, domain_targets) if d==1])
            #t_lengths = torch.stack([l for l,d in zip (lengths, domain_targets) if d==1])
            loss_ce = self.loss_fn_ce(t_predictions) if self.b>0 else 0
            #loss_vat_t = self.loss_fn_vat(t_inputs, t_lengths) if self.c>0 else 0
        else:
            loss_ce = 0
            loss_vat_t = 0
        loss_vat_s = 0
        loss_vat_t = 0
        loss_d = self.loss_fn_d(domain_pred, domain_targets) if self.a>0 else 0
        a = self.calc_a(epoch, step)
        return loss_cl + a * loss_d + self.b * loss_ce + self.c *  loss_vat_s + self.c * loss_vat_t #NOTSURE

class VADAClassifier(nn.Module):
    def __init__(self, encoder, encoded_features, num_classes, num_domains):
        super(VADAClassifier, self).__init__()
        self.encoder = encoder
        self.clf = FF(encoded_features, num_classes,
                      activation='none', layer_norm=False,
                      dropout=0.)
        self.da = FF(encoded_features, num_domains,
                      activation='none', layer_norm=False,
                      dropout=0.)

    def forward(self, x, noise=False, d=None):
            #import ipdb; ipdb.set_trace()
            x = self.encoder(x)
            x = x[1]
            #noise
            y = grad_reverse(x)
            return self.clf(x), self.da(y)

class DASubsetRandomSampler(Sampler):
    def __init__(self, indices_source, indices_target, s_dataset_size, num_source, batch_size):  
        self.indices_source = indices_source
        self.indices_target = indices_target
        self.s_dataset_size = s_dataset_size
        self.num_source = num_source
        self.num_target = batch_size - num_source
        self.batch_size = batch_size

    def __iter__(self):
        perm = torch.randperm(len(self.indices_source))
        tarperm = torch.randperm(len(self.indices_target))
        T = 0
        t = 0
        for i, s in enumerate(perm, 1): 
            yield self.indices_source[s]
            if i % self.num_source == 0:
                for j in range(self.num_target):
                    t = T + j
                    yield self.s_dataset_size + self.indices_target[tarperm[t]]
                T = t + 1 

    def __len__(self):
        n_full = int(np.floor(len(self.indices_source) / self.num_source))
        last = len(self.indices_source) % self.num_source
        return int(n_full * self.batch_size + last)
