import torch.nn as nn

from slp.modules.feedforward import FF


class Classifier(nn.Module):
    def __init__(self, encoder, encoded_features, num_classes):
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.clf = FF(encoded_features, num_classes,
                      activation='none', layer_norm=False,
                      dropout=0.)

    def forward(self, *args, **kwargs):
        x = self.encoder(*args, **kwargs)
        return self.clf(x)

class BertClassifier(nn.Module):
    def __init__(self, encoder, encoded_features, num_classes):
        super(BertClassifier, self).__init__()
        self.encoder = encoder
        self.clf = nn.Linear(encoded_features, num_classes, bias=True)

    def forward(self, *args, **kwargs):
        x = self.encoder(*args, **kwargs)
        #import ipdb; ipdb.set_trace()
        return self.clf(x[1])
