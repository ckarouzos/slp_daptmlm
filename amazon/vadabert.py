import numpy as np
import os 

import torch
import torch.nn as nn

from ignite.metrics import Loss, Accuracy

from torch.utils.data import Dataset, DataLoader, ConcatDataset, SubsetRandomSampler
from torch.optim import Adam
from slp.data.collators import BertDACollator
from transformers import *
from slp.data.bertamz import AmazonZiser17
from slp.data.transforms import ToTokenIds, ToTensor
from slp.modules.vada import ConditionalEntropyLoss, VADALoss, VAT, VADAClassifier, DASubsetRandomSampler
from slp.modules.rnn import WordRNN
from slp.trainer.trainer import BertVADATrainer
from slp.util.embeddings import EmbeddingsLoader

import argparse 
parser = argparse.ArgumentParser(description="Domains and losses")
parser.add_argument("-s", "--source", default="books", help="Source Domain")
parser.add_argument("-t", "--target", default="dvd", help="Target Domain")
parser.add_argument("-a", type=float, default=0.01, help="Domain Adversarial HyperParameter")
parser.add_argument("-b", type=float, default=0.01, help="C.E. HyperParameter")
parser.add_argument("-c", type=float, default=0.01, help="VAT HyperParameter")
parser.add_argument("-i", default="0", help="Path")
args = parser.parse_args()
SOURCE = args.source
TARGET = args.target
a = args.a
b = args.b
c = args.c
path = args.i

def transform_pred_tar(output):
    y_pred, targets, d  = output
    d_targets = d['domain_targets']
    y_pred = torch.stack([p for p,t in zip(y_pred, targets) if t>=0])
    targets = torch.stack([t for t in targets if t>=0])
    return y_pred, targets

def transform_d(output):
    y_pred, targets, d = output
    d_pred = d['domain_pred']
    d_targets = d['domain_targets']
    return d_pred, d_targets

def transform_t(output):
    y_pred, targets, d = output
    d_pred = d['domain_pred']
    d_targets = d['domain_targets']
    y_pred = torch.stack([p for p,d in zip(y_pred, d_targets) if d==1])
    return y_pred

def evaluation(trainer, test_loader, device):
    trainer.model.eval()
    predictions = []
    labels = []
    metric = Accuracy()
    with torch.no_grad():
        for index, batch in enumerate(test_loader):
            review = batch[0].to(device)
            label = batch[1].to(device)
            domain = batch[2].to(device)
            length = batch[3].to(device)
            pred, _ = trainer.model(review, length)
            metric.update((pred, label))
    acc = metric.compute()
    return acc

#DEVICE = 'cpu'
DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'

collate_fn = BertDACollator(device='cpu')

def dataloaders_from_datasets(source_dataset, target_dataset, test_dataset,
                              batch_train, batch_val, batch_test, 
                              val_size=0.2):
    dataset = ConcatDataset([source_dataset, target_dataset])

    s_dataset_size = len(source_dataset)
    s_indices = list(range(s_dataset_size))
    s_val_split = int(np.floor(val_size * s_dataset_size))
    s_train_indices = s_indices[s_val_split:]
    s_val_indices = s_indices[:s_val_split]

    t_dataset_size = len(target_dataset)
    t_indices = list(range(t_dataset_size))
    t_val_split = int(np.floor(val_size * t_dataset_size))
    t_train_indices = t_indices[t_val_split:]
    t_val_indices = t_indices[:t_val_split]

    testset_size = len(test_dataset)
    test_indices = list(range(testset_size))
    x = 4
    train_sampler = DASubsetRandomSampler(s_train_indices, t_train_indices, s_dataset_size, x, batch_train)
    val_sampler = DASubsetRandomSampler(s_val_indices, t_val_indices, s_dataset_size, x, batch_val)
    #test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_train,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=collate_fn)
    val_loader = DataLoader(
        dataset,
        batch_size=batch_val,
        sampler=val_sampler,
        drop_last=True,
        collate_fn=collate_fn)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        drop_last=False,
        collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    source_dataset = AmazonZiser17(ds=SOURCE, dl=0, labeled=True)
    target_dataset = AmazonZiser17(ds=TARGET, dl=1, labeled=False)
    test_dataset = AmazonZiser17(ds=TARGET, dl=1, labeled=True, train=False)
    train_loader, dev_loader, test_loader = dataloaders_from_datasets(source_dataset, 
                                                                      target_dataset, 
                                                                      test_dataset, 
                                                                      4, 4, 1)
    bertmodel = BertModel.from_pretrained('bert-base-uncased')
    model = VADAClassifier(bertmodel, 768, 2, 2)
    optimizer = Adam(model.parameters(), lr=1e-5, weight_decay=1e-2)
    cl_loss = nn.CrossEntropyLoss()
    da_loss = nn.CrossEntropyLoss()
    trg_cent_loss = ConditionalEntropyLoss()
    vat_loss = VAT(model)
    criterion = VADALoss(cl_loss, da_loss, trg_cent_loss, vat_loss, a, b, c, 12, 800)
    metrics = {
        'loss': Loss(criterion),
        'accuracy': Accuracy(transform_pred_tar),
        'Domain accuracy': Accuracy(transform_d),
        'Class. Loss': Loss(cl_loss, transform_pred_tar),
        'D.A. Loss': Loss(da_loss, transform_d)
    }
    trainer = BertVADATrainer(model, optimizer,
                              checkpoint_dir=os.path.join("./checkpoints", path),
                              metrics=metrics,
                              non_blocking=True,
                              retain_graph=True,
                              patience=3,
                              loss_fn=criterion,
                              device=DEVICE)
    trainer.fit(train_loader, dev_loader, epochs=10)
    trainer = BertVADATrainer(model, optimizer=None,
                              checkpoint_dir= os.path.join("./checkpoints", path),
                              model_checkpoint='experiment_model.best.pth',
                              device=DEVICE)
    print(a, b, c, SOURCE, TARGET)
    print(evaluation(trainer, test_loader, DEVICE))
