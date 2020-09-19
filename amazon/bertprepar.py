import numpy as np
import os
import torch
import torch.nn as nn

from ignite.metrics import Loss, Accuracy
from sklearn.preprocessing import LabelEncoder

from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, ConcatDataset, SubsetRandomSampler

from slp.data.collators import BertCollator
from transformers import *
from slp.data.bertamz import AmazonZiser17, NewLabelsData
from slp.modules.doublebert import DoubleHeadBert, DoubleBertCollator
from slp.data.transforms import SpacyTokenizer, ToTokenIds, ToTensor
from slp.modules.classifier import BertClassifier
from slp.modules.rnn import WordRNN
from slp.trainer.trainer import BertTrainer, AugmentBertTrainer
from slp.util.embeddings import EmbeddingsLoader
from slp.util.parallel import DataParallelCriterion, DataParallelModel

import argparse
parser = argparse.ArgumentParser(description="Domains and losses")
parser.add_argument("-s", "--source", default="books", help="Source Domain")
parser.add_argument("-t", "--target", default="dvd", help="Target Domain")
args = parser.parse_args()
SOURCE = args.source
TARGET = args.target
#targets = ["dvd", "books", "electronics", "kitchen"]

def transform_pred_tar(output):
    y_pred, targets, d  = output
    return y_pred, targets

def transform_d(output):
    y_pred, targets, d = output
    d_pred = d['domain_pred']
    d_targets = d['domain_targets']
    return d_pred, d_targets

def evaluation(trainer, test_loader, device):
    trainer.model.eval()
    predictions = []
    labels = []
    metric = Accuracy()
    with torch.no_grad():
        for index, batch in enumerate(test_loader):
            inputs = batch[0].to(device)
            label = batch[1].to(device)
            #import ipdb; ipdb.set_trace()
            pred = trainer.model(inputs)[0]
            metric.update((pred, label))
    acc = metric.compute()
    return acc

DEVICE = 'cpu'
#DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

collate_fA = BertDCollator(device='cpu')
collate_fB = BertLMCollator(tokenizer=...)
collate_fn = DoubleBertCollator(collate_A, collate_B)

def dataloaders_from_datasets(source_dataset, target_dataset,
                              batch_train, batch_val, circle,
                              val_size=0.2):
    dataset = ConcatDataset([source_dataset, target_dataset])
    s_dataset_size = len(source_dataset)
    s_indices = list(range(s_dataset_size))
    s_val_split = int(np.floor(val_size * s_dataset_size))
    s_train_indices = s_indices[s_val_split:]
    s_val_indices = s_indices[:s_val_split]

    t_dataset_size = len(target_dataset)
    t_train_indices = list(range(t_dataset_size))

    train_sampler = DoubleSubsetRandomSampler(s_train_indices, t_train_indices, s_dataset_size, batch_train, batch_train*circle)
    val_sampler = SubsetRandomSampler(s_val_indices)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_train,
        sampler=train_sampler,
        drop_last=False,
        collate_fn=collate_fn)
    val_loader = DataLoader(
        train_dataset,
        batch_size=batch_val,
        sampler=val_sampler,
        drop_last=False,
        collate_fn=collate_fn)
    return train_loader, val_loader

if __name__ == '__main__':
    dataset = AmazonZiser17(ds=SOURCE, dl=0, labeled=True, cldata=False)
    dataset2 = AmazonZiser17(ds=TARGET, dl=1, labeled=False, cldata=True)
    train_loader, val_loader = dataloaders_from_datasets(dataset, dataset2,
                                                         4, 4, 4)

    if TARGET == "books":
       pre = './sbooks'
    elif TARGET == "dvd":
       pre = './sdvd'
    elif TARGET == "electronics":
       pre = './sele'
    else:
       pre = './skit'

    model = DoubleHeadBert(pre)
    #for names, parameters in model.bert.named_parameters():
    #    parameters.requiers_grad=False

    optimizer = AdamW(model.parameters(), lr=1e-5, correct_bias=False)
    criterion = nn.CrossEntropyLoss() 
    metrics = {
        'loss': Loss(criterion),
        'accuracy': Accuracy(transform_pred_tar)
    }
    path=SOURCE+TARGET
    trainer = DoubleBertTrainer(model, optimizer,
                      newbob_period=3,
                      checkpoint_dir=os.path.join('./checkpoints/double', path),
                      metrics=metrics,
                      non_blocking=True,
                      retain_graph=True,
                      patience=3,
                      loss_fn=criterion,
                      device=DEVICE,
                      parallel=False)

    trainer.fit(train_loader, unlabeled_loader, val_loader, epochs=10)
    trainer = DoubleBertTrainer(model, optimizer=None,
                      checkpoint_dir=os.path.join('./checkpoints/double',path),
                      model_checkpoint='experiment_model.best.pth',
                      device=DEVICE)

    dataset3 = AmazonZiser17(ds=TARGET, dl=1, labeled=True, cldata=False)

    final_test_loader = DataLoader(
         dataset3,
         batch_size=1,
         drop_last=False,
         collate_fn=collate_fn)

    print(SOURCE)
    print(TARGET)
    print(evaluation(trainer, final_test_loader, DEVICE))
