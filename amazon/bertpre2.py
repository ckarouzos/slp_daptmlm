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

def prediction(trainer, test_loader, device):
    trainer.model.eval()
    pos = []
    neg = []
    a = 0
    b = 0 
    soft = nn.Softmax(dim=1)
    with torch.no_grad():
        for index, batch in enumerate(test_loader):
            #import ipdb; ipdb.set_trace()
            inputs = batch[0].to(device)
            #print(inputs)
            pred = soft(trainer.model(inputs)[0])
            if pred[0][1] > pred[0][0]:
               a=a+1
               pos.append((index, pred[0][1]))
               #print(pred[0][1])
            else:
               neg.append((index, pred[0][0]))
               b=b+1
        #pos.sort(reverse=True, key=lambda tup: tup[1])
        #neg.sort(reverse=True, key=lambda tup: tup[1])
        #pos = pos[:1000]
        #neg = neg[:1000]
        all = []
        labels = []
        for a,b in neg:
            all.append(a)
            labels.append(0)
        for a,b in pos:
            all.append(a)
            labels.append(1)
    return all, labels

#DEVICE = 'cpu'
#DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

collate_fn = BertCollator(device='cpu')

if __name__ == '__main__':
    dataset = AmazonZiser17(ds=SOURCE, dl=0, labeled=True, cldata=False)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    perm = torch.randperm(len(indices))
    val_size = 0.2
    val_split = int(np.floor(val_size * dataset_size))
    train_indices = perm[val_split:]
    val_indices = perm[:val_split]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(
        dataset,
        batch_size=4,
        sampler=train_sampler,
        drop_last=False,
        collate_fn=collate_fn)
    val_loader = DataLoader(
        dataset,
        batch_size=4,
        sampler=val_sampler,
        drop_last=False,
        collate_fn=collate_fn)

    if TARGET == "books":
       pre = './sbooks'
    elif TARGET == "dvd":
       pre = './sdvd'
    elif TARGET == "electronics":
       pre = './sele'
    else:
       pre = './skit'

    #config = BertConfig.from_json_file('./config.json')
    #bertmodel = BertModel.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(pre)
    #for names, parameters in model.bert.named_parameters():
    #    parameters.requiers_grad=False

    #optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    optimizer = AdamW(model.parameters(), lr=1e-5, correct_bias=False)
    criterion = nn.CrossEntropyLoss() 
    metrics = {
        'loss': Loss(criterion),
        'accuracy': Accuracy()
    }
    path=SOURCE+TARGET
    trainer = AugmentBertTrainer(model, optimizer,
                      newbob_period=3,
                      checkpoint_dir=os.path.join('./checkpoints/bert', path),
                      metrics=metrics,
                      non_blocking=True,
                      retain_graph=True,
                      patience=3,
                      loss_fn=criterion,
                      device=DEVICE,
                      parallel=False)

    dataset2 = AmazonZiser17(ds=TARGET, dl=1, labeled=False, cldata=True)
    unlabeled_loader = DataLoader(
         dataset2,
         batch_size=1,
         drop_last=False,
         collate_fn=collate_fn)

    trainer.fit(train_loader, unlabeled_loader, val_loader, epochs=10)
    trainer = AugmentBertTrainer(model, optimizer=None,
                      checkpoint_dir=os.path.join('./checkpoints/bert',path),
                      model_checkpoint='experiment_model.best.pth',
                      device=DEVICE)

    dataset2 = AmazonZiser17(ds=TARGET, dl=1, labeled=True, cldata=False)
    #dataset2 = AmazonZiser17(ds=TARGET, dl=1, labeled=False, cldata=True)
    #test_loader = DataLoader(
     #    dataset2,
     #    batch_size=1,
     #    drop_last=False,
     #    collate_fn=collate_fn)

    #indices, labels = prediction(trainer, test_loader, DEVICE)
    #newdataset = NewLabelsData(dataset2, indices, labels)
    #dataset_size = len(newdataset)
    #indx = list(range(dataset_size))
    #perm = torch.randperm(len(indx))
    #val_size = 0.2
    #val_split = int(np.floor(val_size * dataset_size))
    # train_indices = perm[val_split:]
    #val_indices = perm[:val_split]
    #train_sampler = SubsetRandomSampler(train_indices)
    #val_sampler = SubsetRandomSampler(val_indices)
    #
    #new_loader = DataLoader(
    #    newdataset,
    #    batch_size=4,
    #    sampler=train_sampler,
    #    drop_last = False,
    #    collate_fn=collate_fn
    #)
    #new_val_loader = DataLoader(
    #    newdataset,
    #    batch_size=4,
    #    sampler=val_sampler,
    #    drop_last=False,
    #    collate_fn=collate_fn)
    #trainer = BertTrainer(model, optimizer,
    #                  newbob_period=3,
    #                  checkpoint_dir=os.path.join('./checkpoints/final',path),
    #                  metrics=metrics,
    #                  non_blocking=True,
    #                  retain_graph=True,
    #                  patience=2,
    #                  loss_fn=criterion,
    #                  device=DEVICE,
    #                  parallel=False)
    #trainer.fit(train_loader, val_loader, epochs=10)

    #trainer = BertTrainer(model, optimizer=None,
    #                  checkpoint_dir=os.path.join('./checkpoints/final',path),
    #                  model_checkpoint='experiment_model.best.pth',
    #                  device=DEVICE)

    final_test_loader = DataLoader(
         dataset2,
         batch_size=1,
         drop_last=False,
         collate_fn=collate_fn)

    print(SOURCE)
    print(TARGET)
    print(evaluation(trainer, final_test_loader, DEVICE))
