import numpy as np

import torch
import torch.nn as nn

from ignite.metrics import Loss, Accuracy
from sklearn.preprocessing import LabelEncoder

from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, ConcatDataset, SubsetRandomSampler

from slp.data.collators import DACollator
from slp.data.data_amz import AmazonDataset
from slp.data.transforms import SpacyTokenizer, ToTokenIds, ToTensor
from slp.modules.daclassifier import DAClassifier, DALoss
from slp.modules.rnn import WordRNN
from slp.trainer.trainer import DATrainer
from slp.util.embeddings import EmbeddingsLoader

def transform_pred_tar(output):
    y_pred, targets, d  = output
    return y_pred, targets


def transform_d(output):
    y_pred, targets, d = output
    d_pred = d['domain_pred']
    d_targets = d['domain_targets']
    return d_pred, d_targets

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

collate_fn = DACollator(device='cpu')

def dataloaders_from_indices(dataset, train_indices, val_indices, batch_train, batch_val):
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_train,
        sampler=train_sampler,
        drop_last=False,
        collate_fn=collate_fn)
    val_loader = DataLoader(
        dataset,
        batch_size=batch_val,
        sampler=val_sampler,
        drop_last=False,
        collate_fn=collate_fn)
    return train_loader, val_loader


def train_test_split(dataset, batch_train, batch_val,
                     test_size=0.2, shuffle=True, seed=None):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    test_split = int(np.floor(test_size * dataset_size))
    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)

    train_indices = indices[test_split:]
    val_indices = indices[:test_split]
    return dataloaders_from_indices(dataset, train_indices, val_indices, batch_train, batch_val)


if __name__ == '__main__':
    loader = EmbeddingsLoader(
        '../cache/glove.840B.300d.txt', 300)
    word2idx, _, embeddings = loader.load()

    tokenizer = SpacyTokenizer()
    to_token_ids = ToTokenIds(word2idx)
    to_tensor = ToTensor(device='cpu')

    d1 = (AmazonDataset('../../data/', 'books').map(tokenizer).map(to_token_ids).map(to_tensor))
    d2 = (AmazonDataset('../../data/', 'dvd').map(tokenizer).map(to_token_ids).map(to_tensor))
    dataset = ConcatDataset([d1, d2])

    train_loader, dev_loader = train_test_split(dataset, 32, 128)
  
    model = DAClassifier(
        WordRNN(256, embeddings, bidirectional=True, merge_bi='cat',
                packed_sequence=True, attention=True, device=DEVICE),
        512, 3, 2)

    optimizer = Adam([p for p in model.parameters() if p.requires_grad],
                     lr=1e-3)
    cl_loss = nn.CrossEntropyLoss()
    da_loss = nn.CrossEntropyLoss()    
    criterion = DALoss(cl_loss, da_loss)

    metrics = {
        'loss': Loss(criterion),
        'accuracy': Accuracy(transform_pred_tar),
        'domain accuracy': Accuracy(transform_d)
    }
    
    trainer = DATrainer(model, optimizer,
                                checkpoint_dir='../checkpoints',
                                metrics=metrics,
                                non_blocking=True,
                                retain_graph=True,
                                patience=5,
                                loss_fn=criterion,
                                device=DEVICE)
    trainer.fit(train_loader, dev_loader, epochs=10)
