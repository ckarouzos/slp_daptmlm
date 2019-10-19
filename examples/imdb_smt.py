import torch
import torch.nn as nn

from ignite.metrics import Loss, Accuracy
from sklearn.preprocessing import LabelEncoder

from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from torchnlp.datasets import imdb_dataset  # type: ignore
from torchnlp.datasets import smt_dataset  # type: ignore

from slp.data.collators import SequenceClassificationCollator
from slp.data.transforms import SpacyTokenizer, ToTokenIds, ToTensor
from slp.modules.daclassifier import DAClassifier, DALoss
from slp.modules.rnn import WordRNN
from slp.trainer import DATrainer
from slp.util.embeddings import EmbeddingsLoader


class DatasetWrapper(Dataset):
    def __init__(self, dataset, label):
        self.dataset = dataset
        self.label = label
        self.transforms = []
        self.label_encoder = (LabelEncoder()
                              .fit([d[self.label] for d in dataset]))

    def map(self, t):
        self.transforms.append(t)
        return self

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        datum = self.dataset[idx]
        text, target = datum['text'], datum[self.label]
        target = self.label_encoder.transform([target])[0]
        for t in self.transforms:
            text = t(text)
        return text, target


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

collate_fn = SequenceClassificationCollator(device='cpu')


if __name__ == '__main__':
    loader = EmbeddingsLoader(
        '../cache/glove.840B.300d.txt', 300)
    word2idx, _, embeddings = loader.load()

    tokenizer = SpacyTokenizer()
    to_token_ids = ToTokenIds(word2idx)
    to_tensor = ToTensor(device='cpu')

    def create_dataloader(d, label):
        d = (DatasetWrapper(d, label).map(tokenizer).map(to_token_ids).map(to_tensor))
        return DataLoader(
            d, batch_size=32,
            num_workers=1,
            pin_memory=True,
            shuffle=True,
            collate_fn=collate_fn)

    train_loader1, dev_loader1 = map(
        create_dataloader,
        imdb_dataset(directory='../data/', train=True, test=True),
        'sentiment')
    
    train_loader2, dev_loader2 = map(
        create_dataloader,
        smt_dataset(directory='../data/', train=True, dev=True),
        'label')
    
    model = DAClassifier(
        WordRNN(256, embeddings, bidirectional=True, merge_bi='cat',
                packed_sequence=True, attention=True, device=DEVICE),
        512, 3, 2)

    optimizer = Adam([p for p in model.parameters() if p.requires_grad],
                     lr=1e-3)
    
    criterion = DALoss(nn.CrossEntropyLoss(), nn.CrossEntropyLoss())
    #TODO 
    metrics = {
        'accuracy': Accuracy(),
        'loss': Loss(criterion)
    }
    
    trainer = DATrainer(model, optimizer,
                                checkpoint_dir='../checkpoints',
                                metrics=metrics,
                                non_blocking=True,
                                retain_graph=True,
                                patience=5,
                                loss_fn=criterion,
                                device=DEVICE)
    trainer.fit(train_loader1, dev_loader1, epochs=10)
