import os
import torch

from torch.utils.data import Dataset
from transformers import *
from slp.util import mktensor


#DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def preprocessing (sequence, tokenizer):
    text = torch.tensor(tokenizer.encode(sequence, add_special_tokens=True, truncation=True, max_length=510), dtype=torch.long)
    return text

class AmazonZiser17(Dataset):
    def __init__(self, ds="books", dl=0, labeled=True, cldata=True):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.labels = []
        self.reviews = []
        self.domains = []
        self.transforms = []
        self.domain = dl 
        if labeled:
            labf = "all.txt"
        else:
            labf = "unl.txt"
        if cldata:
            file = os.path.join("../slpdata/amazon/cldata", ds, labf)
        else:
            file = os.path.join("../slpdata/amazon/ziser17", ds, labf)
        with open(file) as f:
            for row in f:
                if labf == "unl.txt":
                    label = -1
                    review = row[3:]
                else:
                    label, review = int(row[0]), row[2:]
                review = self.tokenizer.encode(review, add_special_tokens=True, max_length=510)
                if len(review)>511:
                   print(review)
                review=torch.tensor(review, dtype=torch.long)
                self.labels.append(label)
                self.reviews.append(review)
                self.domains.append(self.domain)

    def map(self, t):
        self.transforms.append(t)
        return self

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        label = self.labels[idx]
        domain = self.domains[idx]
        return review, label, domain

class NewLabelsData(Dataset):
    def __init__(self, old, indices, newlabels):
        self.old = old
        self.indices = indices
        self.labels = newlabels
        self.reviews = []
        for i in self.indices:
            r,_ = self.old[i]
            self.reviews.append(r)
        
    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        label = self.labels[idx]
        return review, label

    def augment(self, newreviews, newlabels):
        self.reviews = self.reviews + newreviews
        self.labels = self.labels + newlabels


if __name__ == '__main__':
    data = AmazonZiser17()
    indices = [100, 4, 18]
    newlab = [1, 0, 1]
    newdata = NewLabelsData(data, indices, newlab)
    import ipdb; ipdb.set_trace()
    for d in newdata:
        print(d)
    #for d in data:
    #    import ipdb; ipdb.set_trace()
    #    r,l = d
    #    print(l)

