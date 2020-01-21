import json
import os

from torch.utils.data import Dataset 

class AmazonDataset(Dataset):
    def __init__(self, directory, domain, transforms=[]):
        self.directory = directory
        self.domain = domain
        if self.domain == 'books':
                self.f = os.path.join(self.directory, "reviews_Books_5.json")
                self.d = 0
        elif self.domain == 'moovies':
                self.f = os.path.join(self.directory, "reviews_Movies_and_TV_5.json")
                self.d = 1
        self.transforms = transforms
        self.texts, self.ratings, self.labels, self.domains = self.get_data()

    def get_data(self):
        texts = []
        ratings = []
        labels = []
        domains = []
        data = []
        i = 0
        with open(self.f, 'r') as f:
            for line in f:
                data.append(json.loads(line))
                i = i + 1
                if i == 5000:
                   break
        #data = [json.loads(line) for line in open(self.f, 'r')]
        for i in data:
            d = self.d
            t = i['reviewText']
            r = i['overall']
            if r < 4:
                l = 0
            else:
                l = 1
            if (len(t)>0):
               domains.append(d)
               texts.append(t)
               ratings.append(r)
               labels.append(l)
        return texts, ratings, labels, domains

    
    def map(self, t):
        self.transforms.append(t)
        return self

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        domain = self.domains[idx]
        for t in self.transforms:
            text = t(text)
        return text, label, domain
                 
if __name__ == '__main__':
    data = AmazonDataset('../../data/', 'books', [])
    #print(data.__len__())
    for i in data:
        print(i)
