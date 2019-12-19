import os

from sklearn.preprocessing import LabelEncoder

from torch.utils.data import Dataset 

from slp.util.system import download_url, extract_tar

AMZ_URL = "http://www.cs.jhu.edu/~mdredze/datasets/sentiment/processed_acl.tar.gz"

class AmazonDataset(Dataset):
    def __init__(self, directory, domain, transforms=[]):
        self.domain = domain
        self.dl = 0 if domain == 'books' else 1
        self.directory = directory
        self.domain_dir_path = os.path.join(self.directory, "processed_acl/", self.domain, "")
        download_url(AMZ_URL, self.directory)
        extract_tar("processed_acl.tar.gz", self.directory)
        self.transforms = transforms
        self.texts, self.labels = self.get_data()
        self.label_encoder = (LabelEncoder().fit(self.labels))


    def read_line(self, line):
        words = line.split(' ')[:-1]
        label = line.split(' ')[-1].split(':')[1].strip()
        text = ""
        for wc in words:
            word, count = wc.split(':')
            for _ in range(int(count)):
                text = text + " " + word
        return text, label

    def get_data(self):
        reviews = []
        labels = []
        pos_path = os.path.join(self.domain_dir_path, 'positive.review')
        neg_path = os.path.join(self.domain_dir_path, 'negative.review')
        print (pos_path)
        with open(pos_path, encoding='utf-8') as f_pos, open(neg_path, encoding='utf-8') as f_neg:
            for pos_line, neg_line in zip(f_pos, f_neg):
                text, label = self.read_line(pos_line)
                reviews.append(text)
                labels.append(label)
                text, label = self.read_line(neg_line)
                reviews.append(text)
                labels.append(label)
        return reviews, labels
    
    def map(self, t):
        self.transforms.append(t)
        return self

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        label = self.label_encoder.transform([label])[0]
        for t in self.transforms:
            text = t(text)
        return text, label, self.dl
                 
if __name__ == '__main__':
    data = AmazonDataset('../../data/', 'books', [])
    for d in data:
        print(d)
