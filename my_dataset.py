
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from transformers import AutoTokenizer
from funcation import *
'''

class Bert_Classification_Dataset(Dataset):
    def __init__(self, path, mode='train'):
        super().__init__()
        self.data = pd.read_csv(path)
        self.texts = self.data['text'].values
        self.tokenizer = AutoTokenizer.from_pretrained('D:/信息备份/训练权重/bert/bert-base-chinese')
        self.mode = mode
        if mode != 'test':
            self.labels = self.data['label_c'].values



    def __getitem__(self, index):
        if self.mode != 'test':
            data = self.texts[index]
            label = self.labels[index]
            data = self.tokenizer.encode(data, max_length=64, padding='max_length', return_tensors='pt')
            label = torch.tensor(label)
            return data.squeeze(), label
        else:
            data = self.texts[index]
            data = self.tokenizer.encode(data, max_length=64, padding='max_length', return_tensors='pt')
            return data.squeeze()


    def __len__(self):
        return self.data.shape[0]
        

class Bert_NER(Dataset):
    def __init__(self, path, max_length):
        super().__init__()
        self.data = pd.read_csv(path)
        self.texts = self.data['text'].values
        self.tokenizer = AutoTokenizer.from_pretrained('D:/信息备份/训练权重/bert/bert-base-chinese')
        self.labels = self.data['label_ner'].values
        self.max_length = max_length


    def __getitem__(self, index):
        data = self.texts[index]
        label = self.labels[index]
        data = self.tokenizer.encode(data, max_length=self.max_length, padding='max_length', return_tensors='pt').squeeze()
        label = tag2id(label)
        label = torch.tensor([4] + label + [5] + [0] * (self.max_length - len(label) - 2))
        mask = data > 0
        return data, label, mask.byte()



    def __len__(self):
        return self.data.shape[0]
'''


class Bert_Together_Dataset(Dataset):
    
    def __init__(self, path, max_length):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('D:/信息备份/训练权重/bert/bert-base-chinese')
        self.data = pd.read_csv(path)
        self.texts = self.data['text'].values
        self.labels = self.data['label_c'].values
        self.tags = self.data['label_ner'].values
        self.max_length = max_length

    def __getitem__(self, index):
        data = self.texts[index]
        data = self.tokenizer.encode(data, max_length=self.max_length, padding='max_length', return_tensors='pt').squeeze()

        label = self.labels[index]
        label = torch.tensor(label)

        tag = self.tags[index]
        tag_id = tag2id(tag)
        tag_id = torch.tensor([4] + tag_id + [5] + [0] * (self.max_length - len(tag_id) - 2))

        mask = data > 0
        mask = mask.byte()
        return data, label, tag_id, mask



    def __len__(self):
        return self.data.shape[0]





if __name__ == '__main__':
    '''
    # data = pd.read_csv('train.csv')
    trainset = Bert_Classification_Dataset('train.csv', 'train')
    print(trainset[0])
    valset = Bert_Classification_Dataset('val.csv', 'val')
    print(valset[0])
    mydataloader = DataLoader(batch_size=32, dataset=trainset, shuffle=True, drop_last=True)

    tokenizer = AutoTokenizer.from_pretrained('D:/信息备份/训练权重/bert/bert-base-chinese')
    data = pd.read_csv('train.csv')
    texts = data['text']
    labels = data['label_ner']
    label = labels[0]
    text = texts[0]
    encode = tokenizer.encode(text, max_length=64, padding='max_length', return_tensors='pt').squeeze()
    mask = encode > 0
    print(encode.shape)
    print(mask.shape)
    print(mask)
    '''
    trainset = Bert_Together_Dataset('./model/data/train.csv', 64)
    print(trainset[0][0].shape)
    print(trainset[0][1].shape)
    print(trainset[0][2].shape)
    print(trainset[0][3].shape)
    mydataloader = DataLoader(batch_size=32, dataset=trainset, shuffle=True, drop_last=True)