from model import Bert_Together
from my_dataset import Bert_Together_Dataset
from torch.utils.data import DataLoader
import torch 
import numpy as np
from tqdm import tqdm


'''
epochs = 5
learning_rate = 0.000001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = torch.nn.CrossEntropyLoss().to(device)


train_loss = []
valid_loss = []
train_epochs_loss = []
valid_epochs_loss = []


bc = Bert_Classification(12).to(device)
optimizer_bc = torch.optim.Adam(bc.parameters(),lr=learning_rate)
lr_scheduler_bc = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_bc, T_max=epochs, eta_min=0)
train_set_bc = Bert_Classification_Dataset('train.csv', mode='train')
train_dataloader_bc = DataLoader(batch_size=256, dataset=train_set_bc, shuffle=True, drop_last=True)
val_set_bc = Bert_Classification_Dataset('val.csv', mode='val')
val_dataloader_bc = DataLoader(batch_size=256, dataset=val_set_bc, shuffle=True, drop_last=True)

ner = Bert_CRF(6).to(device)
optimizer_ner = torch.optim.Adam(ner.parameters(),lr=learning_rate)
lr_scheduler_ner = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ner, T_max=epochs, eta_min=0)
train_set_ner = Bert_NER('train.csv', 64)
train_dataloader_ner = DataLoader(batch_size=256, dataset=train_set_ner, shuffle=True, drop_last=True)
val_set_ner = Bert_NER('val.csv', 64)
val_dataloader_ner = DataLoader(batch_size=256, dataset=val_set_ner, shuffle=True, drop_last=True)



def train_bert_classifier(net, optimizer, lr_scheduler, criterion ,device, epochs, train_dataloader, val_dataloader):
    for epoch in range(epochs):
        net.train()
        train_epoch_loss = []
        for idx,(data_x,data_y) in tqdm(enumerate(train_dataloader), total =len(train_dataloader)):
            data_x = data_x.to(device)
            data_y = data_y.to(device)
            outputs = net(data_x)
            optimizer.zero_grad()
            loss = criterion(outputs, data_y)
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
            train_loss.append(loss.item())
            if idx % (len(train_dataloader))==0:
                print("epoch={}/{},{}/{} of train, loss={}".format(
                    epoch+1, epochs, idx, len(train_dataloader),loss.item()))
        train_epochs_loss.append(np.average(train_epoch_loss))
        lr_scheduler.step()

    evaluate = Evaluate()
    f1_thresholds = 0.8
    with torch.no_grad():
        net.eval()
        for idx,(data_x,data_y) in tqdm(enumerate(val_dataloader), total =len(val_dataloader)):
            data_x = data_x.to(device)
            data_y = data_y.to(device)
            outputs = net(data_x)
            optimizer.zero_grad()
            loss = criterion(outputs, data_y)

            optimizer.step()
            # train_epoch_loss.append(loss.item())
            train_loss.append(loss.item())

        # train_epochs_loss.append(np.average(train_epoch_loss))

            predicts = torch.argmax(outputs, dim=1)
            predicts = predicts.cpu().detach().numpy()
            data_y = data_y.cpu().detach().numpy()
            evaluate.add_batch(data_y, predicts)
    confusion, accuracy, precision, recall, F1_score = evaluate.confusion_matrix()
    if F1_score > f1_thresholds:
        torch.save(net, './checkpoint/net_{}.pt'.format(F1_score))
    # evaluate.auc_roc()

def train_ner(net, optimizer, lr_scheduler ,device, epochs, train_dataloader, val_dataloader):
    for epoch in range(epochs):
        net.train()
        train_epoch_loss = []
        for idx,(data_x, data_y, mask) in tqdm(enumerate(train_dataloader), total =len(train_dataloader)):
            data_x = data_x.to(device)
            data_y = data_y.to(device)
            mask = mask.to(device)
            loss = net(data_x, data_y, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
            train_loss.append(loss.item())
            if idx % (len(train_dataloader))==0:
                print("epoch={}/{},{}/{} of train, loss={}".format(
                    epoch+1, epochs, idx, len(train_dataloader),loss.item()))
        train_epochs_loss.append(np.average(train_epoch_loss))
        lr_scheduler.step()
        torch.save(net, './checkpoint/net_{}.pt'.format(epoch))

    evaluate = Evaluate()
    f1_thresholds = 0.8
    with torch.no_grad():
        net.eval()
        for idx,(data_x,data_y) in tqdm(enumerate(val_dataloader), total =len(val_dataloader)):
            data_x = data_x.to(device)
            data_y = data_y.to(device)
            loss = net(data_x)
            optimizer.zero_grad()

            optimizer.step()
            # train_epoch_loss.append(loss.item())
            train_loss.append(loss.item())

        # train_epochs_loss.append(np.average(train_epoch_loss))
    
            predicts = torch.argmax(outputs, dim=1)
            predicts = predicts.cpu().detach().numpy()
            data_y = data_y.cpu().detach().numpy()
            evaluate.add_batch(data_y, predicts)
    confusion, accuracy, precision, recall, F1_score = evaluate.confusion_matrix()
    if F1_score > f1_thresholds:
        torch.save(net, './checkpoint/net_{}.pt'.format(F1_score))
    '''

train_loss = []
train_epochs_loss = []

epochs = 5
learning_rate = 0.000001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Bert_Together().to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
train_set = Bert_Together_Dataset('./data/train.csv', 64)
train_dataloader = DataLoader(batch_size=128, dataset=train_set, shuffle=True, drop_last=True)


def train():
    for epoch in range(epochs):
        model.train()
        train_epoch_loss = []
        for idx,(data, label, tag, mask) in tqdm(enumerate(train_dataloader), total =len(train_dataloader)):
            data = data.to(device)
            # print(data)
            label = label.to(device)
            # print(label)
            tag = tag.to(device)
            # print(tag)
            mask = mask.to(device)
            # print(mask)
            loss = model(data, label, tag, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
            train_loss.append(loss.item())
            if idx % (len(train_dataloader))==0:
                print("epoch={}/{},{}/{} of train, loss={}".format(
                    epoch+1, epochs, idx, len(train_dataloader),loss.item()))
        train_epochs_loss.append(np.average(train_epoch_loss))
        lr_scheduler.step()
        torch.save(model, './checkpoint/model_{}.pt'.format(epoch))


# train_ner(ner, optimizer_ner, lr_scheduler_ner, device, epochs, train_dataloader_ner, val_dataloader_ner)
train()