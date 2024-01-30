from transformers import AutoModel
from torchcrf import CRF
import torch
import torch.nn as nn



'''
# 12分类
class Bert_Classification(nn.Module):

    def __init__(self, num_classes, hidden_size=768, dropout_rate=0.2):
        super(Bert_Classification, self).__init__()

        self.bert = AutoModel.from_pretrained('D:\信息备份\训练权重\\bert\\bert-base-chinese')
        self.dropout = nn.Dropout(p=dropout_rate)
        self.linear = nn.Linear(hidden_size, num_classes)
        # self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.bert(x)[1]
        x = self.dropout(x)
        x = self.linear(x)
        # x = self.softmax(x)
        return x


        
class Bert_CRF(nn.Module):
    def __init__(self, num_tags, hidden_size=768, dropout_rate=0.2):
        super(Bert_CRF, self).__init__()

        self.num_tags = num_tags
        self.bert = AutoModel.from_pretrained('D:\信息备份\训练权重\\bert\\bert-base-chinese')
        self.dropout = nn.Dropout(p=dropout_rate)
        self.linear = nn.Linear(hidden_size, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def _get_features(self, sentence): 
        embeds = self.bert(sentence)[0]
        enc = self.dropout(embeds)
        feats = self.linear(enc)
        return feats

    def forward(self, x, tags, mask):
        emissions = self._get_features(x)
        loss = -self.crf.forward(emissions, tags, mask, reduction='mean')
        return loss

    def decode(self, x, mask):
        emissions = self._get_features(x)
        decode = self.crf.decode(emissions, mask)
        return decode
'''




class Bert_Together(nn.Module):
        def __init__(self, num_classes=12, num_tags=6, hidden_size=768, dropout_rate=0.2):
            super(Bert_Together, self).__init__()

            self.bert = AutoModel.from_pretrained('D:\信息备份\训练权重\\bert\\bert-base-chinese')
            # self.num_classes = num_classes
            # self.num_tags = num_tags
            self.dropout = nn.Dropout(p=dropout_rate)
            self.classifier = nn.Linear(hidden_size, num_classes)
            self.hidden_linear = nn.Linear(hidden_size, num_tags)
            self.crf = CRF(num_tags, batch_first=True)
            self.loss_fn = nn.CrossEntropyLoss()

        def _get_crf_features(self, sentence): 
            embeds = self.bert(sentence)[0]
            enc = self.dropout(embeds)
            feats = self.hidden_linear(enc)
            return feats
        
        def _get_classes_features(self, sentence): 
            embeds = self.bert(sentence)[1]
            enc = self.dropout(embeds)
            classifier_feats = self.classifier(enc)
            return classifier_feats

        def forward(self, x, labels, tags, mask):
            classifier_feats = self._get_classes_features(x)
            emissions = self._get_crf_features(x)
            loss_crf = -self.crf.forward(emissions, tags, mask, reduction='mean')
            loss_classifier = self.loss_fn(classifier_feats, labels)
            loss = loss_crf + loss_classifier
            return loss
        
        def predict(self, x, mask):
            emissions = self._get_crf_features(x)
            ids = self.crf.decode(emissions, mask)
            classifier_feats = self._get_classes_features(x)
            classes = torch.argmax(classifier_feats, dim=1)
            return ids, classes.data


# bert = AutoModel.from_pretrained('D:\信息备份\训练权重\\bert\\bert-base-chinese')
if __name__ == '__main__':
    '''
    net = Bert_Classification(12)
    test = torch.randint(0, 10000, (32, 64))
    output = net(test)
    print(output.shape)
    print(output)
    predict = torch.argmax(output, dim=1)
    print(predict)
    crf = CRF(2, batch_first=True)
    net = Bert_CRF(2)
    test = torch.randint(0, 10000, (2, 3)).long()
    print(test.shape)
    # mask = torch.randint(1, 5, (2, 3)).byte()
    mask = torch.ByteTensor([[1, 1, 1], [1, 1, 0]])
    print(mask.shape)
    labels = torch.randint(0, 2, (2, 3)).long()
    print(labels.shape)
    emissions = net._get_features(test)
    print(emissions.shape)
    loss = crf.forward(emissions, labels, mask)
    decode = crf.decode(emissions, mask)
    print(decode)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    sequence_size = 3
    num_labels = 5
    mask = torch.ByteTensor([[1, 1, 1], [1, 1, 0]]).to(device) # (batch_size. sequence_size)
    labels = torch.LongTensor([[0, 2, 3], [1, 4, 1]]).to(device)  # (batch_size, sequence_size)
    hidden = torch.randn((batch_size, sequence_size, num_labels), requires_grad=True).to(device)
    crf = CRF(num_labels).to(device)
    loss = crf.forward(hidden, labels, mask)
    print(loss)
    '''
    print(torch.__version__)