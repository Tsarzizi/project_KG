from transformers import AutoTokenizer
from funcation import *
import torch


def predict(text):
    with torch.no_grad():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = torch.load('./model/checkpoint/model_4.pt').to(device)
        tokenizer = AutoTokenizer.from_pretrained('D:/信息备份/训练权重/bert/bert-base-chinese')
        model.eval()
        encode = tokenizer.encode(text, max_length=64, padding='max_length', return_tensors='pt')
        encode = encode.to(device)
        mask = encode > 0
        mask = mask.byte()
        tag_ids, class_id = model.predict(encode, mask)
        word = id2word(tag_ids[0], text)
        class_id = class_id.cpu().detach().numpy()[0]
        class_ = class_id2class(class_id)
        return word, class_