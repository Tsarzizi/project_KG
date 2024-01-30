import random
# import numpy as np
import pandas as pd
# #from transformers import AutoTokenizer

def find_X(s):
    l = 0
    r = len(s) - 1
    while 1:
        if s[l] != 'X':
            l += 1
        elif s[r] != 'X':
            r -= 1
        elif s[l] == 'X' and s[r] == 'X':
            break
    return(l, r)

def BIOES_template_maker(template):
    l, r = find_X(template)
    BIOES_template = ['O'] * len(template)
    BIOES_template[l: r+1] = ['X'] * (r - l + 1)
    BIOES_template = ''.join(BIOES_template)
    return BIOES_template

def templates_list_maker(file):
    with open(file, 'r', encoding='utf-8') as f:
        templates_list = []
        templates = []
        BIOES_templates = []
        BIOES_templates_list = []
        data = f.readlines()
        data = ''.join(data).split('\n')

        # print(data)
        for template in data:
            if template == '':
                templates_list.append(templates)
                BIOES_templates_list.append(BIOES_templates)
                templates = []
                BIOES_templates = []
            else:
                BIOES_template = BIOES_template_maker(template)
                BIOES_templates.append(BIOES_template)
                templates.append(template)
            
        # print(templates_list)
        # print(len(templates_list))
    return templates_list, BIOES_templates_list

def BIOES_maker(disease):
    label_len = len(disease)
    label = 'B' + 'I' * (label_len- 2) + 'E'
    return label

def question_maker(templates, BIOES_templates, file, specie):
    data = []
    label_C = []
    label_NER = []
    for i in range(len(templates)):
        templates[i] = templates[i].replace('XXX', '{}')
        BIOES_templates[i] = BIOES_templates[i].replace('XXX', '{}')
    with open(file, 'r', encoding='utf-8') as f:
        diseases = f.readlines()
        diseases = ''.join(diseases)
        diseases = diseases.split('\n')
    # return diseases
    for disease in diseases:
        choose = random.randint(0, len(templates)-1)
        question = templates[choose].format(disease)


        data.append(question)
        r = random.randint(1, 11)
        if r % 5 != 0:
            # print(specie)
            label_C.append(specie)
            BIOES = BIOES_maker(disease)
            BIOES = BIOES_templates[choose].format(BIOES)
            label_NER.append(BIOES)
        else:
            label_C.append(-1)
            label_NER.append(-1)
    data_csv = pd.DataFrame([data, label_C, label_NER])
    data_csv = data_csv.T
    data_csv.columns = ['text', 'label_c', 'label_ner']
    data_csv.to_csv(f'./data/{specie}.csv', index=False, encoding='utf_8_sig')

def batch_build_dataset(templates_list, BIOES_templates_list, species):
    for i in range(len(templates_list)):
        print(f'正在创造第{i}个数据集')
        question_maker(templates_list[i], BIOES_templates_list[i], './dict/disease.txt', species[i])
        print(f'创建完成')

def cat_data(csv_list):
    data = pd.DataFrame()
    for csv in csv_list:
        data1 = pd.read_csv(csv, encoding='utf_8_sig')
        data = pd.concat([data, data1])
    data.to_csv('./data.csv', index=False, encoding='utf_8_sig')

def data_maker():
    # species = ['symptom', 'cause', 'acompany', 'food', 'drug', 'prevent', 'lasttime', 'cureway', 'cureprob', 'easyget', 'check', 'belong']
    species = [0, 1, 2, 3, 4, 5, 6 ,7 ,8 ,9, 10, 11]
    templates_list, BIOES_templates_list = templates_list_maker('./data.txt')
    csv_list = ['./data/{}.csv'.format(specie) for specie in species]
    batch_build_dataset(templates_list, BIOES_templates_list, species)
    cat_data(csv_list)

def data_split(data):
# path = 'D:/信息备份/训练权重/bert/bert-base-chinese'
    # data = pd.read_csv(file, encoding='utf_8_sig')
    train = data.loc[data['label_c'] != -1]
    train_index = train.index.to_list()
    val = train.sample(frac=0.3)
    test = data[~data.index.isin(train_index)]
    train.to_csv('./train.csv', index=False, encoding='utf-8-sig')
    val.to_csv('./val.csv', index=False, encoding='utf-8-sig')
    test.to_csv('./test.csv', index=False, encoding='utf-8-sig')

'''
def train_ner_maker(data):
    texts = data['text']
    labels = data['label_ner']
    words = []
    BIOES = []
    for i in range(len(texts)):
        words.append('')
        BIOES.append('')
        for j in range(len(texts[i])):
            words.append(texts[i][j])
            BIOES.append(labels[i][j])
    train_ner = pd.DataFrame([words, BIOES])
    train_ner = train_ner.T
    train_ner.columns = ['word', 'label']
    train_ner.to_csv('train_ner.csv', index=False, encoding='utf-8-sig')

def test_ner_maker(data):
    texts = data['text']
    words = []
    for i in range(len(texts)):
        words.append('')
        for j in range(len(texts[i])):
            words.append(texts[i][j])
    test_ner = pd.DataFrame([words])
    test_ner = test_ner.T
    test_ner.columns = ['word']
    test_ner.to_csv('test_ner.csv', index=False, encoding='utf-8-sig')
'''
if __name__ == '__main__':
    pass
    # data = pd.read_csv('data.csv')
    # data_split(data)