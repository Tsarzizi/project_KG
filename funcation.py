def id2word(ids, text):
    l = ids.index(1)
    r = ids.index(3)
    word = text[l - 1: r]
    return word

def tag2id(tag):
    dic = {'B': 1, 'I': 2, 'E': 3,'O': 0}
    ids = []
    for c in tag:
        ids.append(dic[c])
    return ids

def class_id2class(class_id):
    species = ['symptom', 'cause', 'acompany', 'food', 'drug', 'prevent', 'lasttime', 'cureway', 'cureprob', 'easyget', 'check', 'belong']
    species_id = [0, 1, 2, 3, 4, 5, 6 ,7 ,8 ,9, 10, 11]
    dic = dict(zip(species_id, species))
    class_ = dic[class_id]
    return class_