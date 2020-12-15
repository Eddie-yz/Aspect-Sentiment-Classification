import os
import gensim
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import *
from utils import read_vec, cosSim, softmax
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')

aspect_kw_enhance = {'location_n': ['street', 'parking', 'avenue', 'river', 'view'],
                    'location_adj': ['convenient', 'near'],
                    'drinks_n': ['drinks', 'beverage', 'wines', 'margarita', 'sake'],
                    'drinks_adj': ['alcoholic', 'iced', 'bottled'],
                    'food_n': ['food', 'pizza', 'tuna', 'sushi', 'burger'],
                    'food_adj': ['spicy', 'tasty', 'delicious', 'bland', 'savory'],
                    'ambience_n': ['atmosphere', 'room', 'decor', 'music', 'space'],
                    'ambience_adj': ['romantic', 'small', 'cozy', 'tiny'],
                    'service_n': ['tips', 'manager', 'wait', 'waitress', 'servers'],
                    'service_adj': ['rude', 'attentive', 'friendly'],
                    }

def calculate_topic_emb(aspect_kw, embedding_dict):
    as_topic_emb = dict()
    for asp, words in aspect_kw.items():
        asp_embs = list()
        for word in words:
            if not word in embedding_dict:
                continue
            vec = embedding_dict[word]
            asp_embs.append(vec.tolist())
        tmp_asp_emb = np.array(asp_embs).mean(axis=0)
        as_topic_emb[asp] = tmp_asp_emb / np.linalg.norm(tmp_asp_emb)

    as_topic_mat = np.array([as_topic_emb[aspect] for aspect in aspect_kw])
    return as_topic_mat

def evaluate(embedding_dict, as_topic_mat, current_kw, filename='test_score.txt', soft=False, use_center=True,
             enhance=False, thres=0.0, test_path='data/restaurant'):
    with open(os.path.join(test_path, 'test.txt')) as f:
        test_cont = f.readlines()

    asp_labels, senti_labels, docs = list(), list(), list()
    for line in test_cont:
        _, as_label, senti_label, doc = line.strip().split('\t')
        asp_labels.append(int(as_label))
        senti_labels.append(int(senti_label))
        docs.append(doc)
    emb_size = 200
    doc_embs = list()
    doc_weights= []
    doc_choices = []
    for doc in docs:
        doc_vec = np.zeros(emb_size)
        sen_weights = []
        sen_choices = [] 
        norm = 0
        for word in gensim.utils.simple_preprocess(doc):
            if word not in embedding_dict:
                continue
            word_vec = embedding_dict[word]
            norm_word_vec = word_vec / np.linalg.norm(word_vec)
            if use_center:
                product = np.dot(as_topic_mat, norm_word_vec.reshape(-1,1))
            else:
                product=[]
                for aspect in current_kw:
                    ind_score = [cosSim(embedding_dict[w], norm_word_vec) for w in current_kw[aspect]]
                    product.append(max(ind_score))
                product = np.array(product)
            if soft: 
                word_weight = np.max(softmax(product.reshape(-1)))* np.max(product)
            else:
                word_weight = np.max(product)
            norm += word_weight
            doc_vec += word_vec * word_weight
            word_choice = np.argmax(product)
            sen_weights.append(word_weight)
            sen_choices.append(word_choice)
        doc_embs.append(doc_vec/norm)
        doc_weights.append(sen_weights)
        doc_choices.append(sen_choices)

    doc_embs = np.array(doc_embs)
    norm_doc_embs = np.array([vec/np.linalg.norm(vec) for vec in doc_embs])
    as_scores = np.dot(norm_doc_embs, as_topic_mat.T)
    raw_labels = np.argmax(as_scores, axis=1)
    if enhance:
        as_pseudo_labels = np.array([sub2aspect[l] for l in raw_labels])
    else:
        as_pseudo_labels = raw_labels
    # subsample according to softmax confidence
    as_scores = np.array([softmax(score) for score in as_scores])
    confi_scores =  np.max(as_scores, axis=1)
    print(sum(confi_scores>thres))
    as_pseudo_labels_sub = as_pseudo_labels[confi_scores>thres]
    asp_labels_sub = np.array(asp_labels)[confi_scores>thres]
    
    print(confusion_matrix(asp_labels_sub, as_pseudo_labels_sub))
    acc = accuracy_score(asp_labels_sub, as_pseudo_labels_sub)
    p_mac = precision_score(asp_labels_sub, as_pseudo_labels_sub,average='macro')
    r_mac = recall_score(asp_labels_sub, as_pseudo_labels_sub, average='macro')
    f1_mac = f1_score(asp_labels_sub, as_pseudo_labels_sub, average='macro')
    p_mic = precision_score(asp_labels_sub, as_pseudo_labels_sub,average='micro')
    r_mic = recall_score(asp_labels_sub, as_pseudo_labels_sub, average='micro')
    f1_mic = f1_score(asp_labels_sub, as_pseudo_labels_sub, average='micro')
    print('Macro: acc {} | precision {} | recall {} | f1 {}'.format(acc, p_mac, r_mac, f1_mac))
    print('Micro: acc {} | precision {} | recall {} | f1 {}'.format(acc, p_mic, r_mic, f1_mic))
    with open(filename,'w') as f:
        for i in range(len(test_cont)):
            if thres >= 0.0:
                #if confi_scores[i]>thres:
                f.write(str(asp_labels[i])+'\t')
                f.write(str(as_pseudo_labels[i])+'\t')
                #f.write(str(confi_scores[i])+ '\t')
                if enhance:
                    partial_sum =0
                    for index, s in enumerate(as_scores[i]):
                        partial_sum +=s
                        if index%2 ==1:
                            f.write(str(partial_sum)+ '\t')
                            partial_sum =0
                else:
                    for s in as_scores[i]:
                        f.write(str(s)+ '\t')
                        
                f.write(docs[i]+'\n')
            else:
                if as_pseudo_labels[i] != asp_labels[i]:
                    f.write(test_cont[i])
                    for s in doc_weights[i]:
                        f.write(format(s, '.2f')+' ')
                    f.write('\n')
                    for c in doc_choices[i]:
                        f.write(str(c)+' ')
                    f.write('\n')
                    f.write(str(as_pseudo_labels[i])+'\n')
    return np.sum(as_pseudo_labels_sub == asp_labels_sub)/len(asp_labels_sub)



if __name__ == "__main__":
    wv = read_vec("embedding/wv.txt")
    topic_mat = calculate_topic_emb(aspect_kw_enhance, wv)

    ind2aspect = ['location', 'drinks', 'food', 'ambience', 'service']
    aspect2ind = {k:i for i,k in enumerate(ind2aspect)}
    sub2aspect = {i: i//2 for i in range(len(aspect_kw_enhance))}

    evaluate(wv, topic_mat, aspect_kw_enhance, filename='test_score.txt', soft=False, use_center=True,
         enhance=True, thres=0.0, test_path='data/restaurant')


    