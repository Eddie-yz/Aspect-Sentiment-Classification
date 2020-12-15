import numpy as np
import math

def read_vec(file, dim=200):
    with open(file) as f:
        embs = f.readlines()
    wv = dict()
    wv['unk']=np.zeros(dim)
    for line in embs[1:]:
        line = line.strip().split()
        word = line[0]
        vec = np.array([float(x) for x in line[1:]])
        wv[word] = vec
    return wv

def cosSim(vec1, vec2):
    return np.dot(vec1, vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec2))

def softmax(x):
    summ = sum(math.e**(xi) for xi in x if xi>0)
    y = [math.e**(xi)/summ if xi >0 else 0 for xi in x ]
    return y

def de_sentiment(adj_vec, senti_axis):
    projection = senti_axis * np.linalg.norm(adj_vec) * cosSim(adj_vec, senti_axis)
    return adj_vec - projection

def kl_divergence(p, q):
    p = np.array(p)
    q = np.array(q)
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def expand(aspect_kw=aspect_kw, k=10, thres=1.0,use_center=True, ite=False, wv=wv):
    aspect2ind = {k:i for i,k in enumerate(aspect_kw)}
    ind2aspect = {v:k for k,v in aspect2ind.items()}
    current_kw ={}
    for aspect in aspect_kw:
        current_kw[aspect]=aspect_kw[aspect].copy()
    topic_mat = calculate_topic_emb(aspect_kw, wv)
    topic_mat_update = topic_mat.copy()
    if ite:
        ran = k
        k = 1
    else:
        ran = 1
    for i in range(ran):
        dis_topic = defaultdict(dict)
        for asp in current_kw:
            for w in wv:
                if w not in current_kw[asp]:
                    asp_scores = [max(cosSim(topic_mat_update[aspect2ind[asp]],wv[w]),0) for asp in current_kw]
                    ordered = sorted(asp_scores,reverse=True)
                    if ind2aspect[np.argmax(asp_scores)]==asp and ordered[0]/(ordered[1]+1e-5)<thres:
                        dis_topic[asp][w]=0.0
                    else:
                        dis_topic[asp][w]=cosSim(topic_mat_update[aspect2ind[asp]],wv[w])
        '''
        for w in wv:
            if use_center:
                asp_scores = [max(cosSim(topic_mat[aspect2ind[asp]],wv[w]),0) for asp in current_kw]
            else:
                asp_scores=[]
                for aspect in current_kw:
                    ind_score = [cosSim(wv[key], wv[w]) for key in current_kw[aspect]]
                    asp_scores.append(max(ind_score))
            ordered = sorted(asp_scores,reverse=True)
            dis_topic[ind2aspect[np.argmax(asp_scores)]][w]=ordered[0]/(ordered[1]+1e-5) #ordered[0]-ordered[1])/ordered[0]
        '''
        for asp in current_kw:        
            top_words = sorted(dis_topic[asp],key=dis_topic[asp].get, reverse=True)[:k]
            current_kw[asp].extend(top_words)
        topic_mat_update = calculate_topic_emb(current_kw, wv)
    print(current_kw)
    for asp in current_kw:
        print(asp, cosSim(topic_mat[aspect2ind[asp]],topic_mat_update[aspect2ind[asp]]))
        '''
        for asp2 in current_kw:
            if asp != asp2:
                print('before',asp, asp2,cosSim(topic_mat[aspect2ind[asp]],topic_mat[aspect2ind[asp2]]))
                print('after',asp, asp2,cosSim(topic_mat_update[aspect2ind[asp]],topic_mat_update[aspect2ind[asp2]]))
        '''
    return current_kw, topic_mat_update