from LSTM import RNN
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from torch.nn import functional as F
import numpy as np
import math
import torch
import os
import copy

def train(sub_train_, model, optimizer, scheduler, t):
    train_loss = 0
    train_acc = 0
    pseudo_aspect_train_acc = 0
    aspect_train_acc = 0
    data = DataLoader(sub_train_, batch_size=batch_size, shuffle=True, collate_fn=generate_batch)
    model.train()
    for i, (text, cls, gt,lengths) in enumerate(data):
        optimizer.zero_grad()
        cls = target_score(cls, t)       
        text, cls, gt = text.to(device), cls.to(device), gt.to(device)
        output = model(text, lengths)
        loss = kl_criterion(torch.log(F.softmax(output, dim=-1)), cls)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        pseudo_aspect_train_acc += (output.argmax(1) == cls.argmax(1)).sum().item()
        aspect_train_acc += (output.argmax(1) == gt).sum().item()
    scheduler.step()

    return train_loss / len(sub_train_), aspect_train_acc / len(sub_train_), pseudo_aspect_train_acc / len(sub_train_)

def test(data_, model):
    loss = 0
    acc = 0
    pseudo_aspect_test_acc = 0
    aspect_test_acc = 0
    data = DataLoader(data_, batch_size=128, collate_fn=generate_batch)
    pred_distribution = []
    model.eval()
    gts=[]
    for text, cls, gt,lengths in data:
        text, cls, gt = text.to(device), cls.to(device), gt.to(device)
        with torch.no_grad():
            output = model(text, lengths)
            cls = target_score(cls)
            loss = kl_criterion(torch.log(F.softmax(output, dim=-1)), cls)
            pseudo_aspect_test_acc += (output.argmax(1) == cls.argmax(1) ).sum().item()
            aspect_test_acc += (output.argmax(1)  == gt).sum().item()
            pred_distribution.append(F.softmax(output, dim=-1))
            gts.append(gt)
    pred_distribution = torch.cat(pred_distribution, dim=0)
    pred = pred_distribution.argmax(dim=1)
    gts = torch.cat(gts, dim=0) 
    
    p = precision_score(gts, pred,average='macro')
    r = recall_score(gts, pred, average='macro')
    f1_mac = f1_score(gts, pred, average='macro')
    p_w = precision_score(gts, pred,average='weighted')
    r_w = recall_score(gts, pred, average='weighted')
    f1_w = f1_score(gts, pred, average='weighted')
    print('*'*30)
    print('mac {:.5f} {:.5f} {:.5f}'.format(p, r, f1_mac))
    print('weighted {:.5f} {:.5f} {:.5f}'.format(p_w, r_w, f1_w))
            
    print('confusion matrix: ',confusion_matrix(gts, pred))
    return loss / len(data_), aspect_test_acc / len(data_), pseudo_aspect_test_acc / len(data_), pred_distribution


def generate_batch(batch):
    label = torch.cat([entry[1].unsqueeze(0) for entry in batch])
    text = []
    lengths = []
    for entry in batch:
        length = len(entry[0])
        lengths.append(length)
        tmp = F.pad(torch.tensor(entry[0]), (0,100-len(entry[0])), 'constant', 0).unsqueeze(0)
        text.append(tmp)
        for i in range(100):
            if tmp[0][i] >= len(wv):
                print(tmp[i])

    gt1 = torch.from_numpy(np.array([entry[2] for entry in batch]))
    
    text = torch.cat(text)

    return text, label, gt1,lengths

def read_vec(file):
    with open(file) as f:
        embs = f.readlines()
    wv = dict()
    wv['unk']=np.zeros(200)
    for line in embs[1:]:
        line = line.strip().split()
        word = line[0]
        vec = np.array([float(x) for x in line[1:]])
        wv[word] = vec
    return wv

def target_score(logits, t=1.0):
    weight = logits**t 
    return (weight.t() / torch.sum(weight, dim=1)).t()

def softmax(x):
    summ = sum(math.e**(xi) for xi in x if xi>0)
    y = [math.e**(xi)/summ if xi >0 else 0 for xi in x ]
    return y


if __name__ == "__main__":
    vec_file = "embedding/wv.txt" 
    wv = read_vec(vec_file)
    word2idx = {w: i for i,w in enumerate(wv)}
    idx2word = {i: w for i,w in enumerate(wv)}

    aspect_kw = {'location': ['street', 'block', 'avenue', 'river', 'convenient'],
                 'drinks': ['drinks', 'beverage', 'wines', 'margarita', 'sake'],
                 'food': ['food', 'spicy', 'sushi', 'pizza', 'tasty'],
                 'ambience': ['romantic', 'atmosphere', 'room', 'seating', 'small'],
                 'service': ['tips', 'manager', 'wait', 'waitress', 'servers'],
                 }
    
    learning_rate = 0.005
    batch_size = 16
    thres = 0.0
    output_size = len(aspect_kw)
    embedding_length = 200
    N_EPOCHS_PRE = 20
    N_EPOCHS = 10
    self_training = True
    restore_pretrain = False
    restore_selftrain = False
    file_name = 'test_score'
    label_file = '{}.txt'.format(file_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(label_file)) as f:
        test_cont = f.readlines()
        asp_gt, asp_labels, docs, scores = list(), list(), list() ,list()
        for line in test_cont:
            asp_gti, pseudo, confi0, confi1, confi2, confi3, confi4, doc = line.split('\t')
            asp_gt.append(int(asp_gti))
            asp_labels.append(int(pseudo))
            scores.append([float(confi0),float(confi1),float(confi2),float(confi3),float(confi4)])
            docs.append(doc)

    total_dataset_aspect = []
    for i, t in enumerate(docs):
        s_index = [word2idx[w] if w in wv else 0 for w in t.split(' ')]
        total_dataset_aspect.append([s_index, torch.tensor(scores[i]),asp_gt[i]])
    print('train dataset length',len(total_dataset_aspect))


    aspect_embedding = torch.zeros((len(wv), embedding_length))
    for i in range(len(wv)):
        aspect_embedding[i] = torch.tensor(wv[idx2word[i]])

    if restore_pretrain:
        pretrain_model = torch.load('{}_pretrain.pt'.format(file_name))
    else:
        pretrain_model = RNN(len(wv), embedding_length, embedding_length//2, output_size, 4, True,0.2, aspect_embedding)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    kl_criterion = torch.nn.KLDivLoss()
    pretrain_optimizer = torch.optim.Adam(pretrain_model.parameters(), lr=learning_rate)
    pretrain_scheduler = torch.optim.lr_scheduler.StepLR(pretrain_optimizer, 1, gamma=0.9)
    pretrain_model.to(device)
    
    # pretrain
    update_confi = None
    total_dist = None
    print('start pre-train ...')
    for epoch in range(N_EPOCHS_PRE):
        train_loss, aspect_train_acc,  pseudo_aspect_train_acc = \
        train(total_dataset_aspect, pretrain_model, pretrain_optimizer, pretrain_scheduler, t=1)
        _, aspect_test_acc_total, pseudo_aspect_test_acc_total, total_dist = test(total_dataset_aspect, pretrain_model)
        print('Train loss {:.7f}'.format(train_loss)) # aspect_test_acc, pseudo_aspect_test_acc)
        print('Total validation {:.3f}, total {:.3f}'.format(aspect_test_acc_total, pseudo_aspect_test_acc_total))
        update_confi, choice = torch.max(total_dist,axis=1)
        if epoch >0:
            label_change = (1 - torch.sum(last_choice == choice).item() / len(choice))*100
            print(epoch, label_change)
        last_choice = choice
        
    pretrain_choice = choice
    torch.save(pretrain_model,'{}_pretrain.pt'.format(file_name))
    
    # self-train
    if restore_selftrain:
        aspect_model = torch.load('{}_selftrain.pt'.format(file_name))
    else:
        aspect_model = copy.deepcopy(pretrain_model)
    last_choice = pretrain_choice
    aspect_lr = 0.0005
    temp = 1.2
    aspect_optimizer = torch.optim.Adam(aspect_model.parameters(), lr=aspect_lr)
    aspect_scheduler = torch.optim.lr_scheduler.StepLR(aspect_optimizer, 1, gamma=0.9)
    new_update_confi = update_confi
    update_index = torch.argsort(new_update_confi, descending=True)
    print('start self-train ...')
    for _ in range(N_EPOCHS):
        sub_dataset_aspect = []
        # reorder
        for i in update_index:
            t = docs[i]
            s_index = [word2idx[w] if w in wv else 0 for w in t.split(' ')]
            sub_dataset_aspect.append([s_index, total_dist[i],asp_gt[i]])   
        train_loss, aspect_train_acc,  pseudo_aspect_train_acc = train(sub_dataset_aspect, aspect_model, aspect_optimizer, aspect_scheduler,t=temp)
        _, aspect_test_acc_total, pseudo_aspect_test_acc_total, total_dist = test(total_dataset_aspect, aspect_model)
        print('Train loss', train_loss)#, aspect_test_acc, pseudo_aspect_test_acc)
        print('Total validation {:.5f}, label acc {:.5f}'.format(aspect_test_acc_total, pseudo_aspect_test_acc_total))
        new_update_confi, choice = torch.max(total_dist,axis=1)
        label_change = (1 - torch.sum(last_choice == choice).item() / len(choice))*100
        print(label_change)
        if label_change< 1.0:
            break
        last_choice = choice
        
    torch.save(aspect_model,'{}_selftrain.pt'.format(file_name))
