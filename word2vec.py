


from zhconv import convert
import re
import numpy as np
import jieba
from time import time

filename = 'corpus_cn33w_4.txt'

i = 0
training_len = 60
vocab = set()

for line in open(filename, 'r', encoding='utf8'):
    new_line = ''.join(re.findall('[\u4e00-\u9fa5]', line))
    new_line = convert(new_line, 'zh-hans')
    for w in jieba.cut(new_line, cut_all=False):
        vocab.add(w)
    i += 1
    if i%10==0:
        print("vocab-stage: %d completed while %d left!"%(i, training_len-i))
    if i==training_len:
        break

word2index = {w:i for i, w in enumerate(vocab)}
index2word = {i:w for i, w in enumerate(vocab)}

print("Vocab construction completed, %d words in total"%(len(vocab)))

vocab_size = len(vocab)
embedding_size = 75
window_size = 2
learning_rate = 0.001

W_in = np.random.rand(vocab_size, embedding_size)
W_out =np.random.rand(embedding_size, vocab_size)

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

print("Training begins")
i = 0
t = time()

for line in open(filename, 'r', encoding='utf8'):
    text = ''.join(re.findall('[\u4e00-\u9fa5]', line))
    text = convert(text, 'zh-hans')
    text = [w for w in jieba.cut(new_line, cut_all=False)]
    onehot = []
    dW_out = np.zeros((embedding_size, vocab_size))
    dW_in  = np.zeros((vocab_size, embedding_size))
    Loss = 0
    '''for w in text:
        xx = np.zeros((1, vocab_size))
        xx[0][word2index[w]] = 1  # one-hot encoding here!!!!
        onehot.append(xx)
        '''
    # Forward stage
    for j in range(window_size, len(text)-window_size):
        x = np.zeros((1, vocab_size))
        h = np.zeros((1, embedding_size))
        for jj in range(-window_size, window_size+1):
            if jj == 0:
                continue
            x[0][word2index[text[j+jj]]] += 1
            h += W_in[j+jj]
#            x += onehot[j+jj]
        x /= (2*window_size)
        h /= (2*window_size)
        #h = np.dot(x, W_in) # this is a slow way to calculate h
        y = softmax(np.dot(h, W_out))
        dW_out += np.dot(h.T, y-x)
        tmp = np.dot(y-x, W_out.T)
        for jj in range(-window_size, window_size+1):
            if jj == 0:
                continue
            dW_in[j+jj] += (tmp[0]/(2*window_size))
        #dW_in  += np.dot(x.T, np.dot(y-x, W_out.T)) # another slowing down 
        Loss += (-np.log(y[0][word2index[text[j]]]))
    
    # Backword stage
    W_in -= (learning_rate*dW_in)
    W_out -= (learning_rate*dW_out)
    Loss /= len(text)
    
    i += 1
    if i%10==0:
        print("training-stage: %d completed while %d left! Time Cost: %.5f"%(i, training_len-i, time()-t))
        t = time()
    if i==training_len:
        break
    print("Loss is ", Loss)  
        
def n_near(target, n):
    if target not in vocab:
        print("Unsupported word")
        return 
    x = W_in[word2index[target]]
    res = np.dot(W_in, x.T)
    res = [[val, i] for i, val in enumerate(res)]
    res.sort(key = lambda s:(-s[0]))
    return [[index2word[res[i][1]], res[i][0]] for i in range(1, n+1)]
    
def model_save(outputfile):
    pass


def f2():
    W = np.random.randn(15000, 300)
    t1 = time()
    res = np.zeros((1, 300))
    for i in [9788, 9789, 9791, 9792]:
        res += W[i]
    res /= 4
    return time()-t1

def f1():
    W = np.random.randn(15000, 300)
    t1 = time()
    x = np.zeros((1, 15000))
    for i in [9788, 9789, 9791, 9792]:
        x[0][i] += 1
    x /= 4
    res = np.dot(x, W)
    return time()-t1

