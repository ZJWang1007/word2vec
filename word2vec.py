


from zhconv import convert
import re
import numpy as np
import jieba
from time import time

def vocab_construct(filename, training_len):
    vocab = set()
    texts = []
    i = 0
    for line in open(filename, 'r', encoding='utf8'):
        line_ = ''.join(re.findall('[\u4e00-\u9fa5]', line))
        line_ = convert(line_, 'zh-hans')
        text_ = [word for word in jieba.cut(line_, cut_all=False)]
        texts.append(text_)
        vocab = vocab.union(set(text_))
        i += 1
        if i%10==0:
            print("vocab-stage: %d completed while %d left!"%(i, training_len-i))
        if i==training_len:
            break
    return vocab, texts

def softmax(x):
    e = np.exp(x)
    return e/np.sum(e)

def train():
    global W_in, W_out
    t = time()
    for i, text in enumerate(texts):
        dW_out = np.zeros((embedding_size, vocab_size))
        dW_in  = np.zeros((vocab_size, embedding_size))
        Loss = 0
        for j in range(window_size, len(text)-window_size):
            x_idx = [word2index[text[jj+j]] for jj in range(-window_size, window_size+1)]
            del x_idx[window_size]
            x = np.zeros((1, vocab_size))
            h = np.zeros((1, embedding_size))
            for k in x_idx:
                x[0][k] += 1
                h += W_in[k]
            x /= (2*window_size)
            h /= (2*window_size)
            y = softmax(np.dot(h, W_out))
            Loss += (-np.log(y[0][word2index[text[j]]]))
            for k in x_idx:
                y[0][k] -= x[0][k]
            dW_out += np.dot(h.T, y) #y here is y-x
            tmp = np.dot(y, W_out.T)
            for k in x_idx:
                dW_in[k] += tmp[0]*x[0][k]
        
        W_in -= (learning_rate*dW_in)
        W_out -= (learning_rate*dW_out)
        Loss /= len(text)
        if i%10==0:
            print("training-stage: %d completed while %d left! Time Cost: %.5f"%(i, training_len-i, time()-t))
            t = time()
        print("Loss is ", Loss) 

####if __name__ == '__main__':

filename = 'corpus_cn33w.txt'
training_len = 30

vocab, texts = vocab_construct(filename, training_len)
word2index = {w:i for i, w in enumerate(vocab)}
index2word = {i:w for i, w in enumerate(vocab)}

print("Vocab construction completed, %d words in total"%(len(vocab)))
vocab_size = len(vocab)
embedding_size = 75
window_size = 2
learning_rate = 0.0006
W_in = np.random.rand(vocab_size, embedding_size)
W_out =np.random.rand(embedding_size, vocab_size)


print("Training begins")
train()

def n_near(target, n):
    if target not in vocab:
        print("Unsupported word")
        return 
    x = W_in[word2index[target]]
    res = np.dot(W_in, x.T)
    res = [[val, i] for i, val in enumerate(res)]
    res.sort(key = lambda s:(-s[0]))
    return [[index2word[res[i][1]], res[i][0]] for i in range(1, n+1)]

print(n_near('计算机', 15))
    
def model_save(outputfile):
    pass

