5#encoding:utf-8
from collections import  Counter
import tensorflow.contrib.keras as kr
import numpy as np
import codecs
import re
import jieba

def read_file(filename):
    """
    Args:
        filename:trian_filename,test_filename,val_filename 
    Returns:
        two list where the first is lables and the second is contents cut by jieba
        
    """
    re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)")  # the method of cutting text by punctuation
    contents,labels=[],[]
    jieba.load_userdict("newdict.txt") 
    with codecs.open(filename, 'r', encoding='utf-8') as f:
        for _,line in enumerate(f):
            try:
                line=line.strip()
                line=line.split('\t')
                labels.append(line[1])
                blocks=re_han.split(line[2])
                word=[]
                for blk in blocks:
                    if re_han.match(blk):
                        word.extend(jieba.lcut(blk))
                    contents.append(word) 
            except:
                pass
    return labels,contents#返回句子标签与分词结果

def build_vocab(filenames,vocab_dir,vocab_size=8000):
    """
    Args:
        filename:trian_filename,test_filename,val_filename
        vocab_dir:path of vocab_filename
        vocab_size:number of vocabulary
    Returns:
        writting vocab to vocab_filename

    """
    all_data = []
    for filename in filenames:
        _,data_train=read_file(filename)
        for content in data_train:
            all_data.extend(content)
    counter=Counter(all_data)
    count_pairs=counter.most_common(vocab_size-1)
    words,_=list(zip(*count_pairs))
    words=['<PAD>']+list(words)

    with codecs.open(vocab_dir,'w',encoding='utf-8') as f:
        f.write('\n'.join(words)+'\n')

def read_vocab(vocab_dir):
    """
    Args:
        filename:path of vocab_filename
    Returns:
        words: a list of vocab
        word_to_id: a dict of word to id
        
    """
    words=codecs.open(vocab_dir,'r',encoding='utf-8').read().strip().split('\n')
    word_to_id=dict(zip(words,range(len(words))))
    return words,word_to_id

def read_category():
    """
    Args:
        None
    Returns:
        categories: a list of label
        cat_to_id: a dict of label to id

    """
    categories = ['高','低']
    cat_to_id={'高': 0, '低': 1}
    return categories,cat_to_id

def process_file(filename,word_to_id,cat_to_id,max_length=140):#处理训练集
    """
    Args:
        filename:train_filename or test_filename or val_filename
        word_to_id:get from def read_vocab()
        cat_to_id:get from def read_category()
        max_length:allow max length of sentence 
    Returns:
        x_pad: sequence data from  preprocessing sentence 
        y_pad: sequence data from preprocessing label

    """
    labels,contents=read_file(filename)
    data_id,label_id=[],[]
    for i in range(0,9900):
        
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        if labels[i]=="高":
            label_id.append("0")
        else:
            label_id.append("1")
            
    x_pad=kr.preprocessing.sequence.pad_sequences(data_id,max_length,padding='post', truncating='post')#使句子序列长度相等，不足max_length的部分补0
    y_pad=kr.utils.to_categorical(label_id)
    return x_pad,y_pad
def process_file2(filename,word_to_id,cat_to_id,max_length=140):#处理验证集
    """
    Args:
        filename:train_filename or test_filename or val_filename
        word_to_id:get from def read_vocab()
        cat_to_id:get from def read_category()
        max_length:allow max length of sentence 
    Returns:
        x_pad: sequence data from  preprocessing sentence 
        y_pad: sequence data from preprocessing label

    """
    labels,contents=read_file(filename)
    data_id,label_id=[],[]
    for i in range(0,1100):
        
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        if labels[i]=="高":
            label_id.append("0")
        else:
            label_id.append("1")
            
    x_pad=kr.preprocessing.sequence.pad_sequences(data_id,max_length,padding='post', truncating='post')
    y_pad=kr.utils.to_categorical(label_id)
    return x_pad,y_pad
def process_file3(filename,word_to_id,cat_to_id,max_length=140):#处理测试集
    """
    Args:
        filename:train_filename or test_filename or val_filename
        word_to_id:get from def read_vocab()
        cat_to_id:get from def read_category()
        max_length:allow max length of sentence 
    Returns:
        x_pad: sequence data from  preprocessing sentence 
        y_pad: sequence data from preprocessing label

    """
    labels,contents=read_file(filename)
    data_id,label_id=[],[]
    for i in range(0,4000):
        
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        if labels[i]=="高":
            label_id.append("0")
        else:
            label_id.append("1")
            
    x_pad=kr.preprocessing.sequence.pad_sequences(data_id,max_length,padding='post', truncating='post')
    y_pad=kr.utils.to_categorical(label_id)
    return x_pad,y_pad
def batch_iter(x,y,batch_size=64):
    """
    Args:
        x: x_pad get from def process_file()
        y:y_pad get from def process_file()
    Yield:
        input_x,input_y by batch size

    """

    data_len=len(x)
    num_batch=int((data_len-1)/batch_size)+1

    indices=np.random.permutation(np.arange(data_len))
    x_shuffle=x[indices]
    y_shuffle=y[indices]

    for i in range(num_batch):
        start_id=i*batch_size
        end_id=min((i+1)*batch_size,data_len)
        yield x_shuffle[start_id:end_id],y_shuffle[start_id:end_id]

def export_word2vec_vectors(vocab, word2vec_dir,trimmed_filename):
    """
    Args:
        vocab: word_to_id 
        word2vec_dir:file path of have trained word vector by word2vec
        trimmed_filename:file path of changing word_vector to numpy file
    Returns:
        save vocab_vector to numpy file
        
    """
    file_r = codecs.open(word2vec_dir, 'r', encoding='utf-8')
    line = file_r.readline()
    voc_size, vec_dim = map(int, line.split(' '))
    embeddings = np.zeros([len(vocab), vec_dim])
    line = file_r.readline()
    while line:
        try:
            items = line.split(' ')
            word = items[0]
            vec = np.asarray(items[1:], dtype='float32')
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(vec)
        except:
            pass
        line = file_r.readline()
    np.savez_compressed(trimmed_filename, embeddings=embeddings)

def get_training_word2vec_vectors(filename):
    """
    Args:
        filename:numpy file
    Returns:
        data["embeddings"]: a matrix of vocab vector
    """
    with np.load(filename) as data:
        return data["embeddings"]
