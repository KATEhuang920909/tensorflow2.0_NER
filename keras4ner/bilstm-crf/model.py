# -*- coding: utf-8 -*-
# @Time    : 2023/2/16 15:32
# @Author  : huangkai
# @File    : model.py
# ! -*- coding: utf-8 -*-
# 用bilstm-crf做中文命名实体识别
# 数据集 http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz
# paper: https://arxiv.org/pdf/1603.01360.pdf
# 实测验证集的F1可以到96.48%，测试集的F1可以到95.38%
import tensorflow as tf
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, ViterbiDecoder, to_array
from bert4keras.layers import ConditionalRandomField
from keras.layers import Input,  Dense, Embedding
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.layers import Bidirectional, LSTM
from keras.preprocessing import sequence
from tqdm import tqdm
from config import *
from metrics import METRICS
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
# bert配置
config_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/kg/bert/chinese_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    """加载数据
    单条格式：[text, (start, end, label), (start, end, label), ...]，
              意味着text[start:end + 1]是类型为label的实体。
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            d = ['']
            for i, c in enumerate(l.split('\n')):
                char, flag = c.split(' ')
                d[0] += char
                if flag[0] == 'B':
                    d.append([i, i, flag[2:]])
                    categories.add(flag[2:])
                elif flag[0] == 'I':
                    d[-1][1] = i
            D.append(d)
    return D


class NamedEntityRecognizer(ViterbiDecoder):
    """命名实体识别器
    """

    def recognize(self, text):
        tokens = tokenizer.tokenize(text, maxlen=512)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        nodes = model.predict([token_ids, segment_ids])[0]
        labels = self.decode(nodes)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], categories[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False
        return [(mapping[w[0]][0], mapping[w[-1]][-1], l) for w, l in entities]


def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data, ncols=100):
        R = set(NER.recognize(d[0]))
        T = set([tuple(i) for i in d[1:]])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """

    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        trans = K.eval(CRF.trans)
        NER.trans = trans
        f1, precision, recall = evaluate(valid_data)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('./best_model.weights')
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_labels = [], []
        for is_end, (token_ids,labels) in self.sample(random):
            batch_token_ids.append(token_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_labels = sequence_padding(batch_labels)
                yield batch_token_ids, batch_labels
                batch_token_ids, batch_labels = [], []

def create_dictionaries(model=None,
                        combined=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries

    '''
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),
                            allow_update=True)
        #  freqxiao10->0 所以k+1
        w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 所有频数超过10的词语的索引,(k->v)=>(v->k)
        w2vec = {word: model[word] for word in w2indx.keys()}  # 所有频数超过10的词语的词向量, (word->model(word))

        def parse_dataset(combined):  # 闭包-->临时使用
            ''' Words become integers
            '''
            data = []
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)  # freqxiao10->0
                data.append(new_txt)
            return data  # word=>index

        combined = parse_dataset(combined)
        combined = sequence.pad_sequences(combined, maxlen=maxlen)  # 每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec, combined
    else:
        print('No data provided...')

    # 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引


def word2vec_train(combined):
    model = Word2Vec(size=vocab_dim,
                     min_count=2,
                     window=5)
    model.build_vocab(combined)  # input: list
    model.train(combined, total_examples=model.corpus_count, epochs=model.iter)
    model.save('../model/Word2vec_model.pkl')
    index_dict, word_vectors, combined = create_dictionaries(model=model, combined=combined)
    return index_dict, word_vectors, combined

def get_data(index_dict, word_vectors, combined, y):
    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim))  # 初始化 索引为0的词语，词向量全为0
    for word, index in index_dict.items():  # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    # print x_train.shape,y_train.shape
    return n_symbols, embedding_weights, x_train, y_train, x_test, y_test

class BiLSTMCRF(tf.keras.Model):
    def __init__(self, num_classes=10, lstm_num=100, embedding_weights=None, n_symbols=0):
        super(BiLSTMCRF, self).__init__()

        self.embedding = Embedding(output_dim=vocab_dim,
                                   input_dim=n_symbols,
                                   mask_zero=True,
                                   weights=[embedding_weights],
                                   input_length=maxlen)  # Adding Input Length
        self.bilstm = Bidirectional(LSTM(lstm_num))
        self.d = Dense(num_classes)

    def call(self, x):
        x = self.embedding(x)
        x = self.bilstm(x)
        x = self.d(x)
        x = CRF(x)
        return self.d(x)


def training_model(inputs_shape, optimizer, metrics, loss, Epochs):
    train_generator = data_generator(train_data, batch_size)
    model = BiLSTMCRF()
    inputs = Input(shape=inputs_shape)
    outputs = model(inputs)
    model = Model(inputs, outputs)
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=[metrics])
    model.summary()
    # 这里是y_train是类别的整数id，不用转为one hot
    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=Epochs,
        callbacks=[evaluator]
    )
    return model


if __name__ == '__main__':
    # 标注数据

    CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
    loss = CRF.sparse_loss
    optimizer = Adam(0.001)
    metrics = METRICS.f1_marco
    categories = set()
    train_data = load_data('/root/ner/china-people-daily-ner-corpus/example.train')
    valid_data = load_data('/root/ner/china-people-daily-ner-corpus/example.dev')
    test_data = load_data('/root/ner/china-people-daily-ner-corpus/example.test')
    categories = list(sorted(categories))

    index_dict, word_vectors, combined = word2vec_train(combined)
    n_symbols, embedding_weights, x_train, y_train, x_test, y_test = get_data(index_dict, word_vectors, combined, y)


    NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])
    evaluator = Evaluator()
    inputs_shape = (maxlen,)
    model = training_model(inputs_shape, optimizer, metrics, loss, epochs)


