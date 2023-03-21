# -*- coding: utf-8 -*-
# @Time    : 2023/3/12 20:08
# @Author  : huangkai
# @File    : train.py
import tensorflow as tf
from model import BERTCRF2Model, forward_step
from utils.data_helper import get_tag2index
from tqdm.notebook import tqdm
from config import *
from keras.utils.vis_utils import plot_model
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.tokenizers import Tokenizer
import numpy as np

categories = set()
# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def load_data(filename, is_test=False):
    """加载数据
    单条格式：[text, (start, end, label), (start, end, label), ...]，
              意味着text[start:end + 1]是类型为label的实体。
    """
    if is_test:
        with open(filename, encoding='utf-8') as f:
            data = f.readlines()
            data = [k.strip().split("\x01")[1] for k in data]
        return data if len(data) > 0 else None
    D = []
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            d = ['']
            for i, c in enumerate(l.split('\n')):
                if len(c.strip()) != 0:
                    char, flag = c.split(' ')
                    d[0] += char
                    if flag[0] == 'B':
                        d.append([i, i, flag[2:]])
                        categories.add(flag[2:])
                    elif flag[0] == 'E':
                        d[-1][1] = i
                    elif flag[0] == "O":
                        categories.add("others")

            D.append(d)
    return D


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, d in self.sample(random):
            tokens = tokenizer.tokenize(d[0], maxlen=maxlen)
            mapping = tokenizer.rematch(d[0], tokens)
            start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
            token_ids = tokenizer.tokens_to_ids(tokens)
            segment_ids = [0] * len(token_ids)
            labels = np.zeros(len(token_ids))
            for start, end, label in d[1:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    labels[start] = categories.index(label) * 2 + 1
                    labels[start + 1:end + 1] = categories.index(label) * 2 + 2
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# dp = DataProcess()
# train_data, train_label, val_data, val_label = dp.get_data(one_hot=False)
# train_dataset, val_dataset = load_dataset(train_data, train_label, val_data, val_label, batch_size=batch_size)
# 标注数据
if __name__ == '__main__':

    train_data = load_data('../../data/address/train.conll')
    valid_data = load_data('../../data/address/dev.conll')
    test_data = load_data('../../data/address/final_test.txt', is_test=True)
    categories = list(sorted(categories))
    exit()
    BertCrfmodel = BERTCRF2Model(num_classes=len(get_tag2index()))
    plot_model(BertCrfmodel, to_file='BERT_BILSTM_CRF.png', show_shapes=True)
    optimizer = tf.optimizers.Adam(learning_rate=1e-3)
    train_loss_metric = tf.keras.metrics.Mean()
    train_f1_metric = tf.keras.metrics.Mean()
    valid_loss_metric = tf.keras.metrics.Mean()
    valid_f1_metric = tf.keras.metrics.Mean()

    with tf.device("CPU: 0"):
        best_f1_score = 0.0
        for epoch in range(10):
            with tqdm(total=len(train_data)) as pbar:
                for batch_inputs in train_data:
                    loss, f1_score = forward_step(batch_inputs, BertCrfmodel, optimizer=optimizer)
                    train_loss_metric(loss)
                    train_f1_metric(f1_score)
                    progress = {"BCE": train_loss_metric.result().numpy(), "f1_score": train_f1_metric.result().numpy()}
                    pbar.set_postfix(progress)
                    pbar.update(1)
            print("epoch:", epoch + 1)
            print("train:", train_loss_metric.result().numpy(), train_f1_metric.result().numpy())
            train_loss_metric.reset_states()
            train_f1_metric.reset_states()

            with tqdm(total=len(valid_data)) as pbar:
                for i, batch_inputs in enumerate(valid_data):
                    val_loss, val_f1 = BertCrfmodel(batch_inputs)
                    valid_loss_metric(loss)
                    valid_f1_metric(f1_score)
            valid_f1 = valid_f1_metric.result().numpy()
            if valid_f1 >= best_f1_score:
                best_f1_score = valid_f1
                BertCrfmodel.save_weights("/best_model/best_model.weights")
            print("val:", valid_loss_metric.result().numpy(), valid_f1_metric.result().numpy(),
                  "best_f1_score:", best_f1_score)
