# -*- coding: utf-8 -*-
# @Time    : 2023/3/12 20:22
# @Author  : huangkai
# @File    : data_helper.py
import tensorflow as tf
import numpy as np
import sys
sys.path.append("../../utils")

# define dataset
# class DataLoader():
#
#     def __init__(self, train_histories, train_labels, train_unique_feataures):
#         self.train_histories = train_histories
#         self.train_labels = train_labels
#         self.train_unique_feataures = train_unique_feataures
#
#     def call(self, inputs):
#         session_ix = inputs
#         raw_histories = [tf.gather(self.train_histories[i], session_ix) for i in range(3)]
#         historiy_lengths = [tf.shape(h.values)[0] for h in raw_histories]
#         unique_ix, unique_idx = tf.unique(tf.concat([h.values for h in raw_histories], axis=0))
#         history_values = tf.split(unique_idx, historiy_lengths, axis=0)
#         histories = [tf.RaggedTensor.from_value_rowids(values=history_values[i],
#                                                        value_rowids=raw_histories[i].value_rowids(),
#                                                        nrows=raw_histories[i].nrows()) for i in range(3)]
#         inputs = {}
#         for i in range(3):
#             inputs[f"history_{i}"] = histories[i]
#             label = tf.gather(self.train_labels[i], session_ix)
#             inputs[f"label_{i}"] = label
#         inputs["unique_feature"] = tf.gather(self.train_unique_feataures, unique_ix)
#         return inputs


# 读取数据
def read_file(file_path: str) -> [str]:
    with open(file_path, 'r') as f:
        texts = f.read().split('\n')
    return texts


# 文本映射
def text_map(texts: [str]) -> [str]:
    """
    文本映射处理
    处理好的数据格式:
       ['需 O'
        '要 O'
        '大 B-ORG'
        '连 I-ORG'
        '海 I-ORG'
        '富 I-ORG'
        '集 I-ORG'
        '团 I-ORG']

    :param texts:  例如 中/B_nt 共/M_nt 中/M_nt 央/E_nt 总/O  的文本
    :return: [str] 处理好的数据
    """
    mapping = {'O': 'O',
               'B_nr': 'B-PER',
               'M_nr': 'I-PER',
               'E_nr': 'I-PER',
               'B_ns': 'B-LOC',
               'M_ns': 'I-LOC',
               'E_ns': 'I-LOC',
               'B_nt': 'B-ORG',
               'M_nt': 'I-ORG',
               'E_nt': 'I-ORG'
               }
    deal_texts = []
    for line in texts:
        sub_line = str(line).split(' ')
        for item in sub_line:
            item_list = str(item).split('/')
            if len(item_list) == 2:
                a = item_list[0]
                b = item_list[1]
                flag = mapping.get(b, 'O')
                deal_texts.append(f"{a} {flag}\n")
        deal_texts.append('\n')
    return deal_texts


# 数据预处理
def renminribao_preprocessing(split_rate: float = 0.8,
                              ignore_exist: bool = False) -> None:
    """
    人名日报数据预处理
    :param split_rate:数据分割比例，默认为 0.8
    :param ignore_exist 忽略已经存储的数据，则不需要在判断以及存好的数据
    :return:None
    """
    path_train = os.path.join(path_renmin_dir, "train.txt")
    path_test = os.path.join(path_renmin_dir, "test.txt")
    if not ignore_exist and os.path.exists(path_train) and os.path.exists(path_test):
        print("人民日报数据预处理已经完成")
        return
    else:
        print("正在对人民日报数据进行预处理......")
    path_org = os.path.join(path_renmin_dir, "renmin3.txt")
    texts = read_file(path_org)

    # 对数据进行分割处理
    if split_rate >= 1.0:  # 分割比例不得大于1.0
        split_rate = 0.8
    split_index = int(len(texts) * split_rate)
    train_texts = texts[:split_index]
    test_texts = texts[split_index:]

    # 对数据进行映射处理
    test_texts = text_map(test_texts)
    train_texts = text_map(train_texts)

    # 数据写入到本地文件
    with open(path_train, 'w') as f:
        f.write("".join(train_texts))
    with open(path_test, 'w') as f:
        f.write("".join(test_texts))
    print("人民日报数据进行预处理完成 ---- OK!")


from global_config import *
#
unk_flag = '[UNK]'
pad_flag = '[PAD]'
cls_flag = '[CLS]'
sep_flag = '[SEP]'


# 获取 word to index 词典
def get_w2i(vocab_path=path_vocab):
    w2i = {}
    with open(vocab_path, 'r', encoding="utf8") as f:
        while True:
            text = f.readline()
            if not text:
                break
            text = text.strip()
            if text and len(text) > 0:
                w2i[text] = len(w2i) + 1
    return w2i


# 获取 tag to index 词典
def get_tag2index()->dict:
    return {"O": 0,
            "B-PER": 1, "I-PER": 2,
            "B-LOC": 3, "I-LOC": 4,
            "B-ORG": 5, "I-ORG": 6
            }


class DataProcess(object):
    def __init__(self,
                 max_len=100,
                 data_type='renmin',  # 'data', 'data2', 'msra', 'renmin'
                 model='other',  # 'other'、'bert' bert 数据处理需要单独进行处理
                 ):
        """
        数据处理
        :param max_len: 句子最长的长度，默认为保留100
        :param data_type: 数据类型，当前支持四种数据类型
        """
        self.w2i = get_w2i()  # word to index
        self.tag2index = get_tag2index()  # tag to index
        self.vocab_size = len(self.w2i)
        self.tag_size = len(self.tag2index)
        self.unk_flag = unk_flag
        self.pad_flag = pad_flag
        self.max_len = max_len
        self.model = model

        self.unk_index = self.w2i.get(unk_flag, 101)
        self.pad_index = self.w2i.get(pad_flag, 1)
        self.cls_index = self.w2i.get(cls_flag, 102)
        self.sep_index = self.w2i.get(sep_flag, 103)

        if data_type == 'renmin':
            self.base_dir = path_renmin_dir
            renminribao_preprocessing()
        else:
            raise RuntimeError('type must be "data", "msra", "renmin" or "data2"')

    def get_data(self, one_hot: bool = True) -> ([], [], [], []):
        """
        获取数据，包括训练、测试数据中的数据和标签
        :param one_hot:
        :return:
        """
        # 拼接地址
        path_train = os.path.join(self.base_dir, "train.txt")
        path_test = os.path.join(self.base_dir, "test.txt")

        # 读取数据
        if self.model == 'bert':
            train_data, train_label = self.__bert_text_to_index(path_train)
            test_data, test_label = self.__bert_text_to_index(path_test)
        else:
            train_data, train_label = self.__text_to_indexs(path_train)
            test_data, test_label = self.__text_to_indexs(path_test)

        # 进行 one-hot处理
        if one_hot:
            def label_to_one_hot(index: []) -> []:
                data = []
                for line in index:
                    data_line = []
                    for i, index in enumerate(line):
                        line_line = [0] * self.tag_size
                        line_line[index] = 1
                        data_line.append(line_line)
                    data.append(data_line)
                return np.array(data)

            train_label = label_to_one_hot(index=train_label)
            test_label = label_to_one_hot(index=test_label)
        else:
            train_label = np.expand_dims(train_label, 2)
            test_label = np.expand_dims(test_label, 2)
        return train_data, train_label, test_data, test_label

    def num2tag(self):
        return dict(zip(self.tag2index.values(), self.tag2index.keys()))

    def i2w(self):
        return dict(zip(self.w2i.values(), self.w2i.keys()))

    # texts 转化为 index序列
    def __text_to_indexs(self, file_path: str) -> ([], []):
        data, label = [], []
        with open(file_path, 'r') as f:
            line_data, line_label = [], []
            for line in f:
                if line != '\n':
                    w, t = line.split()
                    char_index = self.w2i.get(w, self.w2i[self.unk_flag])
                    tag_index = self.tag2index.get(t, 0)
                    line_data.append(char_index)
                    line_label.append(tag_index)
                else:
                    if len(line_data) < self.max_len:
                        pad_num = self.max_len - len(line_data)
                        line_data = [self.pad_index] * pad_num + line_data
                        line_label = [0] * pad_num + line_label
                    else:
                        line_data = line_data[:self.max_len]
                        line_label = line_label[:self.max_len]
                    data.append(line_data)
                    label.append(line_label)
                    line_data, line_label = [], []
        return np.array(data), np.array(label)

    def __bert_text_to_index(self, file_path: str):
        """
        bert的数据处理
        处理流程 所有句子开始添加 [CLS] 结束添加 [SEP]
        bert需要输入 ids和types所以需要两个同时输出
        由于我们句子都是单句的，所以所有types都填充0

        :param file_path:  文件路径
        :return: [ids, types], label_ids
        """
        data_ids = []
        data_types = []
        label_ids = []
        with open(file_path, 'r') as f:
            line_data_ids = []
            line_data_types = []
            line_label = []
            for line in f:
                if line != '\n':
                    w, t = line.split()
                    # bert 需要输入index和types 由于我们这边都是只有一句的，所以type都为0
                    w_index = self.w2i.get(w, self.unk_index)
                    t_index = self.tag2index.get(t, 0)
                    line_data_ids.append(w_index)  # index
                    line_data_types.append(0)  # types
                    line_label.append(t_index)  # label index
                else:
                    # 处理填充开始和结尾 bert 输入语句每个开始需要填充[CLS] 结束[SEP]
                    max_len_buff = self.max_len - 2
                    if len(line_data_ids) > max_len_buff:  # 先进行截断
                        line_data_ids = line_data_ids[:max_len_buff]
                        line_data_types = line_data_types[:max_len_buff]
                        line_label = line_label[:max_len_buff]
                    line_data_ids = [self.cls_index] + line_data_ids + [self.sep_index]
                    line_data_types = [0] + line_data_types + [0]
                    line_label = [0] + line_label + [0]

                    # padding
                    if len(line_data_ids) < self.max_len:  # 填充到最大长度
                        pad_num = self.max_len - len(line_data_ids)
                        line_data_ids = [self.pad_index] * pad_num + line_data_ids
                        line_data_types = [0] * pad_num + line_data_types
                        line_label = [0] * pad_num + line_label
                    data_ids.append(np.array(line_data_ids))
                    data_types.append(np.array(line_data_types))
                    label_ids.append(np.array(line_label))
                    line_data_ids = []
                    line_data_types = []
                    line_label = []
        return [np.array(data_ids), np.array(data_types)], np.array(label_ids)


def preprocess(images,labels):
    '''
    最简单的预处理函数:
        转numpy为Tensor、分类问题需要处理label为one_hot编码、处理训练数据
    '''
    # 把numpy数据转为Tensor
    labels = tf.cast(labels, dtype=tf.int32)
    # labels 转为one_hot编码
    labels = tf.one_hot(labels, depth=10)
    # 顺手归一化
    images = tf.cast(images, dtype=tf.float32) / 255
    return labels, images


def load_dataset(train_data, valid_data, batch_size):
    db_train = tf.data.Dataset.from_tensor_slices(train_data) \
        .shuffle(1000, reshuffle_each_iteration=True) \
        .batch(batch_size) \
        .prefetch(tf.data.experimental.AUTOTUNE)

    db_test = tf.data.Dataset.from_tensor_slices(valid_data) \
        .shuffle(1000, reshuffle_each_iteration=True) \
        .batch(batch_size) \
        .prefetch(tf.data.experimental.AUTOTUNE)
    return db_train, db_test


if __name__ == '__main__':
    # dp = DataProcess(data_type='data')
    # x_train, y_train, x_test, y_test = dp.get_data(one_hot=True)
    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)
    #
    # print(y_train[:1, :1, :100])

    dp = DataProcess(data_type='data', model='bert')
    x_train, y_train, x_test, y_test = dp.get_data(one_hot=True)
    print(x_train[0].shape)
    print(x_train[1].shape)
    print(y_train.shape)
    print(x_test[0].shape)
    print(x_test[1].shape)
    print(y_test.shape)

    print(y_train[:1, :1, :100])

    pass
