# -*- coding: utf-8 -*-
# @Time    : 2023/3/21 19:29
# @Author  : huangkai
# @File    : attention.py
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    """

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super().__init__(**kwargs)

    def build(self, input_shape):  # 可训练权重
        self.QKV = self.add_weight(name='QKV',
                                   shape=(3, input_shape[2], self.output_dim),
                                   initializer='uniform',
                                   regularizer=tf.keras.regularizers.L1L2,
                                   trainable=True)
        super(SelfAttention, self).build(input_shape)

    def call(self, x):  # forward
        WQ = tf.matmul(x, self.QKV[0])
        WK = tf.matmul(x, self.QKV[1])
        WV = tf.matmul(x, self.QKV[2])
        QK = tf.matmul(WQ, tf.transpose(WK, [0, 2, 1]))
        QK = QK / (tf.math.sqrt(tf.cast(x.shape[-1],dtype="float32")))
        QK = tf.math.softmax(QK)
        V = tf.matmul(QK, WV)
        return V

    def compute_output_shape(self, input_shape):  # 输出shape
        return (input_shape[0], input_shape[1], self.output_dim)


class Attention(tf.keras.layers.Layer):
    """
    """

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super().__init__(**kwargs)

    def build(self, input_shape):  # 可训练权重
        self.attention_weight = self.add_weight(name='attention_weight',
                                                shape=(input_shape[2], 1),
                                                initializer='uniform',
                                                regularizer=tf.keras.regularizers.L1L2,
                                                trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):  # forward
        WQ = tf.matmul(x, self.attention_weight)
        WK = tf.math.softmax(WQ)
        V = tf.reduce_sum(WQ * WK, axis=-1)
        return V

    def compute_output_shape(self, input_shape):  # 输出shape
        return input_shape
