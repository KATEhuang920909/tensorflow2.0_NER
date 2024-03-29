# -*- coding: utf-8 -*-
# @Time    : 2023/3/21 14:36
# @Author  : huangkai
# @File    : test.py
import tensorflow as tf


class MyModel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
        self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        if training:
            x = self.dropout(x, training=training)
        return self.dense2(x)


model = MyModel()
model.build(input_shape=(None, 128))
model.summary()
