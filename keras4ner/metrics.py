# -*- coding: utf-8 -*-
# @Time    : 2023/2/16 17:16
# @Author  : huangkai
# @File    : metrics.py
import keras.backend as K


class METRICS():
    def __init__(self, num_class=2):
        self.num_class = num_class

    def f1_marco(self, y_true, y_pred):
        epsilon = K.epsilon
        shapes = K.shape(y_pred)
        y_true = K.reshape(y_true, shapes[:-1])
        y_pred = K.cast(K.argmax(y_pred, 2), "int32")
        f1_marco = 0

        ones_, zeros_ = K.ones(shapes), K.zeros(shapes)

        for i in range(self.num_class):
            tp = K.sum(K.switch((y_pred == i) & (y_true == i), ones_, zeros_))
            fp = K.sum(K.switch((y_pred == i) & (y_true != i), ones_, zeros_))
            fn = K.sum(K.switch((y_pred != i) & (y_true == i), ones_, zeros_))
            p = tp / (tp + fp + epsilon)
            r = tp / (tp + fn + epsilon)
            f1 = 2 * p * r / (p + r + epsilon)
            f1_marco += f1
        return f1_marco / self.num_class

# class F1Marco(tf.keras.metrics.Metric):
#     def __init__(self, num_class=2):
#         super().__init__()
#         self.num_class = num_class
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         epsilon = K.epsilon
#         shapes = K.shape(y_pred)
#         y_true = K.reshape(y_true, shapes[:-1])
#         y_pred = K.cast(K.argmax(y_pred, 2), "int32")
#         self.f1_marco = 0
#
#         ones_, zeros_ = K.ones(shapes), K.zeros(shapes)
#
#         for i in range(self.num_class):
#             tp = K.sum(K.switch((y_pred == i) & (y_true == i), ones_, zeros_))
#             fp = K.sum(K.switch((y_pred == i) & (y_true != i), ones_, zeros_))
#             fn = K.sum(K.switch((y_pred != i) & (y_true == i), ones_, zeros_))
#             p = tp / (tp + fp + epsilon)
#             r = tp / (tp + fn + epsilon)
#             f1 = 2 * p * r / (p + r + epsilon)
#             self.f1_marco += f1
#
#     def result(self):
#         return self.f1_marco / self.num_class
