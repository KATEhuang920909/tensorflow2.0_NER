# -*- coding: utf-8 -*-
# @Time    : 2023/2/16 18:44
# @Author  : huangkai
# @File    : train_bak.py
# train_loss = tf.keras.metrics.Mean(name='train_loss')
# test_loss = tf.keras.metrics.Mean(name='test_loss')
# test_acc =F1Marco(name='test_acc')
#
# def train_step(x_train, y_train): # 求梯度利器
#     with tf.GradientTape() as tape:
#         logits = model(x_train)
#         loss = loss_obj(y_train, logits)
#
#     grads = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(grads, model.trainable_variables))
#
#     train_loss(loss)
#
# def test_step(x_test, y_test):
#     logits = model(x_test)
#     loss = loss_obj(y_test, logits)
#
#     test_loss(loss)
#     # test_acc(y_test, logits)
#
# for epoch in range(Epochs):
#     train_loss.reset_states()
#     test_loss.reset_states()
#     test_acc.reset_states()
#
#     for x_train, y_train in train_generator:
#         train_step(x_train, y_train)
#
#     for x_test, y_test in dev_generator:
#         test_step(x_test, y_test)
#
#
#     tmp = 'Epoch {}, train_loss: {}, test_loss: {}, test_acc: {}'
#     print (tmp.format(epoch+1, train_loss.result(), test_loss.result(), test_acc.result()))
