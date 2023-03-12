# -*- coding: utf-8 -*-
# @Time    : 2023/3/7 10:45
# @Author  : huangkai
# @File    : model.py
import gc

import numpy as np
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from tqdm.notebook import tqdm
from ...bert4keras.models import build_transformer_model
from ...bert4keras.layers import Dense, ConditionalRandomField
from ...bert4keras.snippets import ViterbiDecoder
from config import *
from ...utils.metrics import METRICS


# define model
class BERTCRF2Model(tf.keras.Model):

    def __init__(self, feat_dim, out_dim, num_linear):
        super(BERTCRF2Model, self).__init__()
        self.bert_model = build_transformer_model(config_path, checkpoint_path)
        self.CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)

    def call(self, inputs):
        output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)
        outputs = self.bert_model.get_layer(output_layer).output
        outputs = Dense(len(categories) * 2 + 1)(outputs)

        output = self.CRF(outputs) #logits
        return output


class Predictor(tf.keras.Model):

    def __init__(self, out_dim):
        super(Predictor, self).__init__()
        self.out_layer = tf.keras.layers.Dense(out_dim)

    def call(self, inputs):
        return self.out_layer(inputs)


class Trainer(tf.keras.Model,ViterbiDecoder):

    def __init__(self, model, num_class):
        super(Trainer, self).__init__()
        self.model = model
        self.eps = 1e-9
        self.metric=METRICS(num_class=num_class)

    def call(self, inputs):
        """
        # 计算loss、acc、f1值
        """
        nodes = self.model.predict(inputs[0])[0]
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

        loss_sum = 0.
        pos_true_positive = 0.
        pos_false_positive = 0.
        pos_false_negative = 0.
        neg_true_positive = 0.
        neg_false_positive = 0.
        neg_false_negative = 0.
        correct = 0.
        for i in range(3):
            pred_emb = tf.reduce_sum(tf.gather(unique_emb, inputs[f"history_{i}"]), axis=1)
            pred_val = tf.clip_by_value(tf.math.sigmoid(self.predictors[i](tf.nn.l2_normalize(pred_emb, axis=1))),
                                        self.eps, 1. - self.eps)
            # Binary Cross Entropy
            loss = -inputs[f"label_{i}"] * tf.math.log(pred_val) - (1. - inputs[f"label_{i}"]) * tf.math.log(
                1. - pred_val)
            loss_sum += tf.reduce_sum(tf.reduce_mean(loss, axis=0))

            # F1-macro
            pred_label = pred_val > .5
            bool_label = inputs[f"label_{i}"] == 1.
            correct += tf.cast(tf.math.count_nonzero(pred_label == bool_label), "float32")
            pos_true_positive += tf.cast(tf.math.count_nonzero(tf.math.logical_and(bool_label, pred_label)), "float32")
            pos_false_positive += tf.cast(
                tf.math.count_nonzero(tf.math.logical_and(tf.math.logical_not(bool_label), pred_label)), "float32")
            pos_false_negative += tf.cast(
                tf.math.count_nonzero(tf.math.logical_and(bool_label, tf.math.logical_not(pred_label))), "float32")

            pred_label = pred_val < .5
            bool_label = inputs[f"label_{i}"] == 0.
            neg_true_positive += tf.cast(tf.math.count_nonzero(tf.math.logical_and(bool_label, pred_label)), "float32")
            neg_false_positive += tf.cast(
                tf.math.count_nonzero(tf.math.logical_and(tf.math.logical_not(bool_label), pred_label)), "float32")
            neg_false_negative += tf.cast(
                tf.math.count_nonzero(tf.math.logical_and(bool_label, tf.math.logical_not(pred_label))), "float32")

        accuracy = correct / tf.cast(tf.shape(inputs[f"label_0"])[0] * 18, "float32")
        pos_recall = pos_true_positive / tf.maximum(self.eps, pos_true_positive + pos_false_negative)
        pos_precision = pos_true_positive / tf.maximum(self.eps, pos_true_positive + pos_false_positive)
        pos_f1 = 2 * pos_recall * pos_precision / tf.maximum(self.eps, pos_recall + pos_precision)

        neg_recall = neg_true_positive / tf.maximum(self.eps, neg_true_positive + neg_false_negative)
        neg_precision = neg_true_positive / tf.maximum(self.eps, neg_true_positive + neg_false_positive)
        neg_f1 = 2 * neg_recall * neg_precision / tf.maximum(self.eps, neg_recall + neg_precision)

        return loss_sum, (pos_f1 + neg_f1) / 2., accuracy

    def predict_proba(self, inputs):
        unique_emb = self.model(inputs["unique_feature"])
        labels = []
        pred_vals = []
        for i in range(3):
            pred_emb = tf.reduce_sum(tf.gather(unique_emb, inputs[f"history_{i}"]), axis=1)
            pred_val = tf.clip_by_value(tf.math.sigmoid(self.predictors[i](tf.nn.l2_normalize(pred_emb, axis=1))),
                                        self.eps, 1. - self.eps)
            pred_vals.append(pred_val)
            labels.append(inputs[f"label_{i}"])
        return tf.concat(pred_vals, axis=1), tf.concat(labels, axis=1)


@tf.function(experimental_relax_shapes=True)
def forward_step(batch_inputs):
    with tf.GradientTape() as tape:
        loss, f1, acc = trainer(batch_inputs, training=True)
    gradients = tape.gradient(loss, trainer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainer.trainable_variables))
    return loss, f1, acc


if __name__ == '__main__':

    # evaluate model
    from sklearn.model_selection import train_test_split

    # define model and dataset
    data_loader = DataLoader(train_histories, train_labels, unique_feature)
    emb_model = BERTCRF2Model(feature_num_vocabs, feat_dim=8, out_dim=32, num_cross=5, num_linear=0)
    plot_model(emb_model, to_file='BERT_BILSTM_CRF.png', show_shapes=True)

    predictors = [Predictor(3), Predictor(10), Predictor(5)]
    trainer = Trainer(emb_model, predictors)

    optimizer = tfa.optimizers.LazyAdam(learning_rate=1e-3)
    loss_metric = tf.keras.metrics.Mean()
    f1_metric = tf.keras.metrics.Mean()
    acc_metric = tf.keras.metrics.Mean()

    data_sessions = np.arange(num_session)
    train_sessions, val_sessions = train_test_split(data_sessions, test_size=2000, shuffle=True)
    train_dataset = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(train_sessions)) \
        .shuffle(num_session, reshuffle_each_iteration=True) \
        .batch(128) \
        .map(data_loader.call) \
        .prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(val_sessions)) \
        .shuffle(num_session, reshuffle_each_iteration=True) \
        .batch(128) \
        .map(data_loader.call) \
        .prefetch(tf.data.AUTOTUNE)

    with tf.device("CPU: 0"):
        best_f1_score = 0.0
        for epoch in range(10):
            with tqdm(total=len(train_dataset)) as pbar:
                for batch_inputs in train_dataset:
                    loss, f1, acc = forward_step(batch_inputs)
                    loss_metric(loss)
                    f1_metric(f1)
                    acc_metric(acc)
                    progress = {"BCE": loss_metric.result().numpy(), "f1": f1_metric.result().numpy(),
                                "accuracy": acc_metric.result().numpy()}
                    pbar.set_postfix(progress)
                    pbar.update(1)
            print("epoch:", epoch + 1)
            print("train:", loss_metric.result().numpy(), f1_metric.result().numpy())
            loss_metric.reset_states()
            f1_metric.reset_states()
            acc_metric(acc)

            with tqdm(total=len(val_dataset)) as pbar:
                final_loss, final_f1, final_acc = 0.0, 0.0, 0.0
                for i, batch_inputs in enumerate(val_dataset):
                    val_loss, val_f1, val_acc = forward_step(batch_inputs)
                    final_loss += val_loss
                    final_f1 += val_f1
                    final_acc += val_acc
                if final_f1 >= best_f1_score:
                    best_f1_score = final_f1
                    emb_model.save("/best_model")
            print("val:", final_loss / (i + 1), final_f1 / (i + 1), final_acc / (i + 1), "best_f1_score:",
                  best_f1_score)

            # val_loss, val_f1, val_acc = trainer(data_loader.call(val_sessions))
            # print("val:", val_loss.numpy(), val_f1.numpy(), val_acc.numpy())

    # embeddings = emb_model(unique_feature).numpy()
    #
    # del train_histories, train_label, train_labels
    # gc.collect()
    #
    # import jo_wilder
    #
    # env = jo_wilder.make_env()
    # iter_test = env.iter_test()
    #
    # # predict
    # eps = 1e-9
    # for (sample_submission, test) in iter_test:
    #     level_group = ['0-4', '5-12', '13-22'].index(test["level_group"].values[0])
    #     for col in feature_columns:
    #         test[col + "_ix"] = np.vectorize(lambda x: feature_dicts[col].get(x, 0))(
    #             test[col].fillna("nan_value").astype(str)).astype(np.int32)
    #
    #     # update embedding table
    #     test_feats = test[feature_ix_columns].merge(unique_train_df,
    #                                                 on=feature_ix_columns,
    #                                                 how="left")
    #     unseen_test_feats = test_feats[test_feats["unique_ix"].isna()].drop_duplicates(feature_ix_columns)[
    #         feature_ix_columns]
    #     new_embedding = emb_model(unseen_test_feats.values).numpy()
    #
    #     if new_embedding.shape[0]:
    #         unseen_test_feats["unique_ix"] = list(
    #             range(unique_train_df.shape[0], unique_train_df.shape[0] + new_embedding.shape[0]))
    #         unique_train_df = pd.concat([unique_train_df, unseen_test_feats], axis=0)
    #         embeddings = np.concatenate([embeddings, new_embedding], axis=0)
    #
    #     # predict
    #     test_feats = test[feature_ix_columns].merge(unique_train_df,
    #                                                 on=feature_ix_columns,
    #                                                 how="left")
    #     pred_emb = embeddings[test_feats["unique_ix"].values].sum(axis=0).reshape([1, -1])
    #     pred_emb /= np.maximum(eps, np.linalg.norm(pred_emb, ord=2))
    #     sample_submission["correct"] = \
    #         tf.cast(tf.math.sigmoid(predictors[level_group](pred_emb)) > threthold, "int32").numpy()[0]
    #
    #     env.predict(sample_submission)
    #
    # # check prediction
    #
    # df = pd.read_csv('submission.csv')
    # print(df.shape)
    # df.head(10)
    # print(df.correct.mean())
