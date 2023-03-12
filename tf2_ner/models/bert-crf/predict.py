# -*- coding: utf-8 -*-
# @Time    : 2023/3/12 20:08
# @Author  : huangkai
# @File    : predict.py



from model import BERTCRF2Model, Predictor,Trainer
import gc

emb_model = BERTCRF2Model(feature_num_vocabs, feat_dim=8, out_dim=32, num_cross=5, num_linear=0)


predictors = [Predictor(3), Predictor(10), Predictor(5)]
trainer = Trainer(emb_model, predictors)
trainer.load_weights("/best_model/best_model.weights")


embeddings = emb_model(unique_feature).numpy()

del train_histories, train_label, train_labels
gc.collect()

import jo_wilder

env = jo_wilder.make_env()
iter_test = env.iter_test()

# predict
eps = 1e-9
for (sample_submission, test) in iter_test:
    level_group = ['0-4', '5-12', '13-22'].index(test["level_group"].values[0])
    for col in feature_columns:
        test[col + "_ix"] = np.vectorize(lambda x: feature_dicts[col].get(x, 0))(
            test[col].fillna("nan_value").astype(str)).astype(np.int32)

    # update embedding table
    test_feats = test[feature_ix_columns].merge(unique_train_df,
                                                on=feature_ix_columns,
                                                how="left")
    unseen_test_feats = test_feats[test_feats["unique_ix"].isna()].drop_duplicates(feature_ix_columns)[
        feature_ix_columns]
    new_embedding = emb_model(unseen_test_feats.values).numpy()

    if new_embedding.shape[0]:
        unseen_test_feats["unique_ix"] = list(
            range(unique_train_df.shape[0], unique_train_df.shape[0] + new_embedding.shape[0]))
        unique_train_df = pd.concat([unique_train_df, unseen_test_feats], axis=0)
        embeddings = np.concatenate([embeddings, new_embedding], axis=0)

    # predict
    test_feats = test[feature_ix_columns].merge(unique_train_df,
                                                on=feature_ix_columns,
                                                how="left")
    pred_emb = embeddings[test_feats["unique_ix"].values].sum(axis=0).reshape([1, -1])
    pred_emb /= np.maximum(eps, np.linalg.norm(pred_emb, ord=2))
    sample_submission["correct"] = \
        tf.cast(tf.math.sigmoid(predictors[level_group](pred_emb)) > threthold, "int32").numpy()[0]

    env.predict(sample_submission)

# check prediction

df = pd.read_csv('submission.csv')
print(df.shape)
df.head(10)
print(df.correct.mean())
