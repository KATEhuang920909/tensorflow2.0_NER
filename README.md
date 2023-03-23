lattice-lstm  字粒度和词粒度相结合

flat 相对位置编码，减少引入词语信息的损失

|     模型名称     |      |
| :--------------: | ---: |
|    bilstm-crf    |      |
|  bilstm-stt-crf  |      |
| bert-bilstm-crf  |      |
|    idcnn-crf     |      |
|   Lattice LSTM   |      |
| BERT-cascade-CRF |      |
|       flat       |      |
|     biaffine     |      |
|  global-pointer  |      |
|     ner-mrc      |      |

## 版本问题：

1.train.py脚本 模块中对应tf版本2.11

```python
clsnermodel.build(input_shape={"token_id": [batch_size, maxlen],
                                    "segment_id": [batch_size, maxlen],
                                    "ner_label": [batch_size, maxlen],
                                    "cls_label": [batch_size, len(labels)]})
```

低版本如2.3不支持输入dic， 可以写为如下：

```python
clsnermodel.build(input_shape=[[batch_size, maxlen],
                               [batch_size, maxlen],
                               [batch_size, maxlen],
                               [batch_size, len(labels)]])
```

同样将输入格式转换成对应格式。



2. data_helper.py脚本

```python
tf.data.Dataset.from_tensor_slices
```

tf2.11如下：

```python
return {"token_id": batch_token_ids,
        "segment_id": batch_segment_ids,
        "ner_label": ner_batch_labels,
        "cls_label": cls_batch_labels}
```



在低版本中需要将输入转化为元组：

```python
return ( batch_token_ids,batch_segment_ids,ner_batch_labels,cls_batch_labels)
```

