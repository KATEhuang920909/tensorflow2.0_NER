#! -*- coding: utf-8 -*-
# 将cloudwalk的预训练bart模型转换为bert4torch可用的权重
# 权重链接百度云地址：

import torch

ckpt_file = 'E:/pretrain_ckpt/bart/[cloudwalk_torch_base]/pytorch_base_model_2024000.pt'
torch_weights = torch.load(ckpt_file)

map = {'bart.embeddings.word_embeddings.weight': 'encoder.embed_tokens.weight',
 'bart.embeddings.position_embeddings.weight': 'encoder.embed_positions.weight',
 'bart.embeddings.LayerNorm.weight': 'encoder.layernorm_embedding.weight',
 'bart.embeddings.LayerNorm.bias': 'encoder.layernorm_embedding.bias',
 'bart.encoder.encoder_layer.0.attention.self.query.weight': 'encoder.layers.0.self_attn.q_proj.weight',
 'bart.encoder.encoder_layer.0.attention.self.query.bias': 'encoder.layers.0.self_attn.q_proj.bias',
 'bart.encoder.encoder_layer.0.attention.self.key.weight': 'encoder.layers.0.self_attn.k_proj.weight',
 'bart.encoder.encoder_layer.0.attention.self.key.bias': 'encoder.layers.0.self_attn.k_proj.bias',
 'bart.encoder.encoder_layer.0.attention.self.value.weight': 'encoder.layers.0.self_attn.v_proj.weight',
 'bart.encoder.encoder_layer.0.attention.self.value.bias': 'encoder.layers.0.self_attn.v_proj.bias',
 'bart.encoder.encoder_layer.0.attention.output.dense.weight': 'encoder.layers.0.self_attn.out_proj.weight',
 'bart.encoder.encoder_layer.0.attention.output.dense.bias': 'encoder.layers.0.self_attn.out_proj.bias',
 'bart.encoder.encoder_layer.0.attention.output.LayerNorm.weight': 'encoder.layers.0.self_attn_layer_norm.weight',
 'bart.encoder.encoder_layer.0.attention.output.LayerNorm.bias': 'encoder.layers.0.self_attn_layer_norm.bias',
 'bart.encoder.encoder_layer.0.intermediate.dense.weight': 'encoder.layers.0.fc1.weight',
 'bart.encoder.encoder_layer.0.intermediate.dense.bias': 'encoder.layers.0.fc1.bias',
 'bart.encoder.encoder_layer.0.output.dense.weight': 'encoder.layers.0.fc2.weight',
 'bart.encoder.encoder_layer.0.output.dense.bias': 'encoder.layers.0.fc2.bias',
 'bart.encoder.encoder_layer.0.output.LayerNorm.weight': 'encoder.layers.0.final_layer_norm.weight',
 'bart.encoder.encoder_layer.0.output.LayerNorm.bias': 'encoder.layers.0.final_layer_norm.bias',
 'bart.encoder.encoder_layer.1.attention.self.query.weight': 'encoder.layers.1.self_attn.q_proj.weight',
 'bart.encoder.encoder_layer.1.attention.self.query.bias': 'encoder.layers.1.self_attn.q_proj.bias',
 'bart.encoder.encoder_layer.1.attention.self.key.weight': 'encoder.layers.1.self_attn.k_proj.weight',
 'bart.encoder.encoder_layer.1.attention.self.key.bias': 'encoder.layers.1.self_attn.k_proj.bias',
 'bart.encoder.encoder_layer.1.attention.self.value.weight': 'encoder.layers.1.self_attn.v_proj.weight',
 'bart.encoder.encoder_layer.1.attention.self.value.bias': 'encoder.layers.1.self_attn.v_proj.bias',
 'bart.encoder.encoder_layer.1.attention.output.dense.weight': 'encoder.layers.1.self_attn.out_proj.weight',
 'bart.encoder.encoder_layer.1.attention.output.dense.bias': 'encoder.layers.1.self_attn.out_proj.bias',
 'bart.encoder.encoder_layer.1.attention.output.LayerNorm.weight': 'encoder.layers.1.self_attn_layer_norm.weight',
 'bart.encoder.encoder_layer.1.attention.output.LayerNorm.bias': 'encoder.layers.1.self_attn_layer_norm.bias',
 'bart.encoder.encoder_layer.1.intermediate.dense.weight': 'encoder.layers.1.fc1.weight',
 'bart.encoder.encoder_layer.1.intermediate.dense.bias': 'encoder.layers.1.fc1.bias',
 'bart.encoder.encoder_layer.1.output.dense.weight': 'encoder.layers.1.fc2.weight',
 'bart.encoder.encoder_layer.1.output.dense.bias': 'encoder.layers.1.fc2.bias',
 'bart.encoder.encoder_layer.1.output.LayerNorm.weight': 'encoder.layers.1.final_layer_norm.weight',
 'bart.encoder.encoder_layer.1.output.LayerNorm.bias': 'encoder.layers.1.final_layer_norm.bias',
 'bart.encoder.encoder_layer.2.attention.self.query.weight': 'encoder.layers.2.self_attn.q_proj.weight',
 'bart.encoder.encoder_layer.2.attention.self.query.bias': 'encoder.layers.2.self_attn.q_proj.bias',
 'bart.encoder.encoder_layer.2.attention.self.key.weight': 'encoder.layers.2.self_attn.k_proj.weight',
 'bart.encoder.encoder_layer.2.attention.self.key.bias': 'encoder.layers.2.self_attn.k_proj.bias',
 'bart.encoder.encoder_layer.2.attention.self.value.weight': 'encoder.layers.2.self_attn.v_proj.weight',
 'bart.encoder.encoder_layer.2.attention.self.value.bias': 'encoder.layers.2.self_attn.v_proj.bias',
 'bart.encoder.encoder_layer.2.attention.output.dense.weight': 'encoder.layers.2.self_attn.out_proj.weight',
 'bart.encoder.encoder_layer.2.attention.output.dense.bias': 'encoder.layers.2.self_attn.out_proj.bias',
 'bart.encoder.encoder_layer.2.attention.output.LayerNorm.weight': 'encoder.layers.2.self_attn_layer_norm.weight',
 'bart.encoder.encoder_layer.2.attention.output.LayerNorm.bias': 'encoder.layers.2.self_attn_layer_norm.bias',
 'bart.encoder.encoder_layer.2.intermediate.dense.weight': 'encoder.layers.2.fc1.weight',
 'bart.encoder.encoder_layer.2.intermediate.dense.bias': 'encoder.layers.2.fc1.bias',
 'bart.encoder.encoder_layer.2.output.dense.weight': 'encoder.layers.2.fc2.weight',
 'bart.encoder.encoder_layer.2.output.dense.bias': 'encoder.layers.2.fc2.bias',
 'bart.encoder.encoder_layer.2.output.LayerNorm.weight': 'encoder.layers.2.final_layer_norm.weight',
 'bart.encoder.encoder_layer.2.output.LayerNorm.bias': 'encoder.layers.2.final_layer_norm.bias',
 'bart.encoder.encoder_layer.3.attention.self.query.weight': 'encoder.layers.3.self_attn.q_proj.weight',
 'bart.encoder.encoder_layer.3.attention.self.query.bias': 'encoder.layers.3.self_attn.q_proj.bias',
 'bart.encoder.encoder_layer.3.attention.self.key.weight': 'encoder.layers.3.self_attn.k_proj.weight',
 'bart.encoder.encoder_layer.3.attention.self.key.bias': 'encoder.layers.3.self_attn.k_proj.bias',
 'bart.encoder.encoder_layer.3.attention.self.value.weight': 'encoder.layers.3.self_attn.v_proj.weight',
 'bart.encoder.encoder_layer.3.attention.self.value.bias': 'encoder.layers.3.self_attn.v_proj.bias',
 'bart.encoder.encoder_layer.3.attention.output.dense.weight': 'encoder.layers.3.self_attn.out_proj.weight',
 'bart.encoder.encoder_layer.3.attention.output.dense.bias': 'encoder.layers.3.self_attn.out_proj.bias',
 'bart.encoder.encoder_layer.3.attention.output.LayerNorm.weight': 'encoder.layers.3.self_attn_layer_norm.weight',
 'bart.encoder.encoder_layer.3.attention.output.LayerNorm.bias': 'encoder.layers.3.self_attn_layer_norm.bias',
 'bart.encoder.encoder_layer.3.intermediate.dense.weight': 'encoder.layers.3.fc1.weight',
 'bart.encoder.encoder_layer.3.intermediate.dense.bias': 'encoder.layers.3.fc1.bias',
 'bart.encoder.encoder_layer.3.output.dense.weight': 'encoder.layers.3.fc2.weight',
 'bart.encoder.encoder_layer.3.output.dense.bias': 'encoder.layers.3.fc2.bias',
 'bart.encoder.encoder_layer.3.output.LayerNorm.weight': 'encoder.layers.3.final_layer_norm.weight',
 'bart.encoder.encoder_layer.3.output.LayerNorm.bias': 'encoder.layers.3.final_layer_norm.bias',
 'bart.encoder.encoder_layer.4.attention.self.query.weight': 'encoder.layers.4.self_attn.q_proj.weight',
 'bart.encoder.encoder_layer.4.attention.self.query.bias': 'encoder.layers.4.self_attn.q_proj.bias',
 'bart.encoder.encoder_layer.4.attention.self.key.weight': 'encoder.layers.4.self_attn.k_proj.weight',
 'bart.encoder.encoder_layer.4.attention.self.key.bias': 'encoder.layers.4.self_attn.k_proj.bias',
 'bart.encoder.encoder_layer.4.attention.self.value.weight': 'encoder.layers.4.self_attn.v_proj.weight',
 'bart.encoder.encoder_layer.4.attention.self.value.bias': 'encoder.layers.4.self_attn.v_proj.bias',
 'bart.encoder.encoder_layer.4.attention.output.dense.weight': 'encoder.layers.4.self_attn.out_proj.weight',
 'bart.encoder.encoder_layer.4.attention.output.dense.bias': 'encoder.layers.4.self_attn.out_proj.bias',
 'bart.encoder.encoder_layer.4.attention.output.LayerNorm.weight': 'encoder.layers.4.self_attn_layer_norm.weight',
 'bart.encoder.encoder_layer.4.attention.output.LayerNorm.bias': 'encoder.layers.4.self_attn_layer_norm.bias',
 'bart.encoder.encoder_layer.4.intermediate.dense.weight': 'encoder.layers.4.fc1.weight',
 'bart.encoder.encoder_layer.4.intermediate.dense.bias': 'encoder.layers.4.fc1.bias',
 'bart.encoder.encoder_layer.4.output.dense.weight': 'encoder.layers.4.fc2.weight',
 'bart.encoder.encoder_layer.4.output.dense.bias': 'encoder.layers.4.fc2.bias',
 'bart.encoder.encoder_layer.4.output.LayerNorm.weight': 'encoder.layers.4.final_layer_norm.weight',
 'bart.encoder.encoder_layer.4.output.LayerNorm.bias': 'encoder.layers.4.final_layer_norm.bias',
 'bart.encoder.encoder_layer.5.attention.self.query.weight': 'encoder.layers.5.self_attn.q_proj.weight',
 'bart.encoder.encoder_layer.5.attention.self.query.bias': 'encoder.layers.5.self_attn.q_proj.bias',
 'bart.encoder.encoder_layer.5.attention.self.key.weight': 'encoder.layers.5.self_attn.k_proj.weight',
 'bart.encoder.encoder_layer.5.attention.self.key.bias': 'encoder.layers.5.self_attn.k_proj.bias',
 'bart.encoder.encoder_layer.5.attention.self.value.weight': 'encoder.layers.5.self_attn.v_proj.weight',
 'bart.encoder.encoder_layer.5.attention.self.value.bias': 'encoder.layers.5.self_attn.v_proj.bias',
 'bart.encoder.encoder_layer.5.attention.output.dense.weight': 'encoder.layers.5.self_attn.out_proj.weight',
 'bart.encoder.encoder_layer.5.attention.output.dense.bias': 'encoder.layers.5.self_attn.out_proj.bias',
 'bart.encoder.encoder_layer.5.attention.output.LayerNorm.weight': 'encoder.layers.5.self_attn_layer_norm.weight',
 'bart.encoder.encoder_layer.5.attention.output.LayerNorm.bias': 'encoder.layers.5.self_attn_layer_norm.bias',
 'bart.encoder.encoder_layer.5.intermediate.dense.weight': 'encoder.layers.5.fc1.weight',
 'bart.encoder.encoder_layer.5.intermediate.dense.bias': 'encoder.layers.5.fc1.bias',
 'bart.encoder.encoder_layer.5.output.dense.weight': 'encoder.layers.5.fc2.weight',
 'bart.encoder.encoder_layer.5.output.dense.bias': 'encoder.layers.5.fc2.bias',
 'bart.encoder.encoder_layer.5.output.LayerNorm.weight': 'encoder.layers.5.final_layer_norm.weight',
 'bart.encoder.encoder_layer.5.output.LayerNorm.bias': 'encoder.layers.5.final_layer_norm.bias',
 'bart.decoder.decoder_layer.0.attention.self.query.weight': 'decoder.layers.0.self_attn.q_proj.weight',
 'bart.decoder.decoder_layer.0.attention.self.query.bias': 'decoder.layers.0.self_attn.q_proj.bias',
 'bart.decoder.decoder_layer.0.attention.self.key.weight': 'decoder.layers.0.self_attn.k_proj.weight',
 'bart.decoder.decoder_layer.0.attention.self.key.bias': 'decoder.layers.0.self_attn.k_proj.bias',
 'bart.decoder.decoder_layer.0.attention.self.value.weight': 'decoder.layers.0.self_attn.v_proj.weight',
 'bart.decoder.decoder_layer.0.attention.self.value.bias': 'decoder.layers.0.self_attn.v_proj.bias',
 'bart.decoder.decoder_layer.0.attention.output.dense.weight': 'decoder.layers.0.self_attn.out_proj.weight',
 'bart.decoder.decoder_layer.0.attention.output.dense.bias': 'decoder.layers.0.self_attn.out_proj.bias',
 'bart.decoder.decoder_layer.0.attention.output.LayerNorm.weight': 'decoder.layers.0.self_attn_layer_norm.weight',
 'bart.decoder.decoder_layer.0.attention.output.LayerNorm.bias': 'decoder.layers.0.self_attn_layer_norm.bias',
 'bart.decoder.decoder_layer.0.crossattention.self.query.weight': 'decoder.layers.0.encoder_attn.q_proj.weight',
 'bart.decoder.decoder_layer.0.crossattention.self.query.bias': 'decoder.layers.0.encoder_attn.q_proj.bias',
 'bart.decoder.decoder_layer.0.crossattention.self.key.weight': 'decoder.layers.0.encoder_attn.k_proj.weight',
 'bart.decoder.decoder_layer.0.crossattention.self.key.bias': 'decoder.layers.0.encoder_attn.k_proj.bias',
 'bart.decoder.decoder_layer.0.crossattention.self.value.weight': 'decoder.layers.0.encoder_attn.v_proj.weight',
 'bart.decoder.decoder_layer.0.crossattention.self.value.bias': 'decoder.layers.0.encoder_attn.v_proj.bias',
 'bart.decoder.decoder_layer.0.crossattention.output.dense.weight': 'decoder.layers.0.encoder_attn.out_proj.weight',
 'bart.decoder.decoder_layer.0.crossattention.output.dense.bias': 'decoder.layers.0.encoder_attn.out_proj.bias',
 'bart.decoder.decoder_layer.0.crossattention.output.LayerNorm.weight': 'decoder.layers.0.encoder_attn_layer_norm.weight',
 'bart.decoder.decoder_layer.0.crossattention.output.LayerNorm.bias': 'decoder.layers.0.encoder_attn_layer_norm.bias',
 'bart.decoder.decoder_layer.0.intermediate.dense.weight': 'decoder.layers.0.fc1.weight',
 'bart.decoder.decoder_layer.0.intermediate.dense.bias': 'decoder.layers.0.fc1.bias',
 'bart.decoder.decoder_layer.0.output.dense.weight': 'decoder.layers.0.fc2.weight',
 'bart.decoder.decoder_layer.0.output.dense.bias': 'decoder.layers.0.fc2.bias',
 'bart.decoder.decoder_layer.0.output.LayerNorm.weight': 'decoder.layers.0.final_layer_norm.weight',
 'bart.decoder.decoder_layer.0.output.LayerNorm.bias': 'decoder.layers.0.final_layer_norm.bias',
 'bart.decoder.decoder_layer.1.attention.self.query.weight': 'decoder.layers.1.self_attn.q_proj.weight',
 'bart.decoder.decoder_layer.1.attention.self.query.bias': 'decoder.layers.1.self_attn.q_proj.bias',
 'bart.decoder.decoder_layer.1.attention.self.key.weight': 'decoder.layers.1.self_attn.k_proj.weight',
 'bart.decoder.decoder_layer.1.attention.self.key.bias': 'decoder.layers.1.self_attn.k_proj.bias',
 'bart.decoder.decoder_layer.1.attention.self.value.weight': 'decoder.layers.1.self_attn.v_proj.weight',
 'bart.decoder.decoder_layer.1.attention.self.value.bias': 'decoder.layers.1.self_attn.v_proj.bias',
 'bart.decoder.decoder_layer.1.attention.output.dense.weight': 'decoder.layers.1.self_attn.out_proj.weight',
 'bart.decoder.decoder_layer.1.attention.output.dense.bias': 'decoder.layers.1.self_attn.out_proj.bias',
 'bart.decoder.decoder_layer.1.attention.output.LayerNorm.weight': 'decoder.layers.1.self_attn_layer_norm.weight',
 'bart.decoder.decoder_layer.1.attention.output.LayerNorm.bias': 'decoder.layers.1.self_attn_layer_norm.bias',
 'bart.decoder.decoder_layer.1.crossattention.self.query.weight': 'decoder.layers.1.encoder_attn.q_proj.weight',
 'bart.decoder.decoder_layer.1.crossattention.self.query.bias': 'decoder.layers.1.encoder_attn.q_proj.bias',
 'bart.decoder.decoder_layer.1.crossattention.self.key.weight': 'decoder.layers.1.encoder_attn.k_proj.weight',
 'bart.decoder.decoder_layer.1.crossattention.self.key.bias': 'decoder.layers.1.encoder_attn.k_proj.bias',
 'bart.decoder.decoder_layer.1.crossattention.self.value.weight': 'decoder.layers.1.encoder_attn.v_proj.weight',
 'bart.decoder.decoder_layer.1.crossattention.self.value.bias': 'decoder.layers.1.encoder_attn.v_proj.bias',
 'bart.decoder.decoder_layer.1.crossattention.output.dense.weight': 'decoder.layers.1.encoder_attn.out_proj.weight',
 'bart.decoder.decoder_layer.1.crossattention.output.dense.bias': 'decoder.layers.1.encoder_attn.out_proj.bias',
 'bart.decoder.decoder_layer.1.crossattention.output.LayerNorm.weight': 'decoder.layers.1.encoder_attn_layer_norm.weight',
 'bart.decoder.decoder_layer.1.crossattention.output.LayerNorm.bias': 'decoder.layers.1.encoder_attn_layer_norm.bias',
 'bart.decoder.decoder_layer.1.intermediate.dense.weight': 'decoder.layers.1.fc1.weight',
 'bart.decoder.decoder_layer.1.intermediate.dense.bias': 'decoder.layers.1.fc1.bias',
 'bart.decoder.decoder_layer.1.output.dense.weight': 'decoder.layers.1.fc2.weight',
 'bart.decoder.decoder_layer.1.output.dense.bias': 'decoder.layers.1.fc2.bias',
 'bart.decoder.decoder_layer.1.output.LayerNorm.weight': 'decoder.layers.1.final_layer_norm.weight',
 'bart.decoder.decoder_layer.1.output.LayerNorm.bias': 'decoder.layers.1.final_layer_norm.bias',
 'bart.decoder.decoder_layer.2.attention.self.query.weight': 'decoder.layers.2.self_attn.q_proj.weight',
 'bart.decoder.decoder_layer.2.attention.self.query.bias': 'decoder.layers.2.self_attn.q_proj.bias',
 'bart.decoder.decoder_layer.2.attention.self.key.weight': 'decoder.layers.2.self_attn.k_proj.weight',
 'bart.decoder.decoder_layer.2.attention.self.key.bias': 'decoder.layers.2.self_attn.k_proj.bias',
 'bart.decoder.decoder_layer.2.attention.self.value.weight': 'decoder.layers.2.self_attn.v_proj.weight',
 'bart.decoder.decoder_layer.2.attention.self.value.bias': 'decoder.layers.2.self_attn.v_proj.bias',
 'bart.decoder.decoder_layer.2.attention.output.dense.weight': 'decoder.layers.2.self_attn.out_proj.weight',
 'bart.decoder.decoder_layer.2.attention.output.dense.bias': 'decoder.layers.2.self_attn.out_proj.bias',
 'bart.decoder.decoder_layer.2.attention.output.LayerNorm.weight': 'decoder.layers.2.self_attn_layer_norm.weight',
 'bart.decoder.decoder_layer.2.attention.output.LayerNorm.bias': 'decoder.layers.2.self_attn_layer_norm.bias',
 'bart.decoder.decoder_layer.2.crossattention.self.query.weight': 'decoder.layers.2.encoder_attn.q_proj.weight',
 'bart.decoder.decoder_layer.2.crossattention.self.query.bias': 'decoder.layers.2.encoder_attn.q_proj.bias',
 'bart.decoder.decoder_layer.2.crossattention.self.key.weight': 'decoder.layers.2.encoder_attn.k_proj.weight',
 'bart.decoder.decoder_layer.2.crossattention.self.key.bias': 'decoder.layers.2.encoder_attn.k_proj.bias',
 'bart.decoder.decoder_layer.2.crossattention.self.value.weight': 'decoder.layers.2.encoder_attn.v_proj.weight',
 'bart.decoder.decoder_layer.2.crossattention.self.value.bias': 'decoder.layers.2.encoder_attn.v_proj.bias',
 'bart.decoder.decoder_layer.2.crossattention.output.dense.weight': 'decoder.layers.2.encoder_attn.out_proj.weight',
 'bart.decoder.decoder_layer.2.crossattention.output.dense.bias': 'decoder.layers.2.encoder_attn.out_proj.bias',
 'bart.decoder.decoder_layer.2.crossattention.output.LayerNorm.weight': 'decoder.layers.2.encoder_attn_layer_norm.weight',
 'bart.decoder.decoder_layer.2.crossattention.output.LayerNorm.bias': 'decoder.layers.2.encoder_attn_layer_norm.bias',
 'bart.decoder.decoder_layer.2.intermediate.dense.weight': 'decoder.layers.2.fc1.weight',
 'bart.decoder.decoder_layer.2.intermediate.dense.bias': 'decoder.layers.2.fc1.bias',
 'bart.decoder.decoder_layer.2.output.dense.weight': 'decoder.layers.2.fc2.weight',
 'bart.decoder.decoder_layer.2.output.dense.bias': 'decoder.layers.2.fc2.bias',
 'bart.decoder.decoder_layer.2.output.LayerNorm.weight': 'decoder.layers.2.final_layer_norm.weight',
 'bart.decoder.decoder_layer.2.output.LayerNorm.bias': 'decoder.layers.2.final_layer_norm.bias',
 'bart.decoder.decoder_layer.3.attention.self.query.weight': 'decoder.layers.3.self_attn.q_proj.weight',
 'bart.decoder.decoder_layer.3.attention.self.query.bias': 'decoder.layers.3.self_attn.q_proj.bias',
 'bart.decoder.decoder_layer.3.attention.self.key.weight': 'decoder.layers.3.self_attn.k_proj.weight',
 'bart.decoder.decoder_layer.3.attention.self.key.bias': 'decoder.layers.3.self_attn.k_proj.bias',
 'bart.decoder.decoder_layer.3.attention.self.value.weight': 'decoder.layers.3.self_attn.v_proj.weight',
 'bart.decoder.decoder_layer.3.attention.self.value.bias': 'decoder.layers.3.self_attn.v_proj.bias',
 'bart.decoder.decoder_layer.3.attention.output.dense.weight': 'decoder.layers.3.self_attn.out_proj.weight',
 'bart.decoder.decoder_layer.3.attention.output.dense.bias': 'decoder.layers.3.self_attn.out_proj.bias',
 'bart.decoder.decoder_layer.3.attention.output.LayerNorm.weight': 'decoder.layers.3.self_attn_layer_norm.weight',
 'bart.decoder.decoder_layer.3.attention.output.LayerNorm.bias': 'decoder.layers.3.self_attn_layer_norm.bias',
 'bart.decoder.decoder_layer.3.crossattention.self.query.weight': 'decoder.layers.3.encoder_attn.q_proj.weight',
 'bart.decoder.decoder_layer.3.crossattention.self.query.bias': 'decoder.layers.3.encoder_attn.q_proj.bias',
 'bart.decoder.decoder_layer.3.crossattention.self.key.weight': 'decoder.layers.3.encoder_attn.k_proj.weight',
 'bart.decoder.decoder_layer.3.crossattention.self.key.bias': 'decoder.layers.3.encoder_attn.k_proj.bias',
 'bart.decoder.decoder_layer.3.crossattention.self.value.weight': 'decoder.layers.3.encoder_attn.v_proj.weight',
 'bart.decoder.decoder_layer.3.crossattention.self.value.bias': 'decoder.layers.3.encoder_attn.v_proj.bias',
 'bart.decoder.decoder_layer.3.crossattention.output.dense.weight': 'decoder.layers.3.encoder_attn.out_proj.weight',
 'bart.decoder.decoder_layer.3.crossattention.output.dense.bias': 'decoder.layers.3.encoder_attn.out_proj.bias',
 'bart.decoder.decoder_layer.3.crossattention.output.LayerNorm.weight': 'decoder.layers.3.encoder_attn_layer_norm.weight',
 'bart.decoder.decoder_layer.3.crossattention.output.LayerNorm.bias': 'decoder.layers.3.encoder_attn_layer_norm.bias',
 'bart.decoder.decoder_layer.3.intermediate.dense.weight': 'decoder.layers.3.fc1.weight',
 'bart.decoder.decoder_layer.3.intermediate.dense.bias': 'decoder.layers.3.fc1.bias',
 'bart.decoder.decoder_layer.3.output.dense.weight': 'decoder.layers.3.fc2.weight',
 'bart.decoder.decoder_layer.3.output.dense.bias': 'decoder.layers.3.fc2.bias',
 'bart.decoder.decoder_layer.3.output.LayerNorm.weight': 'decoder.layers.3.final_layer_norm.weight',
 'bart.decoder.decoder_layer.3.output.LayerNorm.bias': 'decoder.layers.3.final_layer_norm.bias',
 'bart.decoder.decoder_layer.4.attention.self.query.weight': 'decoder.layers.4.self_attn.q_proj.weight',
 'bart.decoder.decoder_layer.4.attention.self.query.bias': 'decoder.layers.4.self_attn.q_proj.bias',
 'bart.decoder.decoder_layer.4.attention.self.key.weight': 'decoder.layers.4.self_attn.k_proj.weight',
 'bart.decoder.decoder_layer.4.attention.self.key.bias': 'decoder.layers.4.self_attn.k_proj.bias',
 'bart.decoder.decoder_layer.4.attention.self.value.weight': 'decoder.layers.4.self_attn.v_proj.weight',
 'bart.decoder.decoder_layer.4.attention.self.value.bias': 'decoder.layers.4.self_attn.v_proj.bias',
 'bart.decoder.decoder_layer.4.attention.output.dense.weight': 'decoder.layers.4.self_attn.out_proj.weight',
 'bart.decoder.decoder_layer.4.attention.output.dense.bias': 'decoder.layers.4.self_attn.out_proj.bias',
 'bart.decoder.decoder_layer.4.attention.output.LayerNorm.weight': 'decoder.layers.4.self_attn_layer_norm.weight',
 'bart.decoder.decoder_layer.4.attention.output.LayerNorm.bias': 'decoder.layers.4.self_attn_layer_norm.bias',
 'bart.decoder.decoder_layer.4.crossattention.self.query.weight': 'decoder.layers.4.encoder_attn.q_proj.weight',
 'bart.decoder.decoder_layer.4.crossattention.self.query.bias': 'decoder.layers.4.encoder_attn.q_proj.bias',
 'bart.decoder.decoder_layer.4.crossattention.self.key.weight': 'decoder.layers.4.encoder_attn.k_proj.weight',
 'bart.decoder.decoder_layer.4.crossattention.self.key.bias': 'decoder.layers.4.encoder_attn.k_proj.bias',
 'bart.decoder.decoder_layer.4.crossattention.self.value.weight': 'decoder.layers.4.encoder_attn.v_proj.weight',
 'bart.decoder.decoder_layer.4.crossattention.self.value.bias': 'decoder.layers.4.encoder_attn.v_proj.bias',
 'bart.decoder.decoder_layer.4.crossattention.output.dense.weight': 'decoder.layers.4.encoder_attn.out_proj.weight',
 'bart.decoder.decoder_layer.4.crossattention.output.dense.bias': 'decoder.layers.4.encoder_attn.out_proj.bias',
 'bart.decoder.decoder_layer.4.crossattention.output.LayerNorm.weight': 'decoder.layers.4.encoder_attn_layer_norm.weight',
 'bart.decoder.decoder_layer.4.crossattention.output.LayerNorm.bias': 'decoder.layers.4.encoder_attn_layer_norm.bias',
 'bart.decoder.decoder_layer.4.intermediate.dense.weight': 'decoder.layers.4.fc1.weight',
 'bart.decoder.decoder_layer.4.intermediate.dense.bias': 'decoder.layers.4.fc1.bias',
 'bart.decoder.decoder_layer.4.output.dense.weight': 'decoder.layers.4.fc2.weight',
 'bart.decoder.decoder_layer.4.output.dense.bias': 'decoder.layers.4.fc2.bias',
 'bart.decoder.decoder_layer.4.output.LayerNorm.weight': 'decoder.layers.4.final_layer_norm.weight',
 'bart.decoder.decoder_layer.4.output.LayerNorm.bias': 'decoder.layers.4.final_layer_norm.bias',
 'bart.decoder.decoder_layer.5.attention.self.query.weight': 'decoder.layers.5.self_attn.q_proj.weight',
 'bart.decoder.decoder_layer.5.attention.self.query.bias': 'decoder.layers.5.self_attn.q_proj.bias',
 'bart.decoder.decoder_layer.5.attention.self.key.weight': 'decoder.layers.5.self_attn.k_proj.weight',
 'bart.decoder.decoder_layer.5.attention.self.key.bias': 'decoder.layers.5.self_attn.k_proj.bias',
 'bart.decoder.decoder_layer.5.attention.self.value.weight': 'decoder.layers.5.self_attn.v_proj.weight',
 'bart.decoder.decoder_layer.5.attention.self.value.bias': 'decoder.layers.5.self_attn.v_proj.bias',
 'bart.decoder.decoder_layer.5.attention.output.dense.weight': 'decoder.layers.5.self_attn.out_proj.weight',
 'bart.decoder.decoder_layer.5.attention.output.dense.bias': 'decoder.layers.5.self_attn.out_proj.bias',
 'bart.decoder.decoder_layer.5.attention.output.LayerNorm.weight': 'decoder.layers.5.self_attn_layer_norm.weight',
 'bart.decoder.decoder_layer.5.attention.output.LayerNorm.bias': 'decoder.layers.5.self_attn_layer_norm.bias',
 'bart.decoder.decoder_layer.5.crossattention.self.query.weight': 'decoder.layers.5.encoder_attn.q_proj.weight',
 'bart.decoder.decoder_layer.5.crossattention.self.query.bias': 'decoder.layers.5.encoder_attn.q_proj.bias',
 'bart.decoder.decoder_layer.5.crossattention.self.key.weight': 'decoder.layers.5.encoder_attn.k_proj.weight',
 'bart.decoder.decoder_layer.5.crossattention.self.key.bias': 'decoder.layers.5.encoder_attn.k_proj.bias',
 'bart.decoder.decoder_layer.5.crossattention.self.value.weight': 'decoder.layers.5.encoder_attn.v_proj.weight',
 'bart.decoder.decoder_layer.5.crossattention.self.value.bias': 'decoder.layers.5.encoder_attn.v_proj.bias',
 'bart.decoder.decoder_layer.5.crossattention.output.dense.weight': 'decoder.layers.5.encoder_attn.out_proj.weight',
 'bart.decoder.decoder_layer.5.crossattention.output.dense.bias': 'decoder.layers.5.encoder_attn.out_proj.bias',
 'bart.decoder.decoder_layer.5.crossattention.output.LayerNorm.weight': 'decoder.layers.5.encoder_attn_layer_norm.weight',
 'bart.decoder.decoder_layer.5.crossattention.output.LayerNorm.bias': 'decoder.layers.5.encoder_attn_layer_norm.bias',
 'bart.decoder.decoder_layer.5.intermediate.dense.weight': 'decoder.layers.5.fc1.weight',
 'bart.decoder.decoder_layer.5.intermediate.dense.bias': 'decoder.layers.5.fc1.bias',
 'bart.decoder.decoder_layer.5.output.dense.weight': 'decoder.layers.5.fc2.weight',
 'bart.decoder.decoder_layer.5.output.dense.bias': 'decoder.layers.5.fc2.bias',
 'bart.decoder.decoder_layer.5.output.LayerNorm.weight': 'decoder.layers.5.final_layer_norm.weight',
 'bart.decoder.decoder_layer.5.output.LayerNorm.bias': 'decoder.layers.5.final_layer_norm.bias'}
model_new = {}
for key, value in map.items():
    model_new[value] = torch_weights[key]
torch.save(model_new, 'E:/pretrain_ckpt/bart/[cloudwalk_torch_base]/bert4torch_pytorch_model.bin')