import tensorflow as tf
import json
import numpy as np
from kerasbert import bert_encoder_block, model_loader
import tokenization

# BERT配置
config_file = "../../data/chinese_L-12_H-768_A-12/bert_config.json"
ckpt_path = "../../data/chinese_L-12_H-768_A-12/bert_model.ckpt"
vocab_file = "../../data/chinese_L-12_H-768_A-12/vocab.txt"

bert_config = json.load(open(config_file, 'r'))

config = {
    "hidden_size": bert_config["hidden_size"],
    "num_layers": bert_config["num_hidden_layers"],
    "num_attention_heads": bert_config["num_attention_heads"],
    "max_sequence_length": 512,
    "intermediate_size": bert_config["intermediate_size"],
    "type_vocab_size": bert_config["type_vocab_size"],
    "vocab_size": bert_config["vocab_size"],
    "activation": bert_config["hidden_act"],
    "dropout_rate": bert_config["hidden_dropout_prob"],
    "attention_dropout_rate": bert_config["attention_probs_dropout_prob"],
}
sequence_length = config["max_sequence_length"]


def build_model():
    wid_inputs = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32, name="wid")
    mask_inputs = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32, name="mask")
    typeid_inputs = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32, name="typeid")

    bert = bert_encoder_block.BertEncoderBlock(**config)
    seq_output, pool_output = bert([wid_inputs, mask_inputs, typeid_inputs])
    model = tf.keras.Model(inputs=[wid_inputs, mask_inputs, typeid_inputs], outputs=seq_output)
    model_loader.load_block_weights_from_official_checkpoint(bert, config, ckpt_path)
    return model


model = build_model()

tokenizer = tokenization.FullTokenizer(vocab_file)
line = "Google 已经公开了 TensorFlow 版本的预训练模型和代码"
words = tokenizer.tokenize(line)
words = ["[CLS]"] + tokenizer.tokenize(line) + ["[SEP]"]
token_ids = tokenizer.convert_tokens_to_ids(words)
segment_ids = [0] * len(token_ids)
input_mask = [1] * len(token_ids)
length = len(token_ids) - 2
while len(token_ids) < sequence_length:
    token_ids.append(0)
    segment_ids.append(0)
    input_mask.append(0)
token_ids = np.array([token_ids])
segment_ids = np.array([segment_ids])
input_mask = np.array([input_mask])

bert_output = model.predict([token_ids, input_mask, segment_ids])
last_layer_vector = bert_output[-1][0]

print(len(words), words)
print(last_layer_vector.shape)
print(last_layer_vector[0])
