import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from src.model.utils import CustomSchedule, masked_accuracy, masked_loss
from data.preprocess import preprocessing
import pickle

with open('saved_models/x_tokenizer.pkl', 'rb') as f:
    x_tokenizer = pickle.load(f)

with open('saved_models/y_tokenizer.pkl', 'rb') as f:
    y_tokenizer = pickle.load(f)

with open('saved_models/max_lengths.pkl', 'rb') as f:
    lengths = pickle.load(f)
    max_len_x = lengths['max_len_x']
    max_len_y = lengths['max_len_y']

model = tf.keras.models.load_model('saved_models/transformer_1', custom_objects={
    'CustomSchedule': CustomSchedule,
    'masked_loss': masked_loss,
    'masked_accuracy': masked_accuracy
})

def translate(input_text):
    predict_seq = x_tokenizer.texts_to_sequences([input_text])
    predict_seq = tf.keras.preprocessing.sequence.pad_sequences(predict_seq, padding='post', maxlen=max_len_x)
    predict_seq = tf.convert_to_tensor(predict_seq, dtype=tf.int32)

    target_seq = np.zeros((1, max_len_y), dtype=np.int32)
    target_seq[0, 0] = y_tokenizer.word_index['<start>']
    target_seq = tf.convert_to_tensor(target_seq, dtype=tf.int32)

    translated_sentence = []

    for i in range(1, max_len_y):
        predicted = model.predict([predict_seq, target_seq], verbose=0)
        predicted_word_index = np.argmax(predicted[0, i - 1, :])
        target_seq = tf.tensor_scatter_nd_update(
            target_seq, [[0, i]], [predicted_word_index]
        ) 
        word = y_tokenizer.index_word.get(predicted_word_index, '')
        if word == '<end>':
            break
        translated_sentence.append(word)

    return ' '.join(translated_sentence)

input_text = "Dia datang ke Jepang untuk belajar bahasa Jepang."
input_text = preprocessing(pd.Series([input_text])).values[0]
translated_text = translate(input_text)
print("Input:", input_text)
print("Predicted translation:", translated_text)