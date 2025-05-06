import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from src.model.utils import CustomSchedule, masked_accuracy, masked_loss
from src.inference import translate
from data.preprocess import preprocessing
import pickle

def evaluate_bleu_on_validation(x_val, y_val):
    total_bleu_score = 0
    smoothing_function = SmoothingFunction().method1
    num_samples = len(x_val)

    for i in range(num_samples):
        input_text = preprocessing(pd.Series([x_val.iloc[i]])).values[0]
        reference_text = preprocessing(pd.Series([y_val.iloc[i]])).values[0]
        translated_text = translate(input_text, x_tokenizer, y_tokenizer, model, max_len_x, max_len_y)
        reference_tokens = [reference_text.split()]
        translated_tokens = translated_text.split()
        bleu_score = sentence_bleu(reference_tokens, translated_tokens, smoothing_function=smoothing_function)
        total_bleu_score += bleu_score
        print(f"Sample {i + 1}/{num_samples}")
        print(f"Input: {input_text}")
        print(f"Reference: {reference_text}")
        print(f"Predicted: {translated_text}")
        print(f"BLEU Score: {bleu_score:.4f}")
        print("-" * 50)

    average_bleu_score = total_bleu_score / num_samples
    return average_bleu_score

if __name__ == '__main__':
    df = pd.read_csv('data/processed/id-en.csv')

    x_train, x_val, y_train, y_val = train_test_split(df['id'], df['en'], test_size=0.2, random_state=42)

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

    average_bleu = evaluate_bleu_on_validation(x_val, y_val)
    print(f"Average BLEU Score on Validation Data: {average_bleu:.4f}")