from flask import Flask, jsonify, request
import tensorflow as tf
import os 
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from src.model.utils import CustomSchedule, masked_accuracy, masked_loss
from src.inference import translate
from data.preprocess import preprocessing
import pickle
import pandas as pd



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

app = Flask(__name__)

@app.route('/translate', methods=['POST'])
def predict():
    data = request.get_json()
    input_text = data['text']
    input_text = preprocessing(pd.Series([input_text])).values[0]
    translated_text = translate(input_text, x_tokenizer, y_tokenizer, model, max_len_x, max_len_y)
    return jsonify({'translation': translated_text})

@app.route('/')
def index():
    return jsonify({'message': 'Welcome to the Translation API!'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)