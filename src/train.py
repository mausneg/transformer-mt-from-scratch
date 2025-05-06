import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
        )
        print("✅ Set memory limit to full 4096 MB")
    except RuntimeError as e:
        print(e)
        
import pickle
import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from src.model.transformer import Transformer
from src.model.utils import CustomSchedule, masked_accuracy, masked_loss




batch_size = 64
epochs = 50
num_layers = 4
num_heads = 8
d_model = 128
dff = 512
dropout_rate = 0.1

df = pd.read_csv('data/processed/id-en.csv')

x_train, x_val, y_train, y_val = train_test_split(df['id'], df['en'], test_size=0.2, random_state=42)
print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)

y_train = y_train.apply(lambda x: '<start> ' + x + ' <end>')
y_val = y_val.apply(lambda x: '<start> ' + x + ' <end>')

x_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<OOV>')
y_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<OOV>')

x_tokenizer.fit_on_texts(x_train)
y_tokenizer.fit_on_texts(y_train)

x_train_seq = x_tokenizer.texts_to_sequences(x_train)
y_train_seq = y_tokenizer.texts_to_sequences(y_train)
x_val_seq = x_tokenizer.texts_to_sequences(x_val)
y_val_seq = y_tokenizer.texts_to_sequences(y_val)

max_len_x = max(len(x) for x in x_train_seq)
max_len_y = max(len(y) for y in y_train_seq)

decoder_x_train = [seq[:-1] for seq in y_train_seq]
decoder_y_train = [seq[1:] for seq in y_train_seq]
decoder_x_val = [seq[:-1] for seq in y_val_seq]
decoder_y_val = [seq[1:] for seq in y_val_seq]

x_train_pad = tf.keras.preprocessing.sequence.pad_sequences(x_train_seq, padding='post', maxlen=max_len_x)
decoder_x_train_pad = tf.keras.preprocessing.sequence.pad_sequences(decoder_x_train, padding='post', maxlen=max_len_y)
decoder_y_train_pad = tf.keras.preprocessing.sequence.pad_sequences(decoder_y_train, padding='post', maxlen=max_len_y)
x_val_pad = tf.keras.preprocessing.sequence.pad_sequences(x_val_seq, padding='post', maxlen=max_len_x)
decoder_x_val_pad = tf.keras.preprocessing.sequence.pad_sequences(decoder_x_val, padding='post', maxlen=max_len_y)
decoder_y_val_pad = tf.keras.preprocessing.sequence.pad_sequences(decoder_y_val, padding='post', maxlen=max_len_y)

vocab_size_input = len(x_tokenizer.word_index) + 1
vocab_size_output = len(y_tokenizer.word_index) + 1

print('Total input vocab:', vocab_size_input)
print('Total output vocab:', vocab_size_output)
print('Max length input:', max_len_x)
print('Max length output:', max_len_y)
print('Sample input encoder:', x_train_pad[0])
print('Sample input decoder:', decoder_x_train_pad[0])
print('Sample target decoder:', decoder_y_train_pad[0])

model = Transformer(
    num_layers=num_layers,
    num_heads=num_heads,
    d_model=d_model,
    dff=dff,
    input_vocab_size=vocab_size_input,
    target_vocab_size=vocab_size_output,
    dropout_rate=dropout_rate
)
learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
model.compile(optimizer=optimizer, loss=masked_loss, metrics=[masked_accuracy])
model.build(input_shape=[(None, None), (None, None)])
model.summary()

history = model.fit(
    x=[x_train_pad, decoder_x_train_pad],
    y=decoder_y_train_pad,
    validation_data=([x_val_pad, decoder_x_val_pad], decoder_y_val_pad),
    batch_size=batch_size,
    epochs=epochs,
)
model.save('saved_models/transformer_1')

with open('saved_models/x_tokenizer.pkl', 'wb') as f:
    pickle.dump(x_tokenizer, f)

with open('saved_models/y_tokenizer.pkl', 'wb') as f:
    pickle.dump(y_tokenizer, f)

with open('saved_models/max_lengths.pkl', 'wb') as f:
    pickle.dump({'max_len_x': max_len_x, 'max_len_y': max_len_y}, f)