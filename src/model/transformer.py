import tensorflow as tf
import numpy as np
from layers import PositionalEmbedding, EncoderLayer, DecoderLayer

class Encoder(tf.keras.layers):
    def __init__(self, num_layers, num_heads, dff, d_model, vocab_size, dropout_rate=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(d_model=d_model, vocab_size=vocab_size)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.encoder_layers = [EncoderLayer(dff=dff, d_model=d_model, num_heads=num_heads, dropout_rate=dropout_rate) for _ in range(num_layers)]

    def call(self, x):
        x = self.pos_embedding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.encoder_layers[i](x)
        return x
    
class Decoder(tf.keras.layers):
    def __init__(self, num_layers, num_heads, dff, d_model, vocab_size, dropout_rate=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(d_model=d_model, vocab_size=vocab_size)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.decoder = [DecoderLayer(d_model=d_model, dff=dff, num_heads=num_heads, dropout_rate=dropout_rate) for _ in range(num_layers)]

    def call(self, x, context):
        x = self.pos_embedding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.decoder[i](x, context)
        return x
    
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, num_heads, dff, d_model, vocab_size, dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(d_model=d_model, dff=dff, num_heads=num_heads, num_layers=num_layers, vocab_size=vocab_size, dropout_rate=dropout_rate)
        self.decoder = Decoder(d_model=d_model, dff=dff, num_heads=num_heads, num_layers=num_layers, vocab_size=vocab_size, dropout_rate=dropout_rate)
        self.output_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs):
        context, target = inputs
        context = self.encoder(context)
        target = self.decoder(target, context)
        output = self.output_layer(target)
        try:
            del output._keras_mask
        except:
            pass
        return output