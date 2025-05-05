import tensorflow as tf
import numpy as np

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__ (self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = self.positional_encoding(length=2048, depth=d_model)

    def positional_encoding(self, length, depth):
        depth = depth/2
        positions = np.arange(length)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :]/depth
        angle_rates = 1/(10000**depths)
        angle_rads = positions*angle_rates
        pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)
    
    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x
    
class BaseAttention(tf.keras.layers.Layer):
    def __init__(self,  **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.norm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

class CrossAttention(BaseAttention):
    def call(self, x, context):
        output = self.mha(query=x, key=context, value=context)
        x = self.add([x, output])
        x = self.norm(x)
        return x
    
class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        output =  self.mha(query=x, key=x, value=x)
        x = self.add([output, x])
        x = self.norm(x)
        return x
    
class CasualSelfAttention(BaseAttention):
    def call(self, x):
        output = self.mha(query=x, key=x, value=x, use_causal_mask=True)
        x = self.add([x, output])
        x = self.norm(x)
        return x

class FeedFoward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        output = self.seq(x)
        x = self.add([output, x])
        x = self.norm(x)
        return x
    
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.self_attention = GlobalSelfAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout_rate)
        self.feed_foward =  FeedFoward(d_model=d_model, dff=dff, dropout_rate=dropout_rate)

    def call(self, x):
        x = self.self_attention(x)
        x = self.feed_foward(x)
        return x

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.casual_attention = CasualSelfAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout_rate)
        self.cross_attention = CrossAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout_rate)
        self.feed_foward = FeedFoward(d_model=d_model, dff=dff, dropout_rate=dropout_rate)

    def call(self, x, context):
        x = self.casual_attention(x)
        x = self.cross_attention(x, context)
        x = self.feed_foward(x)
        return x