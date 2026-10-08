import tensorflow as tf
import numpy as np

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        pe = np.zeros((max_len, d_model))
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
                pe[pos, i+1] = np.cos(pos / (10000 ** (i / d_model)))
        self.pe = tf.cast(pe, dtype=tf.float32)

    def call(self, x):
        return x + self.pe[:tf.shape(x)[1], :]

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)
        
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled = matmul_qk / tf.math.sqrt(dk)
        
        if mask is not None:
            scaled += (mask * -1e9)
            
        weights = tf.nn.softmax(scaled, axis=-1)
        out = tf.matmul(weights, v)
        out = tf.transpose(out, perm=[0, 2, 1, 3])
        concat = tf.reshape(out, (batch_size, -1, self.d_model))
        return self.dense(concat), weights

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, rate=0.1):
        super().__init__()
        self.att = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = tf.keras.layers.Dropout(rate)
        self.drop2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask=None):
        attn, _ = self.att(x, x, x, mask)
        out1 = self.ln1(x + self.drop1(attn, training=training))
        ffn = self.ffn(out1)
        return self.ln2(out1 + self.drop2(ffn, training=training))

# Test with exact dimensions from Page 41
d_model, num_heads, d_ff = 512, 8, 2048
sample_input = tf.random.uniform((2, 10, d_model))
pe = PositionalEncoding(d_model)
pe_output = pe(sample_input)
block = TransformerBlock(d_model, num_heads, d_ff)
output = block(pe_output, training=False)
print("Input shape:", sample_input.shape)
print("Output shape:", output.shape)