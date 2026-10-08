import tensorflow as tf
import tensorflow_text
import numpy as np

# --- 1. Tiny Dataset (PT-EN pairs) ---
pt_sentences = [
    "eu sou um estudante", "ele é um médico", "nós somos amigos",
    "o gato está no tapete", "a menina come maçã"
]
en_sentences = [
    "i am a student", "he is a doctor", "we are friends",
    "the cat is on the mat", "the girl eats an apple"
]

# --- 2. Tokenization ---
pt_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
pt_tokenizer.fit_on_texts(pt_sentences)
en_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
en_tokenizer.fit_on_texts(en_sentences)

pt_seq = pt_tokenizer.texts_to_sequences(pt_sentences)
en_seq = en_tokenizer.texts_to_sequences(en_sentences)

pt_vocab_size = len(pt_tokenizer.word_index) + 1
en_vocab_size = len(en_tokenizer.word_index) + 1

pt_padded = tf.keras.preprocessing.sequence.pad_sequences(pt_seq, padding='post')
en_padded = tf.keras.preprocessing.sequence.pad_sequences(en_seq, padding='post')

en_input = np.pad(en_padded, ((0,0),(1,0)), 'constant', constant_values=0)
en_target = np.pad(en_padded, ((0,0),(0,1)), 'constant', constant_values=0)

# --- 3. Model Components ---
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=100):
        super().__init__()
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
        if mask is not None: scaled += (mask * -1e9)
        weights = tf.nn.softmax(scaled, axis=-1)
        out = tf.matmul(weights, v)
        out = tf.transpose(out, perm=[0, 2, 1, 3])
        concat = tf.reshape(out, (batch_size, -1, self.d_model))
        return self.dense(concat), weights

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, rate=0.1):
        super().__init__()
        self.att = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([tf.keras.layers.Dense(d_ff, activation='relu'), tf.keras.layers.Dense(d_model)])
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = tf.keras.layers.Dropout(rate)
        self.drop2 = tf.keras.layers.Dropout(rate)
    def call(self, x, training, mask=None):
        attn, _ = self.att(x, x, x, mask)
        out1 = self.ln1(x + self.drop1(attn, training=training))
        ffn = self.ffn(out1)
        return self.ln2(out1 + self.drop2(ffn, training=training))

class Translator(tf.keras.Model):
    def __init__(self, pt_vocab, en_vocab, d_model=512, num_heads=8, d_ff=2048):
        super().__init__()
        self.pt_embed = tf.keras.layers.Embedding(pt_vocab, d_model)
        self.en_embed = tf.keras.layers.Embedding(en_vocab, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.encoder = TransformerBlock(d_model, num_heads, d_ff)
        self.decoder = TransformerBlock(d_model, num_heads, d_ff)
        self.final_layer = tf.keras.layers.Dense(en_vocab)

    def call(self, inputs, training=False):
        pt_input, en_input = inputs
        pt_x = self.pos_enc(self.pt_embed(pt_input))
        
        # FIX: Pass training as keyword argument
        enc_output = self.encoder(pt_x, training=training)
        
        en_x = self.pos_enc(self.en_embed(en_input))
        seq_len = tf.shape(en_input)[1]
        mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        mask = tf.cast(mask, tf.float32)[tf.newaxis, tf.newaxis, :, :]
        
        # FIX: Pass training and mask as keyword arguments
        dec_output = self.decoder(en_x, training=training, mask=mask)
        return self.final_layer(dec_output)

# --- 4. Training ---
model = Translator(pt_vocab_size, en_vocab_size)
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=0))
model.fit([pt_padded, en_input], en_target, epochs=50, batch_size=2)

# --- 5. Autoregressive Decoding ---
def translate(sentence):
    seq = pt_tokenizer.texts_to_sequences([sentence])
    seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=pt_padded.shape[1], padding='post')
    decoder_input = tf.constant([[0]]) # START token
    result = []
    for _ in range(10):
        predictions = model([seq, decoder_input], training=False)
        predicted_id = tf.argmax(predictions[:, -1, :], axis=-1).numpy()[0]
        if predicted_id == 0: break # END token
        result.append(predicted_id)
        decoder_input = tf.concat([decoder_input, [[predicted_id]]], axis=-1)
    return en_tokenizer.sequences_to_texts([result])[0]

print("\n--- Translation Test ---")
print("PT: eu sou um estudante")
print("EN (predicted):", translate("eu sou um estudante"))