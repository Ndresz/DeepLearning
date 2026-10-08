import tensorflow as tf
import numpy as np
import os

print("Downloading Alice in Wonderland...")
url = "https://www.gutenberg.org/files/11/11-0.txt"
path = tf.keras.utils.get_file("alice.txt", url)

text = open(path, 'rb').read().decode(encoding='utf-8')
text = text[text.find('*** START'):text.find('*** END')]
text = text.replace('\n', ' ').replace('\r', '')

vocab = sorted(set(text))
print(f"Unique characters: {len(vocab)}")

char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

seq_length = 100
examples_per_epoch = len(text) // (seq_length + 1)

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(None,), batch_size=batch_size),
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)
model.summary()

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)

EPOCHS = 10
history = model.fit(dataset, epochs=EPOCHS)

model.save_weights('alice_weights.weights.h5')

def generate_text(model, start_string, temperature=1.0):
    num_generate = 200
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []
    
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])
        
    return start_string + ''.join(text_generated)

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights('alice_weights.weights.h5')

print("\n--- Generated Text ---")
print(generate_text(model, start_string="Alice ", temperature=0.5))