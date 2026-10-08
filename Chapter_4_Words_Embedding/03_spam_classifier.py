import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix

# --- 1. Load the data ---
def load_data():
    labels, texts = [], []
    local_file = os.path.join("datasets", "SMSSpamCollection")
    with open(local_file, "r", encoding="utf-8") as fin:
        for line in fin:
            label, text = line.strip().split('\t')
            labels.append(1 if label == "spam" else 0)
            texts.append(text)
    return texts, labels

texts, labels = load_data()

# --- 2. Tokenize and Pad ---
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(texts)
text_sequences = tokenizer.texts_to_sequences(texts)
text_sequences = tf.keras.preprocessing.sequence.pad_sequences(text_sequences)

num_records = len(text_sequences)
max_seqlen = len(text_sequences[0])
print(f"{num_records} sentences, max length: {max_seqlen}")

word2idx = tokenizer.word_index
word2idx["PAD"] = 0
vocab_size = len(word2idx)
print(f"vocab size: {vocab_size}")

# --- 3. Prepare Dataset ---
NUM_CLASSES = 2
cat_labels = tf.keras.utils.to_categorical(labels, num_classes=NUM_CLASSES)

dataset = tf.data.Dataset.from_tensor_slices((text_sequences, cat_labels))
dataset = dataset.shuffle(10000)

test_size = num_records // 4
val_size = (num_records - test_size) // 10

test_dataset = dataset.take(test_size)
val_dataset = dataset.skip(test_size).take(val_size)
train_dataset = dataset.skip(test_size + val_size)

BATCH_SIZE = 128
test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True)
val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)

# --- 4. Build the Model ---
class SpamClassifierModel(tf.keras.Model):
    def __init__(self, vocab_sz, embed_sz, input_length, num_filters, kernel_sz, output_sz, **kwargs):
        super(SpamClassifierModel, self).__init__(**kwargs)
        self.embedding = tf.keras.layers.Embedding(vocab_sz, embed_sz, input_length=input_length, trainable=True)
        self.conv = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=kernel_sz, activation="relu")
        self.dropout = tf.keras.layers.SpatialDropout1D(0.2)
        self.pool = tf.keras.layers.GlobalMaxPooling1D()
        self.dense = tf.keras.layers.Dense(output_sz, activation="softmax")

    def call(self, x):
        x = self.embedding(x)
        x = self.conv(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = self.dense(x)
        return x

EMBEDDING_DIM = 300
conv_num_filters = 256
conv_kernel_size = 3

model = SpamClassifierModel(vocab_size, EMBEDDING_DIM, max_seqlen, conv_num_filters, conv_kernel_size, NUM_CLASSES)
model.build(input_shape=(None, max_seqlen))
model.summary()

# --- 5. Compile and Train ---
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

NUM_EPOCHS = 3
CLASS_WEIGHTS = {0: 1, 1: 8} # Handle class imbalance

model.fit(train_dataset, epochs=NUM_EPOCHS, validation_data=val_dataset, class_weight=CLASS_WEIGHTS)

# --- 6. Evaluate ---
labels_list, predictions_list = [], []
for Xtest, Ytest in test_dataset:
    Ytest_ = model.predict_on_batch(Xtest)
    ytest = np.argmax(Ytest, axis=1)
    ytest_ = np.argmax(Ytest_, axis=1)
    labels_list.extend(ytest.tolist())
    predictions_list.extend(ytest_.tolist())

print("test accuracy: {:.3f}".format(accuracy_score(labels_list, predictions_list)))
print("confusion matrix")
print(confusion_matrix(labels_list, predictions_list))