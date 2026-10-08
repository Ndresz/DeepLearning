import tensorflow as tf
import tensorflow_hub as hub

print("Loading Universal Sentence Encoder... (This will download ~1GB)")
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/4")

embeddings = embed([
    "I like green eggs and ham",
    "would you eat them in a box"
])["outputs"]

print("Output shape:", embeddings.shape)