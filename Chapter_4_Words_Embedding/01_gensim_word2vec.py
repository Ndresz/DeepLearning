import gensim.downloader as api
from gensim.models import Word2Vec
import os

print("Downloading text8 dataset...")
dataset = api.load("text8")

print("Training Word2Vec model... (This will take 5-10 minutes)")
model = Word2Vec(dataset)

# Create a 'data' folder to save it
if not os.path.exists("data"):
    os.makedirs("data")

model.save("data/text8-word2vec.bin")
print("Model saved to data/text8-word2vec.bin")