from gensim.models import KeyedVectors

print("Loading the trained model...")
model = KeyedVectors.load("data/text8-word2vec.bin")
word_vectors = model.wv

print("First 10 words:", list(word_vectors.key_to_index.keys())[:10])

print("\nSimilar words to 'king':")
for word, score in word_vectors.most_similar("king", topn=5):
    print(f"{score:.3f} {word}")

print("\nVector arithmetic: France - Paris + Berlin = ?")
result = word_vectors.most_similar(positive=['france', 'berlin'], negative=['paris'], topn=1)
print(f"{result[0][1]:.3f} {result[0][0]}")

print("\nOdd one out in ['hindus', 'parsis', 'singapore', 'christians']:")
print(word_vectors.doesnt_match(["hindus", "parsis", "singapore", "christians"]))

print("\nSimilarity scores:")
for word in ["woman", "dog", "whale", "tree"]:
    sim = word_vectors.similarity("man", word)
    print(f"similarity(man, {word}) = {sim:.3f}")