import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

print("=== 1. POSITIONAL ENCODING (Page 15) ===")
d_model = 4
for pos in [0, 1, 2]:
    pe = []
    for i in range(d_model // 2):
        pe.append(np.sin(pos / (10000 ** (2 * i / d_model))))
        pe.append(np.cos(pos / (10000 ** (2 * i / d_model))))
    print(f"pos={pos}: {np.round(pe, 3)}")

print("\n=== 2. SCALED DOT-PRODUCT ATTENTION (Pages 21-23) ===")
Q = np.array([[1,0], [0,1], [1,1]])
K = np.array([[1,0], [0,1], [1,1]])
V = np.array([[1,0], [0,2], [1,1]])
d_k = 2

scores = Q @ K.T
scaled = scores / np.sqrt(d_k)
weights = softmax(scaled)
output = weights @ V

print("Raw QK^T:\n", scores)
print("Scaled:\n", np.round(scaled, 3))
print("Softmax Weights:\n", np.round(weights, 3))
print("Output Z:\n", np.round(output, 3))

print("\n=== 3. CAUSAL MASK (Page 30) ===")
scores_3tok = np.array([[1.2, 0.4, 0.1], [0.3, 1.5, 0.6], [0.2, 0.8, 1.4]])
mask = np.triu(np.ones_like(scores_3tok), k=1).astype(bool)
scores_masked = scores_3tok.copy()
scores_masked[mask] = -np.inf
weights_masked = softmax(scores_masked)
print("Masked Weights:\n", np.round(weights_masked, 3))