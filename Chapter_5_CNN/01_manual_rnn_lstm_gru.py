import numpy as np

def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print("=== 1. BASIC RNN CELL (Page 10) ===")
# U=0.5, W=0.8, V=1.2, h0=0
U, W, V, h_prev = 0.5, 0.8, 1.2, 0.0
x = [1.00, 0.50, -0.30]

for t, x_t in enumerate(x, 1):
    h_t = tanh(W * h_prev + U * x_t)
    y_t = V * h_t
    print(f"t={t}: h{t} = {h_t:.3f}, y{t} = {y_t:.3f}")
    h_prev = h_t

print("\n=== 2. LSTM SCALAR STEP (Page 18) ===")
# x=0.50, h_prev=0.40, c_prev=0.60
x_t, h_prev, c_prev = 0.50, 0.40, 0.60
# Gate inputs (from the slide)
i = sigmoid(0.47)
f = sigmoid(0.48)
o = sigmoid(0.31)
g = tanh(0.50)

c_t = f * c_prev + i * g
h_t = o * tanh(c_t)
print(f"i={i:.3f}, f={f:.3f}, o={o:.3f}, g={g:.3f}")
print(f"c_t = {c_t:.3f}")
print(f"h_t = {h_t:.3f}")

print("\n=== 3. GRU SCALAR STEP (Page 20) ===")
# x=0.50, h_prev=0.40. Weights: W=0.5, U=0.4, bias=0
x_t, h_prev = 0.50, 0.40
W_gru, U_gru = 0.5, 0.4

z = sigmoid(W_gru * x_t + U_gru * h_prev)
r = sigmoid(W_gru * x_t + U_gru * h_prev)
c = tanh(W_gru * x_t + U_gru * (r * h_prev))
h_t = z * h_prev + (1 - z) * c

print(f"z={z:.3f}, r={r:.3f}, c={c:.3f}")
print(f"h_t = {h_t:.3f}")