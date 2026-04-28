# draw_plates_cpu_fixed.py
#!/usr/bin/env python
"""DRAW model for license plates - CPU optimized version (FIXED)"""

import tensorflow as tf
import numpy as np
import os

# ============================================
# MODEL PARAMETERS
# ============================================

# Image dimensions (width, height)
A, B = 96, 32
img_size = A * B

# Model architecture
enc_size = 512   # INCREASED from 256
dec_size = 512   # INCREASED from 256
read_n = 12      # INCREASED from 8
write_n = 12     # INCREASED from 8
z_size = 10
T = 30           # INCREASED from 20

# Training parameters
batch_size = 32
train_iters = 5000
learning_rate = 1e-4  # REDUCED from 1e-3
eps = 1e-8

# Attention flags
read_attn = True
write_attn = True

# Loss scaling (normalize to MNIST equivalent)
MNIST_PIXELS = 784
reconstruction_weight = MNIST_PIXELS / img_size  # ≈ 0.255

print(f"🚀 DRAW Model Configuration:")
print(f"   Image size: {A}×{B} = {img_size} pixels")
print(f"   Loss scaling factor: {reconstruction_weight:.3f}")
print(f"   Batch size: {batch_size}")
print(f"   Time steps: {T}")
print(f"   Learning rate: {learning_rate}")

# ============================================
# DATA LOADER (unchanged)
# ============================================

class LicensePlateData:
    def __init__(self, data_path, batch_size):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        print(f"📂 Loading data from: {data_path}")
        self.data = np.load(data_path).astype(np.float32)
        
        # CRITICAL FIX: Ensure data is in [0,1] range
        if self.data.max() > 1.0:
            print(f"⚠️ Normalizing data from [0,{self.data.max()}] to [0,1]")
            self.data = self.data / 255.0
        
        self.num_samples = len(self.data)
        self.batch_size = min(batch_size, self.num_samples)
        self.epochs_completed = 0
        self.index_in_epoch = 0
        
        print(f"✅ Loaded {self.num_samples} license plate images")
        print(f"📊 Data shape: {self.data.shape}, range: [{self.data.min():.3f}, {self.data.max():.3f}]")
        
        self._shuffle_data()
    
    def _shuffle_data(self):
        perm = np.arange(self.num_samples)
        np.random.shuffle(perm)
        self.data = self.data[perm]
    
    def next_batch(self, batch_size):
        start = self.index_in_epoch
        
        if start + batch_size > self.num_samples:
            self.epochs_completed += 1
            self._shuffle_data()
            start = 0
            self.index_in_epoch = batch_size
            return self.data[start:self.index_in_epoch]
        else:
            self.index_in_epoch += batch_size
            return self.data[start:self.index_in_epoch]

# ============================================
# MODEL FUNCTIONS (FIXED)
# ============================================

def linear(x, output_dim, scope=None):
    """Fixed linear layer with proper variable scope"""
    with tf.variable_scope(scope or "linear"):
        w = tf.get_variable("w", [x.get_shape()[1], output_dim])
        b = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
        return tf.matmul(x, w) + b

def filterbank(gx, gy, sigma2, delta, N):
    grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
    mu_x = gx + (grid_i - N / 2 - 0.5) * delta
    mu_y = gy + (grid_i - N / 2 - 0.5) * delta
    a = tf.reshape(tf.cast(tf.range(A), tf.float32), [1, 1, -1])
    b = tf.reshape(tf.cast(tf.range(B), tf.float32), [1, 1, -1])
    mu_x = tf.reshape(mu_x, [-1, N, 1])
    mu_y = tf.reshape(mu_y, [-1, N, 1])
    sigma2 = tf.reshape(sigma2, [-1, 1, 1])
    Fx = tf.exp(-tf.square(a - mu_x) / (2 * sigma2))
    Fy = tf.exp(-tf.square(b - mu_y) / (2 * sigma2))
    Fx = Fx / tf.maximum(tf.reduce_sum(Fx, 2, keep_dims=True), eps)
    Fy = Fy / tf.maximum(tf.reduce_sum(Fy, 2, keep_dims=True), eps)
    return Fx, Fy

def attn_window(scope, h_dec, N):
    """FIXED: Proper variable reuse with AUTO_REUSE"""
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        params = linear(h_dec, 5, scope="attn_params")
    
    gx_, gy_, log_sigma2, log_delta, log_gamma = tf.split(params, 5, 1)
    gx = (A + 1) / 2 * (gx_ + 1)
    gy = (B + 1) / 2 * (gy_ + 1)
    sigma2 = tf.exp(log_sigma2)
    delta = (max(A, B) - 1) / (N - 1) * tf.exp(log_delta)
    Fx, Fy = filterbank(gx, gy, sigma2, delta, N)
    gamma = tf.exp(log_gamma)
    return Fx, Fy, gamma

def read_attn(x, x_hat, h_dec_prev):
    Fx, Fy, gamma = attn_window("read", h_dec_prev, read_n)
    
    def filter_img(img, Fx, Fy, gamma, N):
        Fxt = tf.transpose(Fx, perm=[0, 2, 1])
        img = tf.reshape(img, [-1, B, A])
        glimpse = tf.matmul(Fy, tf.matmul(img, Fxt))
        glimpse = tf.reshape(glimpse, [-1, N * N])
        return glimpse * tf.reshape(gamma, [-1, 1])
    
    x_filtered = filter_img(x, Fx, Fy, gamma, read_n)
    x_hat_filtered = filter_img(x_hat, Fx, Fy, gamma, read_n)
    return tf.concat([x_filtered, x_hat_filtered], 1)

def encode(state, input_tensor):
    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
        return lstm_enc(input_tensor, state)

def sampleQ(h_enc):
    with tf.variable_scope("mu", reuse=tf.AUTO_REUSE):
        mu = linear(h_enc, z_size)
    with tf.variable_scope("sigma", reuse=tf.AUTO_REUSE):
        logsigma = linear(h_enc, z_size)
        sigma = tf.exp(logsigma)
    return mu + sigma * e, mu, logsigma, sigma

def decode(state, input_tensor):
    with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
        return lstm_dec(input_tensor, state)

def write_attn(h_dec):
    with tf.variable_scope("writeW", reuse=tf.AUTO_REUSE):
        w = linear(h_dec, write_n * write_n)
    
    N = write_n
    w = tf.reshape(w, [batch_size, N, N])
    Fx, Fy, gamma = attn_window("write", h_dec, write_n)
    Fyt = tf.transpose(Fy, perm=[0, 2, 1])
    wr = tf.matmul(Fyt, tf.matmul(w, Fx))
    wr = tf.reshape(wr, [batch_size, B * A])
    return wr * tf.reshape(1.0 / gamma, [-1, 1])

# ============================================
# BUILD COMPUTATIONAL GRAPH
# ============================================

tf.reset_default_graph()

# Placeholders
x = tf.placeholder(tf.float32, shape=(batch_size, img_size))
e = tf.random_normal((batch_size, z_size), mean=0, stddev=1)

# RNN cells
lstm_enc = tf.contrib.rnn.LSTMCell(enc_size, state_is_tuple=True)
lstm_dec = tf.contrib.rnn.LSTMCell(dec_size, state_is_tuple=True)

read = read_attn
write = write_attn

# Unroll the model
cs = [0] * T
mus, logsigmas, sigmas = [0] * T, [0] * T, [0] * T
h_dec_prev = tf.zeros((batch_size, dec_size))
enc_state = lstm_enc.zero_state(batch_size, tf.float32)
dec_state = lstm_dec.zero_state(batch_size, tf.float32)

for t in range(T):
    c_prev = tf.zeros((batch_size, img_size)) if t == 0 else cs[t-1]
    x_hat = x - tf.sigmoid(c_prev)
    r = read(x, x_hat, h_dec_prev)
    h_enc, enc_state = encode(enc_state, tf.concat([r, h_dec_prev], 1))
    z, mus[t], logsigmas[t], sigmas[t] = sampleQ(h_enc)
    h_dec, dec_state = decode(dec_state, z)
    cs[t] = c_prev + write(h_dec)
    h_dec_prev = h_dec

# ============================================
# LOSS FUNCTIONS (FIXED WITH SCALING)
# ============================================

def binary_crossentropy(t, o):
    return -(t * tf.log(o + eps) + (1.0 - t) * tf.log(1.0 - o + eps))

x_recons = tf.nn.sigmoid(cs[-1])

# FIXED: Scale loss appropriately
Lx = reconstruction_weight * tf.reduce_mean(tf.reduce_sum(binary_crossentropy(x, x_recons), 1))

kl_terms = [0] * T
for t in range(T):
    mu2 = tf.square(mus[t])
    sigma2 = tf.square(sigmas[t])
    logsigma = logsigmas[t]
    kl_terms[t] = 0.5 * tf.reduce_sum(mu2 + sigma2 - 2 * logsigma, 1) - 0.5

KL = tf.add_n(kl_terms)
Lz = tf.reduce_mean(KL)
cost = Lx + Lz

# ============================================
# OPTIMIZER (FIXED GRADIENT CLIPPING)
# ============================================

optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
gvs = optimizer.compute_gradients(cost)

# FIXED: Better gradient clipping
gradients, variables = zip(*gvs)
clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
train_op = optimizer.apply_gradients(zip(clipped_gradients, variables))

# ============================================
# TRAINING LOOP
# ============================================

def train():
    data_file = "draw_plate_data_prepared_combined\license_plates_combined_96x32.npy"
    train_data = LicensePlateData(data_file, batch_size)
    
    actual_batch_size = min(batch_size, train_data.num_samples)
    
    config = tf.ConfigProto(device_count={'GPU': 0}, intra_op_parallelism_threads=4)
    sess = tf.InteractiveSession(config=config)
    saver = tf.train.Saver(max_to_keep=2)
    tf.global_variables_initializer().run()
    
    print("\n🚀 Starting training...")
    print("=" * 60)
    
    Lxs, Lzs = [], []
    
    for i in range(train_iters):
        xtrain = train_data.next_batch(actual_batch_size)
        
        if len(xtrain) < actual_batch_size:
            continue
        
        feed_dict = {x: xtrain}
        lx, lz, _ = sess.run([Lx, Lz, train_op], feed_dict)
        
        Lxs.append(lx)
        Lzs.append(lz)
        
        if i % 100 == 0:
            print(f"iter {i:5d}: Lx = {lx:.4f}, Lz = {lz:.4f}, Total = {lx + lz:.4f}")
    
    print("\n✅ Training complete!")
    
    # Save results
    out_file = "draw_plates_results.npy"
    np.save(out_file, [np.array(Lxs), np.array(Lzs)])
    print(f"💾 Results saved to: {out_file}")
    
    ckpt_file = "draw_plates_model.ckpt"
    saver.save(sess, ckpt_file)
    print(f"💾 Model saved to: {ckpt_file}")
    
    sess.close()

if __name__ == "__main__":
    train()