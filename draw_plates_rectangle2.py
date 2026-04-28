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
# FIX 4: Separate x/y grid sizes to match the 96×32 (3:1) aspect ratio.
# A square N×N patch on a 3:1 image wastes vertical resolution — 12×12 covers
# the full height in one step while barely touching 12% of the width.
# Using 12 filters along x and 4 along y keeps each filter's coverage proportional.
read_n_x  = 12   # filters along width  (A=96)
read_n_y  = 4    # filters along height (B=32)
write_n_x = 12   # filters along width  (A=96)
write_n_y = 4    # filters along height (B=32)
z_size = 64      # FIX 7 (applied here for clarity): increased from 10 — see below
T = 30           # INCREASED from 20

# Training parameters
batch_size = 32
# FIX 5: Switch from a fixed iteration count to epoch-based training so DRAW
# sees the full dataset as many times as the VAE (100 epochs).
# train_iters is now computed dynamically inside train() once dataset size is known.
N_EPOCHS = 100
learning_rate = 1e-4  # REDUCED from 1e-3
eps = 1e-8

# Attention flags
read_attn = True
write_attn = True

# FIX 3: Removed MNIST-derived reconstruction_weight (was 784/3072 ≈ 0.255).
# That coefficient cut the reconstruction gradient to 25% of its true value,
# making Lx nearly invisible to the optimiser relative to Lz.
# Instead we down-weight KL with kl_weight=0.1 in the cost (see loss section).
kl_weight = 0.1

print(f"🚀 DRAW Model Configuration:")
print(f"   Image size: {A}×{B} = {img_size} pixels")
print(f"   Read  grid: {read_n_x}×{read_n_y}  (x×y)")
print(f"   Write grid: {write_n_x}×{write_n_y}  (x×y)")
print(f"   KL weight:  {kl_weight} (reconstruction receives full weight)")
print(f"   z_size:     {z_size}")
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

def filterbank(gx, gy, sigma2, delta_x, delta_y, Nx, Ny):
    # FIX 4: Separate grid sizes and strides for x (width) and y (height).
    # Previously a single N and single delta were used for both axes, forcing
    # square patches regardless of the image aspect ratio.
    # Now Fx has shape [batch, Nx, A] and Fy has shape [batch, Ny, B],
    # so each axis is sampled at its own resolution.
    grid_x = tf.reshape(tf.cast(tf.range(Nx), tf.float32), [1, -1])  # [1, Nx]
    grid_y = tf.reshape(tf.cast(tf.range(Ny), tf.float32), [1, -1])  # [1, Ny]

    mu_x = gx + (grid_x - Nx / 2.0 - 0.5) * delta_x  # [batch, Nx]
    mu_y = gy + (grid_y - Ny / 2.0 - 0.5) * delta_y  # [batch, Ny]

    a = tf.reshape(tf.cast(tf.range(A), tf.float32), [1, 1, -1])   # [1, 1, A]
    b = tf.reshape(tf.cast(tf.range(B), tf.float32), [1, 1, -1])   # [1, 1, B]

    mu_x = tf.reshape(mu_x, [-1, Nx, 1])  # [batch, Nx, 1]
    mu_y = tf.reshape(mu_y, [-1, Ny, 1])  # [batch, Ny, 1]

    sigma2_x = tf.reshape(sigma2, [-1, 1, 1])
    sigma2_y = sigma2_x  # share the same sigma for both axes

    Fx = tf.exp(-tf.square(a - mu_x) / (2 * sigma2_x))  # [batch, Nx, A]
    Fy = tf.exp(-tf.square(b - mu_y) / (2 * sigma2_y))  # [batch, Ny, B]

    Fx = Fx / tf.maximum(tf.reduce_sum(Fx, 2, keep_dims=True), eps)
    Fy = Fy / tf.maximum(tf.reduce_sum(Fy, 2, keep_dims=True), eps)
    return Fx, Fy

def attn_window(scope, h_dec, Nx, Ny):
    """FIX 4: Now accepts Nx and Ny separately and produces per-axis strides.
    The original used a single delta and single N for both axes, which forced
    square attention regardless of image shape.  Two independent delta values
    let the x-axis stride cover 96 pixels and the y-axis stride cover 32 pixels
    at the same spatial density."""
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # 6 params: gx, gy, log_sigma2, log_delta_x, log_delta_y, log_gamma
        params = linear(h_dec, 6, scope="attn_params")

    gx_, gy_, log_sigma2, log_delta_x, log_delta_y, log_gamma = tf.split(params, 6, 1)

    gx = (A + 1) / 2.0 * (gx_ + 1)
    gy = (B + 1) / 2.0 * (gy_ + 1)
    sigma2   = tf.exp(log_sigma2)
    delta_x  = (A - 1) / (Nx - 1) * tf.exp(log_delta_x)  # stride in x
    delta_y  = (B - 1) / (Ny - 1) * tf.exp(log_delta_y)  # stride in y
    gamma    = tf.exp(log_gamma)

    Fx, Fy = filterbank(gx, gy, sigma2, delta_x, delta_y, Nx, Ny)
    return Fx, Fy, gamma

def read_attn(x, x_hat, h_dec_prev):
    # FIX 4: Pass separate Nx and Ny to attn_window.
    Fx, Fy, gamma = attn_window("read", h_dec_prev, read_n_x, read_n_y)

    def filter_img(img, Fx, Fy, gamma, Nx, Ny):
        # Fx: [batch, Nx, A],  Fy: [batch, Ny, B]
        # img reshaped to [batch, B, A] — height rows, width cols
        Fxt = tf.transpose(Fx, perm=[0, 2, 1])          # [batch, A, Nx]
        img = tf.reshape(img, [-1, B, A])                # [batch, B, A]
        # Fy @ img @ Fxt  →  [batch, Ny, A] @ [batch, A, Nx]  →  [batch, Ny, Nx]
        glimpse = tf.matmul(Fy, tf.matmul(img, Fxt))    # [batch, Ny, Nx]
        glimpse = tf.reshape(glimpse, [-1, Ny * Nx])     # [batch, Ny*Nx]
        return glimpse * tf.reshape(gamma, [-1, 1])

    x_filtered     = filter_img(x,     Fx, Fy, gamma, read_n_x, read_n_y)
    x_hat_filtered = filter_img(x_hat, Fx, Fy, gamma, read_n_x, read_n_y)
    return tf.concat([x_filtered, x_hat_filtered], 1)   # [batch, 2*Nx*Ny]

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
    # FIX 4: Write patch is now Ny×Nx (rectangular) instead of N×N.
    with tf.variable_scope("writeW", reuse=tf.AUTO_REUSE):
        w = linear(h_dec, write_n_y * write_n_x)        # [batch, Ny*Nx]

    w = tf.reshape(w, [batch_size, write_n_y, write_n_x])   # [batch, Ny, Nx]
    Fx, Fy, gamma = attn_window("write", h_dec, write_n_x, write_n_y)

    Fyt = tf.transpose(Fy, perm=[0, 2, 1])               # [batch, B, Ny]
    # Fyt @ w @ Fx  →  [batch, B, Ny] @ [batch, Ny, Nx] @ [batch, Nx, A]  →  [batch, B, A]
    wr = tf.matmul(Fyt, tf.matmul(w, Fx))                # [batch, B, A]
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

# FIX 3: Reconstruction loss now carries its full weight (no MNIST scaling).
# Previously multiplied by 0.255, which cut the reconstruction gradient to 25%.
# We keep Lx at full strength and apply kl_weight to Lz instead so the model
# is not penalised too heavily for having a non-standard prior early in training.
Lx = tf.reduce_mean(tf.reduce_sum(binary_crossentropy(x, x_recons), 1))

kl_terms = [0] * T
for t in range(T):
    mu2 = tf.square(mus[t])
    sigma2 = tf.square(sigmas[t])
    logsigma = logsigmas[t]
    kl_terms[t] = 0.5 * tf.reduce_sum(mu2 + sigma2 - 2 * logsigma, 1) - 0.5

KL = tf.add_n(kl_terms)
Lz = tf.reduce_mean(KL)
# FIX 3: KL is down-weighted so reconstruction dominates the gradient signal.
cost = Lx + kl_weight * Lz

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
    data_file = r"draw_plate_data_prepared_combined/license_plates_combined_96x32.npy"
    train_data = LicensePlateData(data_file, batch_size)

    actual_batch_size = min(batch_size, train_data.num_samples)

    # FIX 5: Compute iterations from dataset size so every epoch = one full pass.
    iters_per_epoch = max(1, train_data.num_samples // actual_batch_size)
    train_iters = N_EPOCHS * iters_per_epoch
    print(f"📊 Dataset: {train_data.num_samples} images  |  "
          f"{iters_per_epoch} iters/epoch  |  {train_iters} total iters ({N_EPOCHS} epochs)")

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

        epoch = i // iters_per_epoch
        if i % iters_per_epoch == 0:
            print(f"Epoch {epoch+1:3d}/{N_EPOCHS} (iter {i:6d}): "
                  f"Lx = {lx:.4f}, Lz = {lz:.4f}, Total = {lx + kl_weight*lz:.4f}")

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