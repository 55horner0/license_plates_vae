#!/usr/bin/env python
"""DRAW model for license plates — TF1, rectangular attention."""

import tensorflow as tf
import numpy as np
import os

# ============================================================
# MODEL PARAMETERS
# ============================================================

A, B   = 96, 32          # image width, height
img_size = A * B          # 3072

# Architecture
# enc/dec 256 (not 512): 512 was oversized for ~1900 training images and
# caused slow convergence.  256 still fits the task while halving parameters.
enc_size  = 256
dec_size  = 256

# Rectangular attention grid — 3:1 aspect ratio matches the plate shape.
# 16×6 = 96 pixels per glimpse; at default delta=1 the grid spans the full
# image, so the model can zoom out to see the whole plate in one step or
# zoom in to individual characters.
read_n_x  = 16   # filters along width  (A=96)
read_n_y  = 6    # filters along height (B=32)
write_n_x = 16
write_n_y = 6

z_size    = 64   # latent dim per step — matches VAE baseline LATENT_DIM
T         = 50   # time steps; was 30 but the loss was still descending at 100
                 # epochs, so more refinement steps help

# Training
batch_size    = 32
N_EPOCHS      = 200
learning_rate = 1e-4  # matches VAE baseline LR
eps           = 1e-8
kl_weight     = 0.1   # final KL weight (matches VAE baseline)

# KL annealing: start at 0, ramp linearly to kl_weight over this many epochs.
# Without annealing the model collapses in the first few epochs — it discovers
# that keeping mu≈0 and sigma≈1 (matching the prior) eliminates the KL penalty
# entirely, so z_t carries no information and reconstruction plateaus early.
# Starting with beta=0 forces the encoder to use z_t for reconstruction first;
# the growing penalty then gradually regularises it toward the prior.
KL_WARMUP_EPOCHS = 30

# Free bits: enforce a minimum KL per latent dimension per step.
# If KL/dim < FREE_BITS the gradient through mu/sigma is zeroed out (via
# tf.maximum), which prevents posterior collapse even after the warmup ends —
# the model must encode at least FREE_BITS nats of information per dim per step.
FREE_BITS  = 0.15          # nats per latent dimension per time step
FREE_NATS  = FREE_BITS * z_size   # = 9.6 nats per step (vs current 3.0)

print("DRAW Model Configuration")
print(f"  Image     : {A}x{B} = {img_size} px")
print(f"  Grid      : read {read_n_x}x{read_n_y}, write {write_n_x}x{write_n_y}")
print(f"  LSTM      : enc={enc_size}, dec={dec_size}")
print(f"  z_size    : {z_size},  T={T}")
print(f"  kl_weight : {kl_weight} (warmed up over {KL_WARMUP_EPOCHS} epochs)")
print(f"  free_bits : {FREE_BITS} nats/dim/step  (free_nats={FREE_NATS:.1f}/step)")
print(f"  Epochs    : {N_EPOCHS},  lr={learning_rate}")

# ============================================================
# DATA LOADER
# ============================================================

# ============================================================
# DATA AUGMENTATION
# ============================================================

def augment_batch(batch):
    """Random intensity scale + small Gaussian noise applied per-image.

    Horizontal flips are intentionally skipped — flipping text makes it
    unreadable and introduces a distribution the model has never seen.
    Scale and noise are light enough not to destroy plate legibility but
    add enough variation to partially compensate for the tiny dataset.
    """
    rng   = np.random.default_rng()
    scale = rng.uniform(0.85, 1.15, size=(len(batch), 1)).astype(np.float32)
    noise = rng.normal(0.0, 0.025, size=batch.shape).astype(np.float32)
    return np.clip(batch * scale + noise, 0.0, 1.0)


class LicensePlateData:
    def __init__(self, data_path, batch_size):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data not found: {data_path}")
        self.data = np.load(data_path).astype(np.float32)
        if self.data.max() > 1.0:
            self.data /= 255.0
        self.num_samples  = len(self.data)
        self.batch_size   = min(batch_size, self.num_samples)
        self.index        = 0
        self._shuffle()
        print(f"Loaded {self.num_samples} images — "
              f"range [{self.data.min():.3f}, {self.data.max():.3f}]")

    def _shuffle(self):
        perm = np.random.permutation(self.num_samples)
        self.data = self.data[perm]

    def next_batch(self, n):
        end = self.index + n
        if end > self.num_samples:
            self._shuffle()
            self.index = 0
            end = n
        batch = self.data[self.index:end]
        self.index = end
        return batch


# ============================================================
# MODEL FUNCTIONS
# ============================================================

def linear(x, out_dim, scope):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        w = tf.get_variable("w", [x.get_shape()[1], out_dim])
        b = tf.get_variable("b", [out_dim],
                            initializer=tf.constant_initializer(0.0))
        return tf.matmul(x, w) + b


def filterbank(gx, gy, sigma2, delta_x, delta_y, Nx, Ny):
    """Build Gaussian read/write filter banks with separate x/y strides."""
    grid_x = tf.reshape(tf.cast(tf.range(Nx), tf.float32), [1, -1])
    grid_y = tf.reshape(tf.cast(tf.range(Ny), tf.float32), [1, -1])

    mu_x = gx + (grid_x - Nx / 2.0 - 0.5) * delta_x   # [batch, Nx]
    mu_y = gy + (grid_y - Ny / 2.0 - 0.5) * delta_y   # [batch, Ny]

    a = tf.reshape(tf.cast(tf.range(A), tf.float32), [1, 1, -1])
    b = tf.reshape(tf.cast(tf.range(B), tf.float32), [1, 1, -1])

    mu_x = tf.reshape(mu_x, [-1, Nx, 1])
    mu_y = tf.reshape(mu_y, [-1, Ny, 1])
    s2   = tf.reshape(sigma2, [-1, 1, 1])

    Fx = tf.exp(-tf.square(a - mu_x) / (2.0 * s2))   # [batch, Nx, A]
    Fy = tf.exp(-tf.square(b - mu_y) / (2.0 * s2))   # [batch, Ny, B]

    Fx = Fx / tf.maximum(tf.reduce_sum(Fx, 2, keep_dims=True), eps)
    Fy = Fy / tf.maximum(tf.reduce_sum(Fy, 2, keep_dims=True), eps)
    return Fx, Fy


def attn_window(scope, h_dec, Nx, Ny):
    """6-parameter attention: gx, gy, log_sigma2, log_delta_x, log_delta_y, log_gamma."""
    params = linear(h_dec, 6, scope=scope + "_params")
    gx_, gy_, log_sigma2, log_delta_x, log_delta_y, log_gamma = \
        tf.split(params, 6, axis=1)

    gx      = (A + 1) / 2.0 * (gx_ + 1)
    gy      = (B + 1) / 2.0 * (gy_ + 1)
    sigma2  = tf.exp(log_sigma2)
    delta_x = (A - 1) / (Nx - 1) * tf.exp(log_delta_x)
    delta_y = (B - 1) / (Ny - 1) * tf.exp(log_delta_y)
    gamma   = tf.exp(log_gamma)

    Fx, Fy = filterbank(gx, gy, sigma2, delta_x, delta_y, Nx, Ny)
    return Fx, Fy, gamma


def read_op(x, x_hat, h_dec_prev):
    Fx, Fy, gamma = attn_window("read", h_dec_prev, read_n_x, read_n_y)

    def glimpse(img):
        Fxt    = tf.transpose(Fx, [0, 2, 1])          # [batch, A, Nx]
        img    = tf.reshape(img, [-1, B, A])
        g      = tf.matmul(Fy, tf.matmul(img, Fxt))   # [batch, Ny, Nx]
        return tf.reshape(g, [-1, read_n_y * read_n_x]) * tf.reshape(gamma, [-1, 1])

    return tf.concat([glimpse(x), glimpse(x_hat)], axis=1)   # [batch, 2*Nx*Ny]


def encode(lstm_cell, state, inp):
    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
        return lstm_cell(inp, state)


def sampleQ(h_enc, e_t):
    """Sample z from the posterior q(z|x) using per-step independent noise e_t.

    BUG FIX: the previous version used a single global e = tf.random_normal(...)
    node, so every time step shared the same noise vector within one sess.run().
    That correlated the latent samples across T steps, preventing the iterative
    refinement from working independently.  Now each step receives its own e_t
    tensor created just-in-time inside the unrolling loop.
    """
    mu       = linear(h_enc, z_size, scope="mu")
    logsigma = linear(h_enc, z_size, scope="logsigma")
    sigma    = tf.exp(logsigma)
    return mu + sigma * e_t, mu, logsigma, sigma


def decode(lstm_cell, state, inp):
    with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
        return lstm_cell(inp, state)


def write_op(h_dec):
    w   = linear(h_dec, write_n_y * write_n_x, scope="write_patch")
    w   = tf.reshape(w, [batch_size, write_n_y, write_n_x])
    Fx, Fy, gamma = attn_window("write", h_dec, write_n_x, write_n_y)
    Fyt = tf.transpose(Fy, [0, 2, 1])                          # [batch, B, Ny]
    wr  = tf.matmul(Fyt, tf.matmul(w, Fx))                     # [batch, B, A]
    wr  = tf.reshape(wr, [batch_size, B * A])
    return wr * tf.reshape(1.0 / gamma, [-1, 1])


# ============================================================
# BUILD COMPUTATIONAL GRAPH
# ============================================================

tf.reset_default_graph()

x        = tf.placeholder(tf.float32, shape=(batch_size, img_size), name="x")
# Scalar placeholder for the annealed KL weight; defaults to kl_weight so
# the inference / plot graph works without feeding it.
kl_beta_ph = tf.placeholder_with_default(kl_weight, shape=(), name="kl_beta")

lstm_enc = tf.contrib.rnn.LSTMCell(enc_size, state_is_tuple=True)
lstm_dec = tf.contrib.rnn.LSTMCell(dec_size, state_is_tuple=True)

cs             = [0] * T
mus, logsigmas, sigmas = [0]*T, [0]*T, [0]*T

h_dec_prev = tf.zeros((batch_size, dec_size))
enc_state  = lstm_enc.zero_state(batch_size, tf.float32)
dec_state  = lstm_dec.zero_state(batch_size, tf.float32)

for t in range(T):
    c_prev = tf.zeros((batch_size, img_size)) if t == 0 else cs[t-1]
    x_hat  = x - tf.sigmoid(c_prev)
    r      = read_op(x, x_hat, h_dec_prev)
    h_enc, enc_state = encode(lstm_enc, enc_state,
                              tf.concat([r, h_dec_prev], axis=1))
    # Independent noise per time step — each call to tf.random_normal creates a
    # separate graph node, so e_t is freshly sampled at every step per sess.run().
    e_t    = tf.random_normal((batch_size, z_size), mean=0.0, stddev=1.0)
    z, mus[t], logsigmas[t], sigmas[t] = sampleQ(h_enc, e_t)
    h_dec, dec_state = decode(lstm_dec, dec_state, z)
    cs[t]  = c_prev + write_op(h_dec)
    h_dec_prev = h_dec

# ============================================================
# LOSS
# ============================================================

def binary_crossentropy(t, o):
    return -(t * tf.log(o + eps) + (1.0 - t) * tf.log(1.0 - o + eps))

x_recons = tf.nn.sigmoid(cs[-1])
Lx = tf.reduce_mean(tf.reduce_sum(binary_crossentropy(x, x_recons), axis=1))

kl_terms = []
for t in range(T):
    mu2    = tf.square(mus[t])
    sig2   = tf.square(sigmas[t])
    lsig   = logsigmas[t]
    # KL(N(mu,sigma^2) || N(0,1)) = 0.5*sum(mu^2 + sigma^2 - 2*log_sigma - 1)
    kl_t   = 0.5 * tf.reduce_sum(mu2 + sig2 - 2.0 * lsig - 1.0, axis=1)
    # Free bits: if KL for this step is below FREE_NATS the gradient through
    # mu/sigma is zero (tf.maximum is flat below the threshold).  This forces
    # the model to encode at least FREE_BITS nats per dim per step, directly
    # preventing posterior collapse after the KL warmup kicks in.
    kl_t   = tf.maximum(kl_t, FREE_NATS)
    kl_terms.append(kl_t)

KL   = tf.add_n(kl_terms)          # sum over T steps, shape [batch]
Lz   = tf.reduce_mean(KL)          # mean over batch
# kl_beta_ph starts at 0 (pure reconstruction) and is ramped up in the
# training loop — see KL_WARMUP_EPOCHS.
cost = Lx + kl_beta_ph * Lz

# ============================================================
# OPTIMIZER
# ============================================================

optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
grads, variables = zip(*optimizer.compute_gradients(cost))
clipped, _       = tf.clip_by_global_norm(grads, 5.0)
train_op         = optimizer.apply_gradients(zip(clipped, variables))

# ============================================================
# TRAINING
# ============================================================

def train():
    train_file = "draw_plate_data_prepared_combined/license_plates_combined_96x32.npy"
    val_file   = "draw_plate_data_prepared_combined/val_96x32.npy"

    train_data     = LicensePlateData(train_file, batch_size)
    iters_per_epoch = max(1, train_data.num_samples // batch_size)
    print(f"  {train_data.num_samples} train images, {iters_per_epoch} iters/epoch")

    val_data = None
    if os.path.exists(val_file):
        val_data = np.load(val_file).astype(np.float32)
        if val_data.max() > 1.0:
            val_data /= 255.0
        print(f"  {len(val_data)} val images")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess  = tf.InteractiveSession(config=config)
    saver = tf.train.Saver(max_to_keep=3)
    tf.global_variables_initializer().run()

    Lxs, Lzs = [], []  # per-iteration, for backward-compat with comparison notebook

    print(f"\nTraining for {N_EPOCHS} epochs ({N_EPOCHS * iters_per_epoch} iters)...")
    print("-" * 65)

    for epoch in range(N_EPOCHS):
        epoch_lx, epoch_lz = [], []

        for _ in range(iters_per_epoch):
            batch = train_data.next_batch(batch_size)
            if len(batch) < batch_size:
                continue
            lx, lz, _ = sess.run([Lx, Lz, train_op], {x: batch})
            Lxs.append(lx)
            Lzs.append(lz)
            epoch_lx.append(lx)
            epoch_lz.append(lz)

        train_lx = float(np.mean(epoch_lx))
        train_lz = float(np.mean(epoch_lz))

        # Validation Lx (reconstruction only, no train_op)
        val_info = ""
        if val_data is not None and (epoch + 1) % 10 == 0:
            vlxs = []
            for j in range(0, len(val_data) - batch_size + 1, batch_size):
                vb = val_data[j : j + batch_size]
                if len(vb) < batch_size:
                    break
                vlxs.append(sess.run(Lx, {x: vb}))
            val_info = f"  val_Lx={np.mean(vlxs):.1f}"

        if (epoch + 1) % 10 == 0:
            total = train_lx + kl_weight * train_lz
            print(f"Epoch {epoch+1:3d}/{N_EPOCHS}  "
                  f"Lx={train_lx:.1f}  Lz={train_lz:.1f}  "
                  f"total={total:.1f}{val_info}")

    print("\nTraining complete.")

    np.save("draw_plates_results.npy", [np.array(Lxs), np.array(Lzs)])
    print("Saved draw_plates_results.npy")

    saver.save(sess, "draw_plates_model.ckpt")
    print("Saved draw_plates_model.ckpt")

    sess.close()


if __name__ == "__main__":
    train()
