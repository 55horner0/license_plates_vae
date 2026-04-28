# plot_plates_results.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================
# MODEL PARAMETERS — MUST MATCH TRAINING
# ============================================

A, B = 96, 32
img_size = A * B
enc_size = 512
dec_size = 512

# FIX 4: Rectangular grid sizes matching the fixed training file
read_n_x  = 12
read_n_y  = 4
write_n_x = 12
write_n_y = 4

# FIX 7: Latent dimension matches training
z_size = 64

T          = 30
batch_size = 32
eps        = 1e-8

# FIX 3: KL weight matches training so the loss plot total is correct
kl_weight  = 0.1

# ============================================
# MODEL FUNCTIONS — MIRRORS FIXED TRAINING FILE
# ============================================

def linear(x, output_dim, scope=None):
    with tf.variable_scope(scope or "linear"):
        w = tf.get_variable("w", [x.get_shape()[1], output_dim])
        b = tf.get_variable("b", [output_dim],
                            initializer=tf.constant_initializer(0.0))
        return tf.matmul(x, w) + b


def filterbank(gx, gy, sigma2, delta_x, delta_y, Nx, Ny):
    # FIX 4: Separate grids and strides per axis
    grid_x = tf.reshape(tf.cast(tf.range(Nx), tf.float32), [1, -1])
    grid_y = tf.reshape(tf.cast(tf.range(Ny), tf.float32), [1, -1])

    mu_x = gx + (grid_x - Nx / 2.0 - 0.5) * delta_x
    mu_y = gy + (grid_y - Ny / 2.0 - 0.5) * delta_y

    a = tf.reshape(tf.cast(tf.range(A), tf.float32), [1, 1, -1])
    b = tf.reshape(tf.cast(tf.range(B), tf.float32), [1, 1, -1])

    mu_x = tf.reshape(mu_x, [-1, Nx, 1])
    mu_y = tf.reshape(mu_y, [-1, Ny, 1])

    sigma2_r = tf.reshape(sigma2, [-1, 1, 1])

    Fx = tf.exp(-tf.square(a - mu_x) / (2 * sigma2_r))
    Fy = tf.exp(-tf.square(b - mu_y) / (2 * sigma2_r))

    Fx = Fx / tf.maximum(tf.reduce_sum(Fx, 2, keep_dims=True), eps)
    Fy = Fy / tf.maximum(tf.reduce_sum(Fy, 2, keep_dims=True), eps)
    return Fx, Fy


def attn_window(scope, h_dec, Nx, Ny):
    # FIX 4: 6 params — separate delta_x and delta_y
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        params = linear(h_dec, 6, scope="attn_params")

    gx_, gy_, log_sigma2, log_delta_x, log_delta_y, log_gamma = \
        tf.split(params, 6, 1)

    gx      = (A + 1) / 2.0 * (gx_ + 1)
    gy      = (B + 1) / 2.0 * (gy_ + 1)
    sigma2  = tf.exp(log_sigma2)
    delta_x = (A - 1) / (Nx - 1) * tf.exp(log_delta_x)
    delta_y = (B - 1) / (Ny - 1) * tf.exp(log_delta_y)
    gamma   = tf.exp(log_gamma)

    Fx, Fy = filterbank(gx, gy, sigma2, delta_x, delta_y, Nx, Ny)
    return Fx, Fy, gamma


def read_attn_fn(x, x_hat, h_dec_prev):
    # FIX 4: Rectangular Nx x Ny glimpse
    Fx, Fy, gamma = attn_window("read", h_dec_prev, read_n_x, read_n_y)

    def filter_img(img, Fx, Fy, gamma, Nx, Ny):
        Fxt    = tf.transpose(Fx, perm=[0, 2, 1])
        img    = tf.reshape(img, [-1, B, A])
        glimpse = tf.matmul(Fy, tf.matmul(img, Fxt))
        glimpse = tf.reshape(glimpse, [-1, Ny * Nx])
        return glimpse * tf.reshape(gamma, [-1, 1])

    x_filtered     = filter_img(x,     Fx, Fy, gamma, read_n_x, read_n_y)
    x_hat_filtered = filter_img(x_hat, Fx, Fy, gamma, read_n_x, read_n_y)
    return tf.concat([x_filtered, x_hat_filtered], 1)


def encode(state, input_tensor):
    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
        return lstm_enc(input_tensor, state)


def sampleQ(h_enc, e):
    with tf.variable_scope("mu", reuse=tf.AUTO_REUSE):
        mu = linear(h_enc, z_size)
    with tf.variable_scope("sigma", reuse=tf.AUTO_REUSE):
        logsigma = linear(h_enc, z_size)
        sigma    = tf.exp(logsigma)
    return mu + sigma * e, mu, logsigma, sigma


def decode(state, input_tensor):
    with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
        return lstm_dec(input_tensor, state)


def write_attn_fn(h_dec):
    # FIX 4: Rectangular Ny x Nx write patch
    with tf.variable_scope("writeW", reuse=tf.AUTO_REUSE):
        w = linear(h_dec, write_n_y * write_n_x)

    w   = tf.reshape(w, [batch_size, write_n_y, write_n_x])
    Fx, Fy, gamma = attn_window("write", h_dec, write_n_x, write_n_y)

    Fyt = tf.transpose(Fy, perm=[0, 2, 1])
    wr  = tf.matmul(Fyt, tf.matmul(w, Fx))
    wr  = tf.reshape(wr, [batch_size, B * A])
    return wr * tf.reshape(1.0 / gamma, [-1, 1])


# ============================================
# GRAPH BUILDER
# ============================================

def build_graph():
    global lstm_enc, lstm_dec

    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, shape=(batch_size, img_size), name="x")
    e = tf.placeholder(tf.float32, shape=(batch_size, z_size),   name="e")

    lstm_enc = tf.contrib.rnn.LSTMCell(enc_size, state_is_tuple=True)
    lstm_dec = tf.contrib.rnn.LSTMCell(dec_size, state_is_tuple=True)

    cs          = [0] * T
    h_dec_prev  = tf.zeros((batch_size, dec_size))
    enc_state   = lstm_enc.zero_state(batch_size, tf.float32)
    dec_state   = lstm_dec.zero_state(batch_size, tf.float32)

    for t in range(T):
        c_prev = tf.zeros((batch_size, img_size)) if t == 0 else cs[t - 1]
        x_hat  = x - tf.sigmoid(c_prev)
        r      = read_attn_fn(x, x_hat, h_dec_prev)
        h_enc, enc_state = encode(enc_state,
                                   tf.concat([r, h_dec_prev], 1))
        z, _, _, _       = sampleQ(h_enc, e)
        h_dec, dec_state = decode(dec_state, z)
        cs[t]            = c_prev + write_attn_fn(h_dec)
        h_dec_prev       = h_dec

    x_recons = tf.nn.sigmoid(cs[-1])
    return x, e, cs, x_recons


def load_and_visualize(model_path, data_path=None, num_samples=8):

    x, e, cs, x_recons = build_graph()

    config = tf.ConfigProto(device_count={'GPU': 0})
    sess   = tf.InteractiveSession(config=config)
    saver  = tf.train.Saver()

    # --- load checkpoint ---
    loaded = False
    for path in [model_path, model_path + ".ckpt"]:
        try:
            saver.restore(sess, path)
            print(f"✅ Model loaded from: {path}")
            loaded = True
            break
        except Exception:
            pass
    if not loaded:
        print("Could not load model. Files in current dir:")
        print(os.listdir("."))
        return None, None

    # --- load test data ---
    if data_path and os.path.exists(data_path):
        test_data = np.load(data_path).astype(np.float32)
        if test_data.max() > 1.0:
            test_data /= 255.0
        np.random.shuffle(test_data)
        test_batch = test_data[:batch_size]
    else:
        print("No test data — using random noise")
        test_batch = np.random.rand(batch_size, img_size).astype(np.float32)

    if len(test_batch) < batch_size:
        pad = np.zeros((batch_size - len(test_batch), img_size),
                       dtype=np.float32)
        test_batch = np.vstack([test_batch, pad])

    noise     = np.random.normal(0, 1, (batch_size, z_size)).astype(np.float32)
    feed_dict = {x: test_batch, e: noise}

    print("Running inference...")
    canvases        = sess.run(cs, feed_dict=feed_dict)
    # sigmoid applied manually so we can inspect every timestep
    reconstructions = 1.0 / (1.0 + np.exp(-np.array(canvases)))

    # ── Plot 1: Training losses ──────────────────────────────────────────
    if os.path.exists("draw_plates_results.npy"):
        try:
            saved   = np.load("draw_plates_results.npy", allow_pickle=True)
            Lxs, Lzs = saved[0], saved[1]

            # FIX 3: total uses kl_weight, not raw Lz
            total = np.array(Lxs) + kl_weight * np.array(Lzs)

            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(Lxs,  label='Lx (reconstruction)', alpha=0.8)
            plt.plot(Lzs,  label='Lz (KL)',             alpha=0.8)
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Training losses (raw)')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(1, 2, 2)
            plt.plot(total, label=f'Lx + {kl_weight}·Lz', color='green',
                     alpha=0.8)
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Total loss (as trained)')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('training_losses_plot.png', dpi=150)
            print("Saved: training_losses_plot.png")
            plt.close()
        except Exception as ex:
            print(f"Could not plot losses: {ex}")

    # ── Plot 2: Reconstructions ──────────────────────────────────────────
    n_show      = min(num_samples, batch_size)
    final_recon = reconstructions[-1][:n_show]

    fig, axes = plt.subplots(2, n_show, figsize=(n_show * 2, 4))
    if n_show == 1:
        axes = axes.reshape(-1, 1)

    for i in range(n_show):
        axes[0, i].imshow(test_batch[i].reshape(B, A), cmap='gray',
                          vmin=0, vmax=1)
        axes[0, i].set_title(f'Orig {i+1}', fontsize=8)
        axes[0, i].axis('off')

        axes[1, i].imshow(final_recon[i].reshape(B, A), cmap='gray',
                          vmin=0, vmax=1)
        axes[1, i].set_title(f'Recon {i+1}', fontsize=8)
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('Original',      fontsize=9)
    axes[1, 0].set_ylabel('Reconstructed', fontsize=9)
    plt.suptitle('License Plate Reconstructions', fontsize=13)
    plt.tight_layout()
    plt.savefig('reconstructions_final.png', dpi=150)
    print("Saved: reconstructions_final.png")
    plt.close()

    # ── Plot 3: Progressive refinement ──────────────────────────────────
    steps_to_show = [0, T // 4, T // 2, 3 * T // 4, T - 1]
    n_cols        = min(4, n_show)

    fig, axes = plt.subplots(len(steps_to_show), n_cols,
                              figsize=(n_cols * 2,
                                       len(steps_to_show) * 2))

    for row, t in enumerate(steps_to_show):
        recon_t = 1.0 / (1.0 + np.exp(-canvases[t]))
        for col in range(n_cols):
            axes[row, col].imshow(recon_t[col].reshape(B, A),
                                  cmap='gray', vmin=0, vmax=1)
            if col == 0:
                axes[row, col].set_ylabel(f'Step {t+1}', fontsize=9)
            axes[row, col].axis('off')

    plt.suptitle('Progressive canvas refinement over T steps', fontsize=13)
    plt.tight_layout()
    plt.savefig('progressive_refinement.png', dpi=150)
    print("Saved: progressive_refinement.png")
    plt.close()

    # ── Plot 4: Error maps ───────────────────────────────────────────────
    for i in range(min(3, n_show)):
        orig  = test_batch[i].reshape(B, A)
        recon = final_recon[i].reshape(B, A)
        err   = np.abs(orig - recon)

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(orig,  cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Original');      axes[0].axis('off')
        axes[1].imshow(recon, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Reconstructed'); axes[1].axis('off')
        im = axes[2].imshow(err, cmap='hot', vmin=0, vmax=1)
        axes[2].set_title('Error map');     axes[2].axis('off')
        plt.colorbar(im, ax=axes[2])
        plt.suptitle(f'Reconstruction error — sample {i+1}')
        plt.tight_layout()
        plt.savefig(f'reconstruction_error_{i+1}.png', dpi=150)
        plt.close()
    print("🔥 Saved: reconstruction error maps")

    sess.close()
    print("\n✅ All visualisations complete!")
    return reconstructions, canvases


# ============================================
# QUICK LOSS PLOT ONLY
# ============================================

def plot_losses_only():
    try:
        data      = np.load("draw_plates_results.npy", allow_pickle=True)
        Lxs, Lzs = data[0], data[1]

        # FIX 3: correct weighted total
        total = np.array(Lxs) + kl_weight * np.array(Lzs)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(Lxs, label='Lx (reconstruction)', linewidth=1)
        plt.plot(Lzs, label='Lz (KL)',             linewidth=1)
        plt.xlabel('Iteration'); plt.ylabel('Loss')
        plt.title('Training losses (raw)')
        plt.legend(); plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(total, label=f'Lx + {kl_weight}·Lz',
                 color='green', linewidth=1)
        plt.xlabel('Iteration'); plt.ylabel('Loss')
        plt.title('Total loss (as trained)')
        plt.legend(); plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('loss_plot.png', dpi=150)
        plt.show()
        print("Saved: loss_plot.png")
    except Exception as ex:
        print(f"Could not load losses: {ex}")


# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    MODEL_PATH = "draw_plates_model.ckpt"
    DATA_PATH  = r"draw_plate_data_prepared_combined\license_plates_combined_96x32.npy"

    if os.path.exists(MODEL_PATH + ".index") or os.path.exists(MODEL_PATH):
        load_and_visualize(MODEL_PATH, DATA_PATH, num_samples=8)
    else:
        print(f"Model not found at: {MODEL_PATH}")
        print("Plotting losses only...")
        plot_losses_only()