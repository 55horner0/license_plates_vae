#!/usr/bin/env python
"""Visualise trained DRAW model results.

IMPORTANT: all parameters below must exactly match draw_plates_rectangle2.py
so that the checkpoint variables can be restored into this graph.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# ============================================================
# PARAMETERS — MUST MATCH draw_plates_rectangle2.py
# ============================================================

A, B      = 96, 32
img_size  = A * B

enc_size  = 256
dec_size  = 256
read_n_x  = 16
read_n_y  = 6
write_n_x = 16
write_n_y = 6
z_size    = 64
T         = 50
batch_size = 32
eps        = 1e-8
kl_weight  = 0.1

# ============================================================
# MODEL FUNCTIONS — mirrors draw_plates_rectangle2.py exactly
# ============================================================

def linear(x, out_dim, scope):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        w = tf.get_variable("w", [x.get_shape()[1], out_dim])
        b = tf.get_variable("b", [out_dim],
                            initializer=tf.constant_initializer(0.0))
        return tf.matmul(x, w) + b


def filterbank(gx, gy, sigma2, delta_x, delta_y, Nx, Ny):
    grid_x = tf.reshape(tf.cast(tf.range(Nx), tf.float32), [1, -1])
    grid_y = tf.reshape(tf.cast(tf.range(Ny), tf.float32), [1, -1])
    mu_x   = gx + (grid_x - Nx / 2.0 - 0.5) * delta_x
    mu_y   = gy + (grid_y - Ny / 2.0 - 0.5) * delta_y
    a      = tf.reshape(tf.cast(tf.range(A), tf.float32), [1, 1, -1])
    b      = tf.reshape(tf.cast(tf.range(B), tf.float32), [1, 1, -1])
    mu_x   = tf.reshape(mu_x, [-1, Nx, 1])
    mu_y   = tf.reshape(mu_y, [-1, Ny, 1])
    s2     = tf.reshape(sigma2, [-1, 1, 1])
    Fx     = tf.exp(-tf.square(a - mu_x) / (2.0 * s2))
    Fy     = tf.exp(-tf.square(b - mu_y) / (2.0 * s2))
    Fx     = Fx / tf.maximum(tf.reduce_sum(Fx, 2, keep_dims=True), eps)
    Fy     = Fy / tf.maximum(tf.reduce_sum(Fy, 2, keep_dims=True), eps)
    return Fx, Fy


def attn_window(scope, h_dec, Nx, Ny):
    params  = linear(h_dec, 6, scope=scope + "_params")
    gx_, gy_, log_sigma2, log_delta_x, log_delta_y, log_gamma = \
        tf.split(params, 6, axis=1)
    gx      = (A + 1) / 2.0 * (gx_ + 1)
    gy      = (B + 1) / 2.0 * (gy_ + 1)
    sigma2  = tf.exp(log_sigma2)
    delta_x = (A - 1) / (Nx - 1) * tf.exp(log_delta_x)
    delta_y = (B - 1) / (Ny - 1) * tf.exp(log_delta_y)
    gamma   = tf.exp(log_gamma)
    Fx, Fy  = filterbank(gx, gy, sigma2, delta_x, delta_y, Nx, Ny)
    return Fx, Fy, gamma


def read_op(x_in, x_hat, h_dec_prev):
    Fx, Fy, gamma = attn_window("read", h_dec_prev, read_n_x, read_n_y)
    def glimpse(img):
        Fxt = tf.transpose(Fx, [0, 2, 1])
        img = tf.reshape(img, [-1, B, A])
        g   = tf.matmul(Fy, tf.matmul(img, Fxt))
        return tf.reshape(g, [-1, read_n_y * read_n_x]) * tf.reshape(gamma, [-1, 1])
    return tf.concat([glimpse(x_in), glimpse(x_hat)], axis=1)


def sampleQ(h_enc, e_t):
    mu       = linear(h_enc, z_size, scope="mu")
    logsigma = linear(h_enc, z_size, scope="logsigma")
    sigma    = tf.exp(logsigma)
    return mu + sigma * e_t, mu, logsigma, sigma


def write_op(h_dec):
    w   = linear(h_dec, write_n_y * write_n_x, scope="write_patch")
    w   = tf.reshape(w, [batch_size, write_n_y, write_n_x])
    Fx, Fy, gamma = attn_window("write", h_dec, write_n_x, write_n_y)
    Fyt = tf.transpose(Fy, [0, 2, 1])
    wr  = tf.matmul(Fyt, tf.matmul(w, Fx))
    wr  = tf.reshape(wr, [batch_size, B * A])
    return wr * tf.reshape(1.0 / gamma, [-1, 1])


# ============================================================
# GRAPH BUILDER
# ============================================================

def build_graph():
    global lstm_enc, lstm_dec

    tf.reset_default_graph()
    x_ph = tf.placeholder(tf.float32, shape=(batch_size, img_size), name="x")

    lstm_enc = tf.contrib.rnn.LSTMCell(enc_size, state_is_tuple=True)
    lstm_dec = tf.contrib.rnn.LSTMCell(dec_size, state_is_tuple=True)

    cs         = [None] * T
    h_dec_prev = tf.zeros((batch_size, dec_size))
    enc_state  = lstm_enc.zero_state(batch_size, tf.float32)
    dec_state  = lstm_dec.zero_state(batch_size, tf.float32)

    for t in range(T):
        c_prev = tf.zeros((batch_size, img_size)) if t == 0 else cs[t-1]
        x_hat  = x_ph - tf.sigmoid(c_prev)
        r      = read_op(x_ph, x_hat, h_dec_prev)
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            h_enc, enc_state = lstm_enc(tf.concat([r, h_dec_prev], axis=1),
                                        enc_state)
        e_t    = tf.random_normal((batch_size, z_size))
        z, _, _, _ = sampleQ(h_enc, e_t)
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            h_dec, dec_state = lstm_dec(z, dec_state)
        cs[t]      = c_prev + write_op(h_dec)
        h_dec_prev = h_dec

    x_recons = tf.nn.sigmoid(cs[-1])
    return x_ph, cs, x_recons


# ============================================================
# HELPERS
# ============================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def load_data(data_path, n=batch_size):
    data = np.load(data_path).astype(np.float32)
    if data.max() > 1.0:
        data /= 255.0
    np.random.shuffle(data)
    batch = data[:n]
    if len(batch) < n:
        pad   = np.zeros((n - len(batch), img_size), dtype=np.float32)
        batch = np.vstack([batch, pad])
    return batch


# ============================================================
# PLOTS
# ============================================================

def plot_losses(save=True):
    """Training curves: per-epoch smoothed Lx, Lz, and total."""
    if not os.path.exists("draw_plates_results.npy"):
        print("draw_plates_results.npy not found — skipping loss plot.")
        return

    data      = np.load("draw_plates_results.npy", allow_pickle=True)
    Lxs, Lzs = np.array(data[0]), np.array(data[1])

    # Fold per-iteration losses to per-epoch means
    # iters_per_epoch = combined_dataset (1872) // batch_size (32) = 58
    iters_per_epoch = 1872 // batch_size
    n_epochs        = len(Lxs) // iters_per_epoch
    trim            = n_epochs * iters_per_epoch
    lx_ep = Lxs[:trim].reshape(n_epochs, iters_per_epoch).mean(axis=1)
    lz_ep = Lzs[:trim].reshape(n_epochs, iters_per_epoch).mean(axis=1)
    tot_ep = lx_ep + kl_weight * lz_ep
    epochs = np.arange(1, n_epochs + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, lx_ep, color="steelblue")
    axes[0].axhline(0.6931 * img_size, color="grey", linestyle="--",
                    alpha=0.6, label=f"random baseline ({0.6931*img_size:.0f})")
    axes[0].set_title("Reconstruction Loss (Lx)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("BCE")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, lz_ep, color="tomato")
    axes[1].set_title("KL Loss (Lz, summed over T steps)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("KL")
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, tot_ep, color="green")
    axes[2].set_title(f"Total Loss (Lx + {kl_weight}·Lz)")
    axes[2].set_xlabel("Epoch"); axes[2].grid(alpha=0.3)

    plt.suptitle("DRAW Training Curves (per-epoch averages)", fontsize=13)
    plt.tight_layout()
    if save:
        plt.savefig("training_losses_plot.png", dpi=150, bbox_inches="tight")
        print("Saved training_losses_plot.png")
    plt.show()


def plot_reconstructions(sess, x_ph, x_recons, test_batch, n_show=8, save=True):
    """Side-by-side originals vs reconstructions with error heatmaps."""
    feed     = {x_ph: test_batch}
    recon_np = sess.run(x_recons, feed)          # [batch, img_size]

    n_show = min(n_show, batch_size)
    fig, axes = plt.subplots(3, n_show, figsize=(n_show * 2, 5))

    maes = []
    for i in range(n_show):
        orig  = test_batch[i].reshape(B, A)
        recon = recon_np[i].reshape(B, A)
        err   = np.abs(orig - recon)
        maes.append(err.mean())

        axes[0, i].imshow(orig,  cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow(recon, cmap="gray", vmin=0, vmax=1)
        im = axes[2, i].imshow(err, cmap="hot", vmin=0, vmax=0.5)
        axes[2, i].set_title(f"MAE={err.mean():.3f}", fontsize=7)
        for row in range(3):
            axes[row, i].axis("off")

    axes[0, 0].set_ylabel("Original",      fontsize=9)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=9)
    axes[2, 0].set_ylabel("|Error|",        fontsize=9)
    plt.colorbar(im, ax=axes[2, -1], fraction=0.046, pad=0.04)
    plt.suptitle(f"DRAW Reconstructions  (mean MAE={np.mean(maes):.4f})", fontsize=12)
    plt.tight_layout()
    if save:
        plt.savefig("reconstructions_final.png", dpi=150, bbox_inches="tight")
        print("Saved reconstructions_final.png")
    plt.show()


def plot_progressive(sess, x_ph, cs, test_batch, n_cols=5, save=True):
    """Show how the canvas evolves across T time steps."""
    feed     = {x_ph: test_batch}
    canvases = sess.run(cs, feed)   # list of T arrays, each [batch, img_size]

    steps = [0, T // 5, 2 * T // 5, 3 * T // 5, 4 * T // 5, T - 1]
    steps = sorted(set(steps))      # deduplicate
    n_rows = len(steps) + 1         # +1 for originals
    n_cols = min(n_cols, batch_size)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2, n_rows * 1.8))

    for col in range(n_cols):
        axes[0, col].imshow(test_batch[col].reshape(B, A), cmap="gray",
                            vmin=0, vmax=1)
        if col == 0:
            axes[0, col].set_ylabel("Original", fontsize=8)
        axes[0, col].axis("off")

    for row, t in enumerate(steps):
        canvas = sigmoid(canvases[t])
        for col in range(n_cols):
            axes[row + 1, col].imshow(canvas[col].reshape(B, A), cmap="gray",
                                      vmin=0, vmax=1)
            if col == 0:
                axes[row + 1, col].set_ylabel(f"Step {t+1}", fontsize=8)
            axes[row + 1, col].axis("off")

    plt.suptitle("Canvas refinement over time steps", fontsize=12)
    plt.tight_layout()
    if save:
        plt.savefig("progressive_refinement.png", dpi=150, bbox_inches="tight")
        print("Saved progressive_refinement.png")
    plt.show()


def plot_comparison_with_vae(save=True):
    """Compare DRAW and VAE baseline training curves on one figure."""
    vae_json = os.path.join("..", "vae_baseline_metrics.json")
    draw_npy  = "draw_plates_results.npy"

    if not os.path.exists(draw_npy):
        print("draw_plates_results.npy not found — skipping comparison.")
        return

    import json
    draw_data = np.load(draw_npy, allow_pickle=True)
    Lxs, Lzs = np.array(draw_data[0]), np.array(draw_data[1])
    iters_per_epoch = 1872 // batch_size
    n_epochs_draw   = len(Lxs) // iters_per_epoch
    trim = n_epochs_draw * iters_per_epoch
    draw_lx_ep  = Lxs[:trim].reshape(n_epochs_draw, -1).mean(axis=1)
    draw_tot_ep = (Lxs[:trim] + kl_weight * Lzs[:trim]) \
                      .reshape(n_epochs_draw, -1).mean(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ep_draw = np.arange(1, n_epochs_draw + 1)

    axes[0].plot(ep_draw, draw_tot_ep, color="tomato", label="DRAW (train)")
    axes[1].plot(ep_draw, draw_lx_ep,  color="tomato", label="DRAW recon (train)")

    if os.path.exists(vae_json):
        with open(vae_json) as fh:
            vm = json.load(fh)
        ep_vae = np.arange(1, len(vm["train_losses"]) + 1)
        axes[0].plot(ep_vae, vm["train_losses"], color="steelblue", label="VAE (train)")
        axes[0].plot(ep_vae, vm["val_losses"],   color="steelblue", linestyle="--",
                     label="VAE (val)")
        axes[1].plot(ep_vae, vm["train_recons"], color="steelblue", label="VAE recon (train)")
        axes[1].plot(ep_vae, vm["val_recons"],   color="steelblue", linestyle="--",
                     label="VAE recon (val)")
        print(f"VAE baseline: final train loss={vm['train_losses'][-1]:.2f}, "
              f"val MAE={vm.get('val_mae', 'n/a')}")
    else:
        print(f"VAE metrics not found at {vae_json} — plotting DRAW only.")

    axes[0].set_title("Total Loss (ELBO)");    axes[0].set_xlabel("Epoch")
    axes[1].set_title("Reconstruction Loss");  axes[1].set_xlabel("Epoch")
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle("DRAW vs VAE Baseline", fontsize=13)
    plt.tight_layout()
    if save:
        plt.savefig("draw_vs_vae_comparison.png", dpi=150, bbox_inches="tight")
        print("Saved draw_vs_vae_comparison.png")
    plt.show()


# ============================================================
# MAIN
# ============================================================

def run(model_path="draw_plates_model.ckpt",
        data_path=r"draw_plate_data_prepared_combined\test_96x32.npy",
        n_show=8):

    # Always plot losses — doesn't need the model
    plot_losses()
    plot_comparison_with_vae()

    # Check model exists before building the graph
    if not (os.path.exists(model_path + ".index") or
            os.path.exists(model_path)):
        print(f"Model not found at {model_path} — skipping reconstruction plots.")
        return

    x_ph, cs_tensors, x_recons = build_graph()

    config = tf.ConfigProto(device_count={"GPU": 0})
    sess   = tf.InteractiveSession(config=config)
    saver  = tf.train.Saver()

    try:
        saver.restore(sess, model_path)
        print(f"Model loaded from {model_path}")
    except Exception as exc:
        print(f"Could not load model: {exc}")
        sess.close()
        return

    # Load test data
    if os.path.exists(data_path):
        test_batch = load_data(data_path, batch_size)
    else:
        fallback = r"draw_plate_data_prepared_combined\license_plates_combined_96x32.npy"
        print(f"Test split not found — using combined dataset ({fallback})")
        test_batch = load_data(fallback, batch_size)

    plot_reconstructions(sess, x_ph, x_recons, test_batch, n_show=n_show)
    plot_progressive(sess, x_ph, cs_tensors, test_batch, n_cols=min(5, n_show))

    sess.close()
    print("Done.")


if __name__ == "__main__":
    run()
