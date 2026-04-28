# plot_plates_results.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================
# MODEL PARAMETERS (MUST MATCH TRAINING)
# ============================================

A, B = 96, 32  # Width, Height
img_size = A * B
enc_size = 512
dec_size = 512
read_n = 12
write_n = 12
z_size = 10
T = 30
batch_size = 32
eps = 1e-8
read_attn = True
write_attn = True

# ============================================
# MODEL FUNCTIONS (COPY FROM YOUR TRAINING)
# ============================================

def linear(x, output_dim, scope=None):
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

def sampleQ(h_enc, e):  # FIXED: Pass e as parameter
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
# REBUILD GRAPH FOR INFERENCE
# ============================================

def load_and_visualize(model_path, data_path=None, num_samples=8):
    """Load trained model and generate visualizations"""
    
    tf.reset_default_graph()
    
    # Placeholders - FIXED: Define e as a placeholder
    x = tf.placeholder(tf.float32, shape=(batch_size, img_size))
    e = tf.placeholder(tf.float32, shape=(batch_size, z_size))  # FIXED: e as placeholder
    
    # RNN cells
    global lstm_enc, lstm_dec
    lstm_enc = tf.contrib.rnn.LSTMCell(enc_size, state_is_tuple=True)
    lstm_dec = tf.contrib.rnn.LSTMCell(dec_size, state_is_tuple=True)
    
    read = read_attn
    write = write_attn
    
    # Unroll the model
    cs = [0] * T
    h_dec_prev = tf.zeros((batch_size, dec_size))
    enc_state = lstm_enc.zero_state(batch_size, tf.float32)
    dec_state = lstm_dec.zero_state(batch_size, tf.float32)
    
    for t in range(T):
        c_prev = tf.zeros((batch_size, img_size)) if t == 0 else cs[t-1]
        x_hat = x - tf.sigmoid(c_prev)
        r = read(x, x_hat, h_dec_prev)
        h_enc, enc_state = encode(enc_state, tf.concat([r, h_dec_prev], 1))
        z, _, _, _ = sampleQ(h_enc, e)  # FIXED: Pass e to sampleQ
        h_dec, dec_state = decode(dec_state, z)
        cs[t] = c_prev + write(h_dec)
        h_dec_prev = h_dec
    
    # Reconstruction
    x_recons = tf.nn.sigmoid(cs[-1])
    
    # Load model
    config = tf.ConfigProto(device_count={'GPU': 0})
    sess = tf.InteractiveSession(config=config)
    saver = tf.train.Saver()
    
    # Try to restore the model
    try:
        saver.restore(sess, model_path)
        print(f"✅ Model loaded from: {model_path}")
    except:
        # Try with .ckpt extension
        try:
            saver.restore(sess, model_path + ".ckpt")
            print(f"✅ Model loaded from: {model_path}.ckpt")
        except Exception as e:
            print(f"❌ Could not load model: {e}")
            print("Looking for files in current directory...")
            print(os.listdir("."))
            return None, None
    
    # Load or create test data
    if data_path and os.path.exists(data_path):
        print(f"📂 Loading test data from: {data_path}")
        test_data = np.load(data_path).astype(np.float32)
        if test_data.max() > 1.0:
            print(f"   Normalizing data (max was {test_data.max()})")
            test_data = test_data / 255.0
        np.random.shuffle(test_data)
        test_batch = test_data[:num_samples]
    else:
        print("⚠️ No test data provided, using random noise")
        test_batch = np.random.rand(num_samples, img_size).astype(np.float32)
    
    # Pad or truncate to match batch size
    if len(test_batch) < batch_size:
        test_batch = np.vstack([test_batch, np.zeros((batch_size - len(test_batch), img_size))])
    
    # Generate reconstructions - FIXED: Provide e feed_dict
    print("🎨 Generating reconstructions...")
    random_noise = np.random.normal(0, 1, (batch_size, z_size)).astype(np.float32)
    feed_dict = {
        x: test_batch[:batch_size],
        e: random_noise  # FIXED: Provide noise
    }
    canvases = sess.run(cs, feed_dict=feed_dict)
    reconstructions = 1.0 / (1.0 + np.exp(-np.array(canvases)))
    
    # ===== VISUALIZATION 1: Training Progress (if losses were saved) =====
    if os.path.exists("draw_plates_results.npy"):
        try:
            saved_data = np.load("draw_plates_results.npy", allow_pickle=True)
            if isinstance(saved_data, np.ndarray) and len(saved_data) >= 2:
                Lxs, Lzs = saved_data[0], saved_data[1]
                
                plt.figure(figsize=(12, 5))
                
                plt.subplot(1, 2, 1)
                plt.plot(Lxs, label='Reconstruction Loss (Lx)', alpha=0.7)
                plt.plot(Lzs, label='KL Loss (Lz)', alpha=0.7)
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.title('Training Losses')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.subplot(1, 2, 2)
                plt.plot(np.array(Lxs) + np.array(Lzs), label='Total Loss', color='green', alpha=0.7)
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.title('Total Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig('training_losses_plot.png', dpi=150)
                print("📊 Saved: training_losses_plot.png")
                plt.close()
        except Exception as e:
            print(f"⚠️ Could not plot losses: {e}")
    
    # ===== VISUALIZATION 2: Sample Reconstructions =====
    final_recon = reconstructions[-1][:num_samples]
    
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
    if num_samples == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(num_samples):
        # Original
        axes[0, i].imshow(test_batch[i].reshape(B, A), cmap='gray')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Reconstruction
        axes[1, i].imshow(final_recon[i].reshape(B, A), cmap='gray')
        axes[1, i].set_title(f'Recon {i+1}')
        axes[1, i].axis('off')
    
    plt.suptitle(f'License Plate Reconstructions (Final Model)', fontsize=14)
    plt.tight_layout()
    plt.savefig('reconstructions_final.png', dpi=150)
    print("🖼️ Saved: reconstructions_final.png")
    plt.close()
    
    # ===== VISUALIZATION 3: Progressive Refinement =====
    time_steps_to_show = [0, T//4, T//2, 3*T//4, T-1]
    
    fig, axes = plt.subplots(len(time_steps_to_show), min(4, num_samples), 
                              figsize=(min(4, num_samples) * 2, len(time_steps_to_show) * 2))
    
    for row, t in enumerate(time_steps_to_show):
        recon_at_t = 1.0 / (1.0 + np.exp(-canvases[t]))[:min(4, num_samples)]
        for col in range(min(4, num_samples)):
            axes[row, col].imshow(recon_at_t[col].reshape(B, A), cmap='gray')
            if col == 0:
                axes[row, col].set_ylabel(f'Step {t+1}', fontsize=10)
            axes[row, col].axis('off')
    
    plt.suptitle('Progressive Refinement Over Time Steps', fontsize=14)
    plt.tight_layout()
    plt.savefig('progressive_refinement.png', dpi=150)
    print("📈 Saved: progressive_refinement.png")
    plt.close()
    
    # ===== VISUALIZATION 4: Side-by-Side Comparison Grid =====
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            ax.imshow(test_batch[i].reshape(B, A), cmap='gray')
            ax.set_title(f'Original {i+1}', fontsize=8)
        else:
            ax.axis('off')
    plt.suptitle('Sample Test Images', fontsize=12)
    plt.tight_layout()
    plt.savefig('test_samples.png', dpi=150)
    print("📸 Saved: test_samples.png")
    plt.close()
    
    # ===== VISUALIZATION 5: Reconstruction Error Map =====
    for i in range(min(3, num_samples)):
        original = test_batch[i].reshape(B, A)
        reconstructed = final_recon[i].reshape(B, A)
        error_map = np.abs(original - reconstructed)
        
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(reconstructed, cmap='gray')
        axes[1].set_title('Reconstructed')
        axes[1].axis('off')
        
        im = axes[2].imshow(error_map, cmap='hot', vmin=0, vmax=1)
        axes[2].set_title('Error Map')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2])
        
        plt.suptitle(f'Reconstruction Error - Sample {i+1}')
        plt.tight_layout()
        plt.savefig(f'reconstruction_error_{i+1}.png', dpi=150)
        plt.close()
    print("🔥 Saved: reconstruction error maps")
    
    sess.close()
    print("\n✅ All visualizations complete!")
    
    return reconstructions, canvases

# ============================================
# QUICK LOSS PLOT ONLY (Simplest version)
# ============================================

def plot_losses_only():
    """Just plot the losses from saved results file"""
    try:
        data = np.load("draw_plates_results.npy", allow_pickle=True)
        Lxs, Lzs = data[0], data[1]
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(Lxs, label='Reconstruction Loss (Lx)', linewidth=1)
        plt.plot(Lzs, label='KL Loss (Lz)', linewidth=1)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(np.array(Lxs) + np.array(Lzs), label='Total Loss', color='green', linewidth=1)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Total Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('loss_plot.png', dpi=150)
        plt.show()
        print("✅ Loss plot saved to loss_plot.png")
        
    except Exception as e:
        print(f"❌ Could not load losses: {e}")
        print("Make sure draw_plates_results.npy exists")

# ============================================
# RUN VISUALIZATION
# ============================================

if __name__ == "__main__":
    # Paths (adjust as needed)
    MODEL_PATH = "draw_plates_model.ckpt"  # Your saved model
    DATA_PATH = "draw_plate_data_prepared_rectangle/license_plates_96x32.npy"
    
    # Option 1: Just plot losses (if you don't have the model)
    # plot_losses_only()
    
    # Option 2: Full visualization with model
    if os.path.exists(MODEL_PATH + ".index") or os.path.exists(MODEL_PATH):
        reconstructions, canvases = load_and_visualize(MODEL_PATH, DATA_PATH, num_samples=8)
    else:
        print(f"⚠️ Model not found at: {MODEL_PATH}")
        print("Generating loss plot only...")
        plot_losses_only()