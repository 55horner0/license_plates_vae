"""Shared hyperparameters for the VAE and DRAW VAE experiments."""

# --- Image dimensions ---
IMG_HEIGHT = 32
IMG_WIDTH = 64
IMG_CHANNELS = 1  # grayscale

# --- Training ---
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
SEED = 42

# --- Dataset ---
NUM_TRAIN = 8000
NUM_VAL = 1000
NUM_TEST = 1000

# --- Standard VAE ---
VAE_LATENT_DIM = 128
VAE_HIDDEN_DIMS = [32, 64, 128]

# --- DRAW VAE ---
DRAW_T = 10          # number of sequential read/write steps
DRAW_Z_DIM = 32      # latent dimension per step
DRAW_H_DIM = 256     # LSTM hidden size
DRAW_ATTN_N = 12     # attention grid size (N×N Gaussian filters)

# --- Evaluation ---
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"
