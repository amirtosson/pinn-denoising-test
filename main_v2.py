# -*- coding: utf-8 -*-
"""
Simple training script for XPCS PINN denoiser on synthetic 2TCF data.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import sys

from ttc_ai_v2 import (
    XPCS_PINN_Denoiser,
    PhysicsInformedLoss,
    XPCSPINNTrainer,
    XPCSDataPreprocessor,
    create_dataset,
    XPCS_PINN_Denoiser_V2
)


# -------------------------------------------------
# 1) Synthetic XPCS-like 2TCF generator
# -------------------------------------------------

def create_pure_synthetic_xpcs_v2(n_samples=100, size=100):
    """
    Generate PURE (noise-free) synthetic TTCFs for XPCS, normalized 0→1.

    g2_norm(tau) = exp( -2 * (gamma * tau)^alpha )

    Args:
        n_samples : number of TTCFs to generate
        size      : TTCF dimension (size x size)

    Returns:
        pure_list : list of pure TTCFs each [size, size], baseline=0, peak=1
    """

    pure_list = []

    # Time-grid
    t = np.arange(size)
    tt1, tt2 = np.meshgrid(t, t)
    tau = np.abs(tt1 - tt2).astype(np.float32)

    for _ in range(n_samples):

        # ---- Physical parameters ----
        gamma = np.random.uniform(0.01, 0.08)    # decay rate
        alpha = np.random.uniform(0.8, 1.2)      # stretching exponent
        beta  = np.random.uniform(0.2, 0.4)      # speckle contrast

        # ---- Raw g2 ----
        g2 = 1.0 + beta * np.exp(-2.0 * (gamma * tau)**alpha)

        # ---- Normalize to 0→1 ----
        # baseline = 1
        # peak    = 1 + beta
        g2_norm = (g2 - 1.0) / beta          # now baseline=0, peak=1
        g2_norm = g2_norm.astype(np.float32)

        pure_list.append(g2_norm)

    return pure_list




# pure_list = create_pure_synthetic_xpcs_v2(n_samples=10, size=100)

# def plot_pure_ttcfs(pure_list, n_show=3):
#     plt.figure(figsize=(12, 4))

#     for i in range(n_show):
#         plt.subplot(1, n_show, i+1)
#         plt.imshow(pure_list[i], cmap='viridis')
#         plt.title(f"Pure TTCF #{i+1}")
#         plt.colorbar()
#         plt.axis('off')

#     plt.tight_layout()
#     plt.show()
    
# plot_pure_ttcfs(pure_list)




def create_synthetic_xpcs_data(n_samples=100, size=100, noise_std=0.05):
    """
    Create synthetic XPCS 2TCF data with stretched exponential decay.

    Args:
        n_samples: number of TTCF maps to generate
        size: time bins (T x T)
        noise_std: standard deviation of additive Gaussian noise

    Returns:
        list of 2D numpy arrays [size, size]
    """
    data = []
    t = np.arange(size)
    tt1, tt2 = np.meshgrid(t, t)
    tau = np.abs(tt1 - tt2)

    for _ in range(n_samples):
        # random dynamic parameters
        gamma = np.random.uniform(0.01, 0.1)
        alpha = np.random.uniform(0.8, 1.2)
        beta = np.random.uniform(0.2, 0.4)

        # stretched exponential correlation
        corr = 1.0 + beta * np.exp(-2.0 * (gamma * tau) ** alpha)

        # add noise
        #corr += np.random.normal(0.0, noise_std, corr.shape).astype(np.float32)

        data.append(corr.astype(np.float32))

    return data
def interactive_preview(noisy_data, target_data, cmap="viridis"):
    """
    Interactive loop to inspect noisy vs pure TTCF before training.

    Features:
      - Noisy vs pure TTCF images
      - Noise histogram
      - Random row + column intensity profiles
      - Controls: (n)ext, (c)ontinue, (s)top
    """
    print("\n🔍 Interactive TTCF Inspection Mode")
    print("Press:")
    print("  n = next random sample")
    print("  c = continue training")
    print("  s = stop program\n")

    import random

    while True:
        idx = random.randint(0, noisy_data.shape[0] - 1)
        noisy = noisy_data[idx, :, :, 0]
        pure = target_data[idx, :, :, 0]

        # Noise array
        noise = noisy - pure

        # Pick random row + column
        H, W = noisy.shape
        random_row = 50
        random_col = 50

        # ---- FIGURE: TTCF + Histogram + Profiles ----
        fig = plt.figure(figsize=(16, 10))

        # Noisy TTCF
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.set_title(f"Noisy TTCF (idx={idx})")
        ax1.imshow(noisy, cmap=cmap)
        ax1.axis("off")

        # Pure TTCF
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.set_title("Pure Target TTCF")
        ax2.imshow(pure, cmap=cmap)
        ax2.axis("off")

        # Noise Histogram
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.set_title("Noise Distribution")
        ax3.hist(noise.flatten(), bins=50, color="steelblue", alpha=0.8)
        ax3.set_xlabel("Noise Value")
        ax3.set_ylabel("Count")
        ax3.grid(alpha=0.3)

        # Random Row Profile
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.set_title(f"Row Intensity Profile (row={random_row})")
        ax4.plot(noisy[random_row, :], label="Noisy", alpha=0.7)
        ax4.plot(pure[random_row, :], label="Pure", alpha=0.7)
        ax4.legend()
        ax4.grid(alpha=0.3)

        # Random Column Profile
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.set_title(f"Column Intensity Profile (col={random_col})")
        ax5.plot(noisy[:, random_col], label="Noisy", alpha=0.7)
        ax5.plot(pure[:, random_col], label="Pure", alpha=0.7)
        ax5.legend()
        ax5.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

        # ---- USER INPUT ----
        choice = input("Enter n (next), c (continue), s (stop): ").strip().lower()

        if choice == "n":
            continue
        elif choice == "c":
            print("➡️ Continuing to training...\n")
            break
        elif choice == "s":
            print("🛑 Stopping the program.")
            sys.exit(0)
        else:
            print("⚠️ Invalid input. Try again.")
            

def main():
    # ---------------------------------------------
    # Settings
    # ---------------------------------------------
    t_bins = 100       # TTC size: 100 x 100
    n_samples = 500    # number of synthetic TTCFs
    batch_size = 16
    epochs = 5

    # ---------------------------------------------
    # 2) Create synthetic training data
    # ---------------------------------------------n
    
    print("Creating synthetic XPCS data...")
    ttcf_list = create_pure_synthetic_xpcs_v2(
        n_samples=n_samples,
        size=t_bins
    )

    # ---------------------------------------------
    # 3) Preprocess -> (noisy, clean) pairs
    # ---------------------------------------------
    print("Preprocessing data...")
    preprocessor = XPCSDataPreprocessor(target_size=t_bins)

    noisy_data, target_data = preprocessor.create_training_pairs_v2(
        ttcf_list,
        noise_level=0.60,  # extra noise level for training pairs
    )
    
    print(f"Training data shape (noisy):  {noisy_data.shape}")
    print(f"Training data shape (target): {target_data.shape}")
    interactive_preview(noisy_data, target_data)
    # ---------------------------------------------
    # 4) Train/val splitn
    # ---------------------------------------------
    split_idx = int(0.8 * len(noisy_data))
    train_noisy, val_noisy = noisy_data[:split_idx], noisy_data[split_idx:]
    train_target, val_target = target_data[:split_idx], target_data[split_idx:]

    # ---------------------------------------------
    # 5) Build tf.data datasets
    # ---------------------------------------------
    train_dataset = create_dataset(
        train_noisy, train_target,
        batch_size=batch_size,
        shuffle=True
    )
    val_dataset = create_dataset(
        val_noisy, val_target,
        batch_size=batch_size,
        shuffle=False
    )

    # ---------------------------------------------
    # 6) Build model
    # ---------------------------------------------
    print("Building PINN model...")
    model = XPCS_PINN_Denoiser(name='ttc_AE_V1')
    #model = XPCS_PINN_Denoiser_V2(base_filters=32)
    # Build by calling once
    dummy_input = tf.random.normal([1, t_bins, t_bins, 1])
    out = model(dummy_input)

    model.summary()
    print("Output shape:", out.shape)
        
    # ---------------------------------------------
    # 7) Loss + optimizer
    # ---------------------------------------------
    loss_fn = PhysicsInformedLoss(
        lambda_recon=5.0,
        lambda_1tcf=1.0,
        lambda_symmetry=0.3,
        lambda_siegert=0.5,
        lambda_causality=0.5,
        lambda_boundary=0.3,
        lambda_smoothness=0.05,
        lambda_baseline=1.0,
        lambda_contrast= 1.0
    )

    optimizer = keras.optimizers.Adam(learning_rate=1e-4)

    # ---------------------------------------------
    # 8) Trainer
    # ---------------------------------------------
    trainer = XPCSPINNTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        checkpoint_dir="./xpcs_pinn_checkpoints",
    )

    # ---------------------------------------------
    # 9) Train
    # ---------------------------------------------
    print("\nStarting training...")
    history = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=epochs,
        save_freq=5,
        verbose=1,
    )

    # ---------------------------------------------
    # 10) Plot training history
    # ---------------------------------------------
    plt.figure(figsize=(15, 10))

    # total loss
    plt.subplot(3, 3, 1)
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.legend()
    plt.title("Total Loss")
    plt.grid(True)

    # individual components
    components = ["reconstruction", "1tcf", "symmetry",
                 "siegert", "causality", "boundary", "smoothness", "baseline", "contrast"]
    for idx, component in enumerate(components):
        plt.subplot(3, 3, idx + 2)
        plt.plot(history["loss_components"][component])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{component.capitalize()} Loss")
        plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    print("Training history saved to training_history.png")

    # ---------------------------------------------
    # 11) Test on a few validation samples
    # ---------------------------------------------
    print("\nTesting on validation data...")
    test_noisy = val_noisy[:10]
    test_target = val_target[:10]

    denoised = model(test_noisy, training=False).numpy()

    # visualize maps
    fig, axes = plt.subplots(10, 3, figsize=(12, 20))

    for i in range(10):
        axes[i, 0].imshow(test_noisy[i, :, :, 0], cmap="viridis")
        axes[i, 0].set_title("Noisy Input")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(denoised[i, :, :, 0], cmap="viridis")
        axes[i, 1].set_title("Denoised (PINN)")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(test_target[i, :, :, 0], cmap="viridis")
        axes[i, 2].set_title("Target")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig("denoising_results.png", dpi=150)
    print("Denoising results saved to denoising_results.png")

    # ---------------------------------------------
    # 12) Compare 1TCF for a few samples
    # ---------------------------------------------
    print("\nComparing 1-time correlation functions...")
    plt.figure(figsize=(12, 4))

    for i in range(3):
        plt.subplot(1, 3, i + 1)

        tcf_noisy = loss_fn.extract_1tcf(test_noisy[i:i+1]).numpy()[0]
        tcf_denoised = loss_fn.extract_1tcf(denoised[i:i+1, :, :, :]).numpy()[0]
        tcf_target = loss_fn.extract_1tcf(test_target[i:i+1]).numpy()[0]

        lags = np.arange(len(tcf_noisy))

        plt.plot(lags, tcf_noisy, "o-", alpha=0.5, label="Noisy", markersize=2)
        plt.plot(lags, tcf_denoised, "s-", alpha=0.8, label="Denoised", markersize=2)
        plt.plot(lags, tcf_target, "^-", alpha=0.8, label="Target", markersize=2)

        plt.xlabel("Lag (frames)")
        plt.ylabel("g₂(τ)")
        plt.title(f"Sample {i+1}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale("log")

    plt.tight_layout()
    plt.savefig("1tcf_comparison.png", dpi=150)
    print("1TCF comparison saved to 1tcf_comparison.png")

    # ---------------------------------------------
    # 13) Final summary
    # ---------------------------------------------
    print("\n✓ Training complete!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss:   {history['val_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
