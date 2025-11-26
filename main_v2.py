# -*- coding: utf-8 -*-
"""
Simple training script for XPCS PINN denoiser on synthetic 2TCF data.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from ttc_ai_v2 import (
    XPCS_PINN_Denoiser,
    PhysicsInformedLoss,
    XPCSPINNTrainer,
    XPCSDataPreprocessor,
    create_dataset,
)


# -------------------------------------------------
# 1) Synthetic XPCS-like 2TCF generator
# -------------------------------------------------
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


def main():
    # ---------------------------------------------
    # Settings
    # ---------------------------------------------
    t_bins = 100       # TTC size: 100 x 100
    n_samples = 600    # number of synthetic TTCFs
    batch_size = 16
    epochs = 10

    # ---------------------------------------------
    # 2) Create synthetic training data
    # ---------------------------------------------
    print("Creating synthetic XPCS data...")
    ttcf_list = create_synthetic_xpcs_data(
        n_samples=n_samples,
        size=t_bins,
        noise_std=0.05
    )

    # ---------------------------------------------
    # 3) Preprocess -> (noisy, clean) pairs
    # ---------------------------------------------
    print("Preprocessing data...")
    preprocessor = XPCSDataPreprocessor(target_size=t_bins)

    noisy_data, target_data = preprocessor.create_training_pairs(
        ttcf_list,
        noise_level=1.60,  # extra noise level for training pairs
    )

    print(f"Training data shape (noisy):  {noisy_data.shape}")
    print(f"Training data shape (target): {target_data.shape}")

    # ---------------------------------------------
    # 4) Train/val split
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
    model = XPCS_PINN_Denoiser(

    )

    # Build by calling once
    dummy_input = tf.random.normal([1, t_bins, t_bins, 1])
    out = model(dummy_input)

    model.summary()
    print("Output shape:", out.shape)
    
    # ---------------------------------------------
    # 7) Loss + optimizer
    # ---------------------------------------------
    loss_fn = PhysicsInformedLoss(
        lambda_recon=1.0,
        lambda_1tcf=0.5,
        lambda_symmetry=0.3,
        lambda_siegert=0.5,
        lambda_causality=0.5,
        lambda_boundary=0.3,
        lambda_smoothness=0.1,
        lambda_baseline=1.0, 
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
                  "siegert", "causality", "boundary"]
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
