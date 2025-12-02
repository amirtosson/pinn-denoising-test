# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 09:36:59 2025

@author: amirt
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from ttc_ai_v2 import (
    XPCS_PINN_Denoiser,
    PhysicsInformedLoss,
    XPCSPINNTrainer,
    XPCSDataPreprocessor,
    create_dataset,
    XPCS_PINN_Denoiser_V2
)


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


class XPCSPINNEvaluator:
    """
    Helper for loading a trained XPCS PINN, testing it on data,
    and plotting TTCF maps + g2(τ) curves.
    """

    def __init__(self, model, loss_fn=None):
        """
        Args:
            model   : an instance of XPCS_PINN_Denoiser (architecture already built)
            loss_fn : instance of PhysicsInformedLoss (for g2 extraction), optional
        """
        self.model = model
        self.loss_fn = loss_fn

    # ---------- weights ----------

    def load_weights(self, weights_path):
        """
        Load model weights from a .weights.h5 file.

        Example:
            evaluator.load_weights("./xpcs_pinn_checkpoints/epoch_50.weights.h5")
        """
        self.model.load_weights(weights_path)
        print(f"✅ Loaded weights from: {weights_path}")

    # ---------- core inference ----------

    def denoise(self, noisy_data, batch_size=16):
        """
        Run the model on a batch of noisy TTCFs.

        Args:
            noisy_data: np.array [N, H, W, 1]
            batch_size: prediction batch size

        Returns:
            denoised: np.array [N, H, W, 1]
        """
        noisy_data = noisy_data.astype("float32")
        denoised = self.model.predict(noisy_data, batch_size=batch_size, verbose=0)
        denoised = self.model.predict(denoised, batch_size=batch_size, verbose=0)
        #denoised = self.model.predict(denoised, batch_size=batch_size, verbose=0)
        #denoised = self.model.predict(denoised, batch_size=batch_size, verbose=0)
        #denoised = self.model.predict(denoised, batch_size=batch_size, verbose=0)

        return denoised

    # ---------- TTCF image plots ----------

    def plot_ttc_examples(
        self,
        noisy_data,
        target_data=None,
        num_examples=5,
        save_path=None,
        cmap="viridis",
        random_seed=None,
    ):
        """
        Plot triplets: Noisy / Denoised / Target for a few random samples.

        Args:
            noisy_data  : [N, H, W, 1]
            target_data : [N, H, W, 1] or None
            num_examples: number of samples to show
            save_path   : if not None, save figure to this path
            cmap        : matplotlib colormap
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        N = noisy_data.shape[0]
        num_examples = min(num_examples, N)
        indices = np.random.choice(N, size=num_examples, replace=False)

        # Run model
        denoised = self.denoise(noisy_data[indices])

        has_target = target_data is not None

        # Determine number of columns
        n_cols = 3 if has_target else 2
        titles = ["Noisy Input", "Denoised (PINN)"] + (["Target"] if has_target else [])

        fig, axes = plt.subplots(
            num_examples, n_cols, figsize=(4 * n_cols, 4 * num_examples)
        )

        if num_examples == 1:
            axes = np.expand_dims(axes, axis=0)  # unify indexing

        for row, idx in enumerate(indices):
            noisy = noisy_data[idx, :, :, 0]
            deno  = denoised[row, :, :, 0]
            if has_target:
                targ = target_data[idx, :, :, 0]

            # Noisy
            ax = axes[row, 0]
            im = ax.imshow(noisy, cmap=cmap)
            ax.set_title(titles[0])
            ax.axis("off")

            # Denoised
            ax = axes[row, 1]
            im = ax.imshow(deno, cmap=cmap)
            ax.set_title(titles[1])
            ax.axis("off")

            # Target
            if has_target:
                ax = axes[row, 2]
                im = ax.imshow(targ, cmap=cmap)
                ax.set_title(titles[2])
                ax.axis("off")

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"💾 TTCF figure saved to: {save_path}")
        plt.show()

    # ---------- g2 plots ----------

    def _extract_g2_numpy(self, ttc):
        """
        Helper: extract g2(τ) from a single TTCF numpy array using PhysicsInformedLoss.

        ttc: [H, W] or [H, W, 1]
        returns: 1D np.array [lags]
        """
        if self.loss_fn is None:
            raise ValueError(
                "loss_fn is None. Provide PhysicsInformedLoss to XPCSPINNEvaluator "
                "to enable g2 extraction."
            )

        if ttc.ndim == 2:
            ttc = ttc[..., np.newaxis]  # H,W,1

        ttc_batch = ttc[np.newaxis, ...]  # 1,H,W,1
        ttc_tf = tf.convert_to_tensor(ttc_batch, dtype=tf.float32)
        g2_tf = self.loss_fn.extract_1tcf(ttc_tf)  # [1, max_lag]
        g2 = g2_tf.numpy()[0]
        return g2

    def plot_g2_examples(
        self,
        noisy_data,
        target_data,
        num_examples=3,
        save_path=None,
        random_seed=None,
    ):
        """
        Plot g2(τ) for Noisy vs Denoised vs Target for a few samples.

        Args:
            noisy_data  : [N, H, W, 1]
            target_data : [N, H, W, 1]
            num_examples: number of samples
            save_path   : optional filepath to save figure
        """
        if self.loss_fn is None:
            raise ValueError(
                "PhysicsInformedLoss instance required for g2 plotting "
                "(pass as loss_fn to XPCSPINNEvaluator)."
            )

        if random_seed is not None:
            np.random.seed(random_seed)

        N = noisy_data.shape[0]
        num_examples = min(num_examples, N)
        indices = np.random.choice(N, size=num_examples, replace=False)

        # Denoise subset
        noisy_subset = noisy_data[indices]
        denoised_subset = self.denoise(noisy_subset)

        fig, axes = plt.subplots(1, num_examples, figsize=(5 * num_examples, 4))

        if num_examples == 1:
            axes = [axes]

        for i, (ax, idx) in enumerate(zip(axes, indices)):
            noisy = noisy_data[idx, :, :, 0]
            targ  = target_data[idx, :, :, 0]
            deno  = denoised_subset[i, :, :, 0]

            g2_noisy    = self._extract_g2_numpy(noisy)
            g2_denoised = self._extract_g2_numpy(deno)
            g2_target   = self._extract_g2_numpy(targ)

            lags = np.arange(len(g2_target))

            ax.plot(lags, g2_noisy,    "o-", ms=3, alpha=0.6, label="Noisy")
            ax.plot(lags, g2_denoised, "s-", ms=3, alpha=0.8, label="Denoised")
            ax.plot(lags, g2_target,   "^-", ms=3, alpha=0.8, label="Target")

            ax.set_xlabel("Lag (frames)")
            ax.set_ylabel("g₂(τ)")
            ax.set_title(f"Sample {idx}")
            ax.grid(alpha=0.3)
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.legend()

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"💾 g2 figure saved to: {save_path}")
        plt.show()




t_bins = 100       # TTC size: 100 x 100
n_samples = 10    # number of synthetic TTCFs
batch_size = 16
epochs = 15

# ---------------------------------------------
# 2) Create synthetic training data
# ---------------------------------------------
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



model = XPCS_PINN_Denoiser()
loss_fn = PhysicsInformedLoss(
     lambda_recon=50.0,
     lambda_1tcf=0,
     lambda_symmetry=0.0,
     lambda_siegert=0.0,
     lambda_causality=0.0,
     lambda_boundary=0.0,
     lambda_smoothness=0.0,
     lambda_baseline=0.0,
     lambda_contrast= 0.0
  )

evaluator = XPCSPINNEvaluator(model=model, loss_fn=loss_fn)
dummy_input = tf.random.normal([1, t_bins, t_bins, 1])
out = model(dummy_input)

model.summary()


weights_path = "./xpcs_pinn_checkpoints/epoch_5_20251202_111714.weights.h5"
evaluator.load_weights(weights_path)

# 5. TTCF triplet plots
evaluator.plot_ttc_examples(
    noisy_data=noisy_data,
    target_data=target_data,
    num_examples=6,
    save_path="ttcf_examples.png",
)

# 6. g2(τ) comparison plots
evaluator.plot_g2_examples(
    noisy_data=noisy_data,
    target_data=target_data,
    num_examples=3,
    save_path="g2_examples.png",
)


