# -*- coding: utf-8 -*-
"""
Simplified XPCS PINN denoiser (100x100 TTC)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
import os 
import datetime


# ============================================================
# 1) Model: simple 2-level Conv + ConvTranspose autoencoder
# ============================================================

class XPCS_PINN_Denoiser_V2(Model):
    """
    Residual U-Net style CNN for XPCS 2TCF denoising (100x100x1).

    - Encoder with residual blocks and max-pooling
    - Decoder with upsampling + skip connections
    - Bottleneck at ~13x13 spatial resolution
    """

    def __init__(self,
                 base_filters=32,
                 name="xpcs_pinn_resunet_100x100"):
        super().__init__(name=name)
        self.base_filters = base_filters

        # ---- encoder blocks ----
        self.enc1 = self._make_res_block(64,  name="enc1")
        self.pool1 = layers.MaxPooling2D(pool_size=2, padding="same")  # 100 -> 50

        #self.enc2 = self._make_res_block(32, name="enc2")
        #self.pool2 = layers.MaxPooling2D(pool_size=2, padding="same")  # 50 -> 25

        self.enc3 = self._make_res_block(16, name="enc3")
        self.pool3 = layers.MaxPooling2D(pool_size=2, padding="same")  # 25 -> 13

        # ---- bottleneck ----
        self.bottleneck = self._make_res_block(16, name="bottleneck")

        # for latent representation (optional)
        self.global_pool = layers.GlobalAveragePooling2D(name="latent_pool")

        # ---- decoder blocks ----
        self.up3 = layers.UpSampling2D(size=2)  # 13 -> 26 (will crop to 25)
        self.crop3 = layers.CenterCrop(25, 25)
        self.dec3 = self._make_res_block(32, name="dec3")

        self.up2 = layers.UpSampling2D(size=2)  # 25 -> 50
        self.dec2 = self._make_res_block(32, name="dec2")

        self.up1 = layers.UpSampling2D(size=2)  # 50 -> 100
        self.dec1 = self._make_res_block(base_filters, name="dec1")

        # final 1-channel output
        self.out_conv = layers.Conv2D(
            1, (3, 3), padding="same", activation="linear", name="output_conv"
        )

    # -------- helper: residual block --------
    def _make_res_block(self, filters, name):
        """
        Build a small residual block:
        x -> Conv(3x3) -> BN -> ReLU -> Conv(3x3) -> BN + skip -> ReLU
        """
        conv1 = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=name+"_conv1")
        bn1   = layers.BatchNormalization(name=name+"_bn1")
        act1  = layers.ReLU(name=name+"_relu1")

        conv2 = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=name+"_conv2")
        bn2   = layers.BatchNormalization(name=name+"_bn2")

        # 1x1 conv for channel-matching on the skip path (if needed)
        proj  = layers.Conv2D(filters, 1, padding="same", use_bias=False, name=name+"_proj")
        proj_bn = layers.BatchNormalization(name=name+"_proj_bn")
        act_out = layers.ReLU(name=name+"_relu_out")

        def block(x):
            shortcut = x

            y = conv1(x)
            y = bn1(y)
            y = act1(y)

            y = conv2(y)
            y = bn2(y)

            # project shortcut if channels mismatch
            if shortcut.shape[-1] != filters:
                shortcut = proj(shortcut)
                shortcut = proj_bn(shortcut)

            y = layers.Add()([y, shortcut])
            y = act_out(y)
            return y

        return block

    # -------- forward pass --------
    def call(self, inputs, training=False):
        # Encoder
        x1 = self.enc1(inputs)           # 100x100, F
        p1 = self.pool1(x1)              # 50x50

        x2 = self.enc2(p1)               # 50x50, 2F
        p2 = self.pool2(x2)              # 25x25

        x3 = self.enc3(p2)               # 25x25, 4F
        p3 = self.pool3(x3)              # 13x13

        # Bottleneck
        b = self.bottleneck(p3)          # 13x13, 8F

        # Decoder
        u3 = self.up3(b)                 # 26x26
        u3 = self.crop3(u3)              # 25x25 to match x3
        u3 = layers.Concatenate()([u3, x3])
        d3 = self.dec3(u3)               # 25x25, 4F

        u2 = self.up2(d3)                # 50x50
        u2 = layers.Concatenate()([u2, x2])
        d2 = self.dec2(u2)               # 50x50, 2F

        u1 = self.up1(d2)                # 100x100
        u1 = layers.Concatenate()([u1, x1])
        d1 = self.dec1(u1)               # 100x100, F

        out = self.out_conv(d1)          # 100x100x1

        return out

    # optional latent vector extractor for analysis
    def get_latent_representation(self, inputs):
        # run encoder + bottleneck then global pool
        x1 = self.enc1(inputs)
        p1 = self.pool1(x1)

        x2 = self.enc2(p1)
        p2 = self.pool2(x2)

        x3 = self.enc3(p2)
        p3 = self.pool3(x3)

        b  = self.bottleneck(p3)
        latent = self.global_pool(b)
        return latent


class XPCS_PINN_Denoiser(Model):
    """
    Physics-Informed CNN Autoencoder for XPCS Two-Time Correlation Denoising

    - Fixed input: 100x100x1 TTC maps
    - Encoder: 2 downsampling blocks
    - Decoder: 2 Conv2DTranspose blocks, output 100x100x1
    """

    def __init__(self, input_shape=(100, 100, 1), name="ttc_autoencoder"):
        super().__init__(name=name)
        self.input_shape_ = input_shape

        # ---------- Encoder ----------
        self.conv1 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")
        #self.pool1 = layers.MaxPooling2D((2, 2), padding="same")   # 100 -> 50

        #self.conv2 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")
        #self.pool2 = layers.MaxPooling2D((2, 2), padding="same")   # 50 -> 25

        self.conv3 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")
        #self.pool3 = layers.MaxPooling2D((2, 2), strides=1, padding="same")  # 25 -> 25

        self.bottleneck_conv = layers.Conv2D(
            64, (3, 3), activation="relu", padding="same"
        )  # 25x25x512

        # ---------- Decoder ----------
        # 25 -> 50
        
        self.deconv1 = layers.Conv2DTranspose(
            64, (3, 3), activation="relu", padding="same"
        )
        # 50 -> 100
        self.deconv2 = layers.Conv2DTranspose(
            32, (3, 3), activation="relu", padding="same"
        )
        # 100 -> 100
        self.deconv3 = layers.Conv2DTranspose(
            16, (3, 3), activation="relu", padding="same"
        )

        self.out_conv = layers.Conv2D(
            1, (3, 3), activation="linear", padding="same"
        )

    def call(self, inputs, training=False):
        # Encoder
        x = self.conv1(inputs)
        #x = self.pool1(x)

        #x = self.conv2(x)
        #x = self.pool2(x)

        x = self.conv3(x)
        #x = self.pool3(x)

        x = self.bottleneck_conv(x)

        # Decoder
        #x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)

        outputs = self.out_conv(x)
        return outputs

    def build_graph(self):
        # helper to print summary
        x = layers.Input(shape=self.input_shape_)
        return Model(inputs=x, outputs=self.call(x))

# ============================================================
# 2) Physics-informed loss
# ============================================================

class PhysicsInformedLoss:
    """
    Physics-informed loss functions for XPCS two-time correlation denoising
    """

    def __init__(self,
                 lambda_recon=1.0,
                 lambda_1tcf=1.0,
                 lambda_symmetry=0.5,
                 lambda_siegert=0.3,
                 lambda_causality=0.2,
                 lambda_boundary=0.3,
                 lambda_smoothness=0.1, 
                 lambda_baseline=0.5, 
                 lambda_contrast = 0.5):

        self.lambda_recon = lambda_recon
        self.lambda_1tcf = lambda_1tcf
        self.lambda_symmetry = lambda_symmetry
        self.lambda_siegert = lambda_siegert
        self.lambda_causality = lambda_causality
        self.lambda_boundary = lambda_boundary
        self.lambda_smoothness = lambda_smoothness
        self.lambda_baseline = lambda_baseline
        self.lambda_contrast = lambda_contrast

    # ---------- basic losses ----------

    def reconstruction_loss(self, y_true, y_pred):
        y_true_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
        y_pred_flat = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    
        # subtract means
        y_true_mean = tf.reduce_mean(y_true_flat, axis=1, keepdims=True)
        y_pred_mean = tf.reduce_mean(y_pred_flat, axis=1, keepdims=True)
    
        y_true_centered = y_true_flat - y_true_mean
        y_pred_centered = y_pred_flat - y_pred_mean
    
        # numerator: sum((x - mx)*(y - my))
        numerator = tf.reduce_sum(y_true_centered * y_pred_centered, axis=1)
    
        # denominator: std_x * std_y
        denom = tf.sqrt(
            tf.reduce_sum(tf.square(y_true_centered), axis=1)
            * tf.reduce_sum(tf.square(y_pred_centered), axis=1)
            + 1e-8
        )
    
        ncc = numerator / denom
    
        # loss = 1 - NCC
        return tf.reduce_mean(1.0 - ncc)
    def contrast_loss(self, y_true, y_pred, tail_fraction=0.2):
        t_true = self.extract_1tcf(y_true)
        t_pred = self.extract_1tcf(y_pred)
    
        T = tf.shape(t_true)[1]
        k = tf.cast(tf.round(tf.cast(T, tf.float32)*(1.0-tail_fraction)), tf.int32)
    
        # baseline ~ mean of tail
        base_true = tf.reduce_mean(t_true[:, k:], axis=1, keepdims=True)
        base_pred = tf.reduce_mean(t_pred[:, k:], axis=1, keepdims=True)
    
        # contrast = g2(0) - baseline
        c_true = t_true[:, :1] - base_true
        c_pred = t_pred[:, :1] - base_pred
    
        return tf.reduce_mean(tf.square(c_true - c_pred))


    def extract_1tcf(self, ttcf_2d):
        """
        Extract 1-time correlation function from 2TCF by averaging diagonals.
        ttcf_2d: [batch, H, W, 1]
        """
        ttcf_2d = tf.squeeze(ttcf_2d, axis=-1)  # [batch, H, W]
        size = tf.shape(ttcf_2d)[1]
        max_lag = size

        ones_tcf = []
        for lag in range(max_lag):
            if lag == 0:
                diag = tf.linalg.diag_part(ttcf_2d)
            else:
                upper = tf.linalg.diag_part(ttcf_2d, k=lag)
                lower = tf.linalg.diag_part(ttcf_2d, k=-lag)
                diag = (upper + lower) / 2.0
            ones_tcf.append(tf.reduce_mean(diag, axis=-1))

        ones_tcf = tf.stack(ones_tcf, axis=-1)  # [batch, max_lag]
        return ones_tcf

    def one_time_correlation_loss(self, y_true, y_pred):
        tcf_true = self.extract_1tcf(y_true)
        tcf_pred = self.extract_1tcf(y_pred)
        return tf.reduce_mean(tf.square(tcf_true[:, 1:] - tcf_pred[:, 1:]))

    def symmetry_loss(self, y_pred):
        y = tf.squeeze(y_pred, axis=-1)
        yT = tf.transpose(y, perm=[0, 2, 1])
        return tf.reduce_mean(tf.square(y - yT))

    def siegert_relation_loss(self, y_pred, beta=0.3):
        y = tf.squeeze(y_pred, axis=-1)  # [batch, H, W]

        baseline = tf.reduce_mean(y[:, -10:, -10:], axis=[1, 2])
        baseline = tf.reshape(baseline, (-1, 1, 1))

        max_c = tf.reduce_max(y, axis=[1, 2], keepdims=True)
        contrast_violation = tf.maximum(0.0, max_c - baseline[:, None, :] - beta - 0.1)

        below_baseline = tf.maximum(0.0, baseline[:, None, :] - y)

        return tf.reduce_mean(contrast_violation) + tf.reduce_mean(below_baseline)

    def causality_loss(self, y_pred):
        y = tf.squeeze(y_pred, -1)
        diff = y[:,:,1:] - y[:,:,:-1]
        pos = tf.nn.relu(diff)
        return tf.reduce_mean(pos)

    def boundary_condition_loss(self, y_pred):
        y = tf.squeeze(y_pred, axis=-1)
        corner_size = 20
        bottom_right = y[:, -corner_size:, -corner_size:]
        return tf.reduce_mean(tf.square(bottom_right - 1.0))

    def smoothness_loss(self, y_pred):
        dy = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]
        dx = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
        return tf.reduce_mean(tf.square(dy)) + tf.reduce_mean(tf.square(dx))

    def baseline_one_loss(self, y_pred):
        # 1-TCF from predicted 2TCF
        g2 = self.extract_1tcf(y_pred)
        tail = g2[:, -10:]
        return tf.reduce_mean(tf.square(tail))
    # ---------- total ----------

    def total_loss(self, y_true, y_pred):
        loss_recon = self.reconstruction_loss(y_true, y_pred)
        loss_1tcf = self.one_time_correlation_loss(y_true, y_pred)
        loss_symmetry = self.symmetry_loss(y_pred)
        loss_siegert = self.siegert_relation_loss(y_pred)
        loss_causality = self.causality_loss(y_pred)
        loss_boundary = self.boundary_condition_loss(y_pred)
        loss_smooth = self.smoothness_loss(y_pred)
        loss_baseline  = self.baseline_one_loss(y_pred)
        loss_contrast = self.contrast_loss(y_true, y_pred)
        total = (
            self.lambda_recon * loss_recon +
            self.lambda_1tcf * loss_1tcf +
            self.lambda_symmetry * loss_symmetry +
            self.lambda_siegert * loss_siegert +
            self.lambda_causality * loss_causality +
            self.lambda_boundary * loss_boundary +
            self.lambda_smoothness * loss_smooth +
            self.lambda_baseline  * loss_baseline +
            self.lambda_contrast * loss_contrast
        )

        return total, {
            "reconstruction": loss_recon,
            "1tcf": loss_1tcf,
            "symmetry": loss_symmetry,
            "siegert": loss_siegert,
            "causality": loss_causality,
            "boundary": loss_boundary,
            "smoothness": loss_smooth,
            "baseline": loss_baseline, 
            "contrast": loss_contrast,
        }


# ============================================================
# 3) Trainer
# ============================================================

class XPCSPINNTrainer:
    """
    Training class for XPCS PINN with custom training loop
    """

    def __init__(self, model, loss_fn, optimizer, checkpoint_dir="./checkpoints"):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir

        self.train_loss_tracker = keras.metrics.Mean(name="train_loss")
        self.val_loss_tracker = keras.metrics.Mean(name="val_loss")

        self.loss_components_tracker = {
            name: keras.metrics.Mean(name=name)
            for name in ["reconstruction", "1tcf", "symmetry",
                         "siegert", "causality", "boundary", "smoothness", "baseline", "contrast"]
        }

    def train_step(self, noisy_2tcf, target_2tcf):
        with tf.GradientTape() as tape:
            denoised = self.model(noisy_2tcf, training=True)
            total_loss, loss_comp = self.loss_fn.total_loss(target_2tcf, denoised)

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.train_loss_tracker.update_state(total_loss)
        for k, v in loss_comp.items():
            self.loss_components_tracker[k].update_state(v)

        return total_loss

 
    def val_step(self, noisy_2tcf, target_2tcf):
        denoised = self.model(noisy_2tcf, training=False)
        total_loss, _ = self.loss_fn.total_loss(target_2tcf, denoised)
        self.val_loss_tracker.update_state(total_loss)
        return total_loss

    def train(self, train_dataset, val_dataset, epochs, save_freq=10, verbose=1):
        history = {
            "train_loss": [],
            "val_loss": [],
            "loss_components": {k: [] for k in self.loss_components_tracker.keys()},
        }

        for epoch in range(epochs):
            self.train_loss_tracker.reset_state()
            self.val_loss_tracker.reset_state()
            for tracker in self.loss_components_tracker.values():
                tracker.reset_state()

            # train
            for noisy, target in train_dataset:
                self.train_step(noisy, target)

            # val
            for noisy, target in val_dataset:
                self.val_step(noisy, target)

            train_loss = self.train_loss_tracker.result().numpy()
            val_loss = self.val_loss_tracker.result().numpy()
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            for k, tracker in self.loss_components_tracker.items():
                history["loss_components"][k].append(tracker.result().numpy())

            if verbose:
                print(f"\nEpoch {epoch+1}/{epochs}")
                print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                print("Loss components:")
                for k, tracker in self.loss_components_tracker.items():
                    print(f"  {k}: {tracker.result():.4f}")

            if (epoch + 1) % save_freq == 0:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f"epoch_{epoch+1}_{ts}.weights.h5"   # <-- important
                )
                self.model.save_weights(checkpoint_path)
                if verbose >= 1:
                    print(f"Checkpoint saved: {checkpoint_path}")

        return history


# ============================================================
# 4) Preprocessing + dataset
# ============================================================

class XPCSDataPreprocessor:
    """
    Preprocessing utilities for XPCS 2TCF data
    """

    def __init__(self, target_size=100):
        self.target_size = target_size

    def normalize_2tcf(self, ttcf, contrast=None):
        ttcf = np.asarray(ttcf, dtype=np.float32)
        if contrast is None:
            lag1 = np.diag(ttcf, k=1)
            contrast = np.mean(lag1) - 1.0
            contrast = max(contrast, 0.01)
        normalized = (ttcf - 1.0) / contrast + 1.0
        return normalized

    def resize_2tcf(self, ttcf, target_size=None):
        if target_size is None:
            target_size = self.target_size

        ttcf_tf = tf.convert_to_tensor(ttcf, dtype=tf.float32)[..., tf.newaxis]
        resized = tf.image.resize(ttcf_tf, [target_size, target_size],
                                  method="bilinear", preserve_aspect_ratio=False)
        return resized.numpy().squeeze().astype(np.float32)

    def replace_diagonal(self, ttcf):
        ttcf = ttcf.copy()
        lag1_mean = np.mean([np.diag(ttcf, k=1), np.diag(ttcf, k=-1)])
        np.fill_diagonal(ttcf, lag1_mean)
        return ttcf

    def add_noise(self, ttcf, noise_level=0.1,diag_boost = 1):
        """
        Add symmetric Gaussian noise to TTCF, mirrored around the diagonal.
        
        Args:
            ttcf       : 2D TTCF array [H, W]
            noise_level: base std of noise relative to std(ttcf)
            diag_boost : additional noise factor near diagonal (0.0 = off)
        
        Returns:
            noisy TTCF (float32, [H, W])
        """
        ttcf = np.asarray(ttcf, dtype=np.float32)
        H, W = ttcf.shape
        assert H == W, "TTCF must be square"
        
        # Base random noise field
        base_noise = np.random.normal(loc=0.0, scale=noise_level, size=(H, W)).astype(
            np.float32
        )
        
        # Make it symmetric: N[i,j] = N[j,i]
        sym_noise = 0.5 * (base_noise + base_noise.T)
        
        # Optional: boost noise near diagonal (more realistic XPCS)
        if diag_boost > 0.0:
            idx = np.arange(H)
            # distance from diagonal |i-j|
            dist = np.abs(idx[:, None] - idx[None, :]).astype(np.float32)
            # decaying mask from 1 on the diagonal to ~0 away from it
            diag_mask = np.exp(-dist / (H * 0.1)).astype(np.float32)
            sym_noise *= (1.0 + diag_boost * diag_mask)
        
        # Scale by TTCF std
        scale = float(np.std(ttcf)) if np.std(ttcf) > 0 else 1.0
        sym_noise *= scale
        
        return ttcf + sym_noise
    
    def add_noise_schaetzel(self, ttcf, n_frames, n_speckles=1, scale=5.0):
        """
        Add noise to a 2TCF using a Schätzel-like analysis.

        Args:
            ttcf       : clean 2TCF array [T, T] (g2 or C2)
            n_frames   : number of time frames used to build TTCF
            n_speckles : effective number of independent speckles / q-pixels
            scale      : extra scaling factor to tune noise level

        Returns:
            noisy 2TCF with variance ~ 1 / sqrt( n_speckles * (N_frames - |Δt|) )
        """
        scale = np.random.randint(5, 60)
        ttcf = np.asarray(ttcf, dtype=np.float32)
        T = ttcf.shape[0]

        # build lag matrix τ = |t1 - t2|
        t = np.arange(T, dtype=np.int32)
        tt1, tt2 = np.meshgrid(t, t)
        tau = np.abs(tt1 - tt2)  # [T, T]

        # number of contributing pairs for each lag
        # M_pairs(τ) = max(N_frames - τ, 1)
        M_pairs = np.maximum(n_frames - tau, 1)

        # total independent samples (pairs * speckles)
        M_eff = np.maximum(M_pairs * n_speckles, 1)

        # Schätzel-style σ ~ 1 / sqrt(M_eff)
        sigma = scale / np.sqrt(M_eff).astype(np.float32)

        # sample Gaussian noise with local σ
        noise = np.random.normal(loc=0.0, scale=1.0, size=ttcf.shape).astype(np.float32)
        noisy = ttcf + sigma * noise

        return noisy

    def estimate_ns_from_beta(self, ttcf):
        # beta ≈ std / mean near diagonal
        diag = np.diag(ttcf)
        beta = np.std(diag) / np.mean(diag)
        beta = np.clip(beta, 0.05, 0.5)
        ns = int((1.0 / beta) ** 2)
        return ns

    def create_training_pairs(self, ttcf_list, noise_level=0.1, num_noise= 10):
        """
        Very simple:
        - For each clean TTCF:
            normalize -> replace diag -> resize to 100x100
            target = this map
            input  = target + noise
        """
        noisy_list = []
        target_list = []

        for ttcf in ttcf_list:
            x = self.normalize_2tcf(ttcf)
            x = self.replace_diagonal(x)
            x = self.resize_2tcf(x, self.target_size)

            target = x.astype(np.float32)
            for n in range(num_noise):
                ns = self.estimate_ns_from_beta(target)
                noisy = self.add_noise_schaetzel(target, 100, ns)
                noisy = self.add_noise(target, noise_level=(noise_level +(n*0.5)))
    
                noisy_list.append(noisy)
                target_list.append(target)

        noisy_array = np.array(noisy_list, dtype=np.float32)[..., np.newaxis]
        target_array = np.array(target_list, dtype=np.float32)[..., np.newaxis]
        return noisy_array, target_array
    def create_training_pairs_v2(self, ttcf_list, noise_level=0.1, num_noise=10):
        """
        Generate (noisy, target) training pairs from PURE normalized TTCFs.
        
        Steps:
          1. Input pure TTCF is already normalized (baseline=0, peak=1)
          2. Apply Schätzel noise (diagonal)
          3. Apply Gaussian & symmetric noise
          4. Re-normalize noisy TTCF back to 0..1
          5. Return (noisy, target)
        """
    
        noisy_list = []
        target_list = []
    
        for ttcf in ttcf_list:
    
            # pure TTCF already normalized 0..1
            target = ttcf.astype(np.float32)
    
            for n in range(num_noise):
    
                # ----- 1) Estimate number of speckles (Schätzel parameter) -----
                ns = self.estimate_ns_from_beta(target)
    
                # ----- 2) Add Schätzel noise (diagonal) -----
                noisy = self.add_noise_schaetzel(
                    target, 
                    n_frames=100, 
                    n_speckles=ns,
                    scale= 5*n
                )
    
                noisy = self.add_noise(noisy, noise_level=(noise_level +(n*3.5)))

    
                # -------------------------------
                # 🔥 5) Re-normalize noisy TTCF back to 0..1
                # -------------------------------
                # new_peak   ~ max value near diagonal
                # new_base   ~ average of tail region
                # ensures noisy is ALWAYS in [0..1]
                # -------------------------------
                peak = np.percentile(noisy[:5, :5], 95)       # near-diagonal
                base = np.mean(noisy[-10:, -10:])             # long-time tail
    
                noisy = (noisy - base) / max(peak - base, 1e-6)
    
                # Clip to valid range
                noisy = np.clip(noisy, 0.0, 1.0).astype(np.float32)
    
                # Store
                noisy_list.append(noisy)
                target_list.append(target)
    
        noisy_array = np.array(noisy_list, dtype=np.float32)[..., np.newaxis]
        target_array = np.array(target_list, dtype=np.float32)[..., np.newaxis]
    
        return noisy_array, target_array

def create_dataset(noisy_data, target_data, batch_size=8, shuffle=True):
    noisy_data = noisy_data.astype("float32")
    target_data = target_data.astype("float32")

    ds = tf.data.Dataset.from_tensor_slices((noisy_data, target_data))
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
