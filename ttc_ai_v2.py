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

        self.conv2 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")
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

        x = self.conv2(x)
        #x = self.pool2(x)

        #x = self.conv3(x)
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
                 lambda_baseline=0.5):

        self.lambda_recon = lambda_recon
        self.lambda_1tcf = lambda_1tcf
        self.lambda_symmetry = lambda_symmetry
        self.lambda_siegert = lambda_siegert
        self.lambda_causality = lambda_causality
        self.lambda_boundary = lambda_boundary
        self.lambda_smoothness = lambda_smoothness
        self.lambda_baseline = lambda_baseline

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
        y = tf.squeeze(y_pred, axis=-1)  # [batch, H, W]
        diff = y[:, :, 1:] - y[:, :, :-1]
        pos_diff = tf.maximum(0.0, diff)

        h = tf.shape(y)[1]
        w = tf.shape(y)[2] - 1
        mask_size = tf.minimum(50, h)
        mask = tf.linalg.band_part(tf.ones((h, w)), mask_size, mask_size)
        mask = tf.expand_dims(mask, 0)

        pos_diff = pos_diff * mask
        return tf.reduce_mean(pos_diff)

    def boundary_condition_loss(self, y_pred):
        y = tf.squeeze(y_pred, axis=-1)
        corner_size = 20
        bottom_right = y[:, -corner_size:, -corner_size:]
        return tf.reduce_mean(tf.square(bottom_right - 1.0))

    def smoothness_loss(self, y_pred):
        dy = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]
        dx = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
        return tf.reduce_mean(tf.square(dy)) + tf.reduce_mean(tf.square(dx))

    def baseline_one_loss(self, y_pred, tail_fraction=0.2):
    
            # 1-TCF from predicted 2TCF
            tcf_pred = self.extract_1tcf(y_pred)      # [B, T]
    
            T = tf.shape(tcf_pred)[1]
            k = tf.cast(tf.round(tf.cast(T, tf.float32) * (1.0 - tail_fraction)), tf.int32)
    
            tail = tcf_pred[:, k:]                   # last 20% of lags
            mean_tail = tf.reduce_mean(tail)         # scalar
    
            return tf.square(mean_tail - 1.0)
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

        total = (
            self.lambda_recon * loss_recon +
            self.lambda_1tcf * loss_1tcf +
            self.lambda_symmetry * loss_symmetry +
            self.lambda_siegert * loss_siegert +
            self.lambda_causality * loss_causality +
            self.lambda_boundary * loss_boundary +
            self.lambda_smoothness * loss_smooth +
            self.lambda_baseline  * loss_baseline 
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
                         "siegert", "causality", "boundary", "smoothness", "baseline"]
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

    def add_noise(self, ttcf, noise_level=0.1):
        ttcf = np.asarray(ttcf, dtype=np.float32)
        noise = np.random.normal(0, noise_level, ttcf.shape).astype(np.float32)
        return ttcf + noise * np.std(ttcf)

    def create_training_pairs(self, ttcf_list, noise_level=0.1, num_noise= 15):
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
                noisy = self.add_noise(target, noise_level=(noise_level +(n*0.2)))
    
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
