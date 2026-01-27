# -*- coding: utf-8 -*-
"""
Simplified XPCS PINN denoiser (100x100 TTC)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
import os, json, datetime
from contextlib import redirect_stdout

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def export_model_summary(model, filepath: str):
    _ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            model.summary()

def extract_architecture_details(model):
    """
    Returns a compact list of layer configs including kernel size/filters/strides.
    Works best for Conv2D / Conv2DTranspose / Dense / Pooling layers etc.
    """
    details = []
    for i, layer in enumerate(model.layers):
        cfg = layer.get_config()
        entry = {
            "index": i,
            "name": layer.name,
            "class_name": layer.__class__.__name__,
        }

        # Common informative keys
        for k in [
            "filters", "kernel_size", "strides", "padding", "activation",
            "units", "pool_size", "rate", "dilation_rate",
            "use_bias", "groups"
        ]:
            if k in cfg:
                entry[k] = cfg[k]

        # input/output shapes if available
        try:
            entry["input_shape"] = layer.input_shape
        except Exception:
            pass
        try:
            entry["output_shape"] = layer.output_shape
        except Exception:
            pass

        details.append(entry)
    return details

def extract_loss_config(loss_fn):
    """
    Extract the weights of your PhysicsInformedLoss object.
    Assumes the loss object has attributes like lambda_recon etc.
    """
    cfg = {}

    # Safe attribute extraction
    for attr in [
        "lambda_recon", "lambda_1tcf", "lambda_symmetry", "lambda_siegert",
        "lambda_causality", "lambda_boundary", "lambda_smoothness",
        "lambda_baseline", "lambda_contrast", "lambda_fft"
    ]:
        if hasattr(loss_fn, attr):
            cfg[attr] = float(getattr(loss_fn, attr))

    # FFT settings (optional)
    for attr in ["fft_use_log", "fft_use_ttc_minus_one", "fft_r_min", "fft_r_max", "fft_eps"]:
        if hasattr(loss_fn, attr):
            val = getattr(loss_fn, attr)
            cfg[attr] = float(val) if isinstance(val, (int, float)) else bool(val) if isinstance(val, bool) else val

    # Which losses are active (weight > 0)
    cfg["active_losses"] = [k for k, v in cfg.items() if k.startswith("lambda_") and v > 0]

    return cfg

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

    def __init__(self, input_shape=(100, 100, 1),pad = 10,  name="ttc_autoencoder"):
        super().__init__(name=name)
        self.input_shape_ = input_shape
        self.padded_size = input_shape[0] + 2 * pad  # 100 + 20 = 120

        # --- padding + cropping ---
        self.zero_pad = layers.ZeroPadding2D(padding=pad, name="pad10")
        self.center_crop = layers.CenterCrop(input_shape[0], input_shape[1], name="crop_back")

        # ---------- Encoder ----------
        self.conv1 = layers.Conv2D(16, (3, 7), activation="relu", padding="same")
        self.conv3 = layers.Conv2D(32, (7, 3), activation="relu", padding="same")
        self.conv4 = layers.Conv2D(32, (5, 5), activation="relu", padding="same")
        self.deconv2 = layers.Conv2DTranspose( 32, (5, 5), activation="relu", padding="same")
        self.deconv3 = layers.Conv2DTranspose( 16, (3, 3), activation="relu", padding="same")
        self.out_conv = layers.Conv2D( 1, (3, 3), activation="linear", padding="same")

    def call(self, inputs, training=False):
        # Encoder
        x = self.zero_pad(inputs)
        x = self.conv1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.out_conv(x)
        x = self.center_crop(x)
        return x

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
                 lambda_contrast = 0.5,
                 lambda_fft=0.05,             
                 fft_use_log=True,
                 fft_use_ttc_minus_one=True,
                 fft_r_min=0.05,
                 fft_r_max=0.95,
                 fft_eps=1e-8):

        self.lambda_recon = lambda_recon
        self.lambda_1tcf = lambda_1tcf
        self.lambda_symmetry = lambda_symmetry
        self.lambda_siegert = lambda_siegert
        self.lambda_causality = lambda_causality
        self.lambda_boundary = lambda_boundary
        self.lambda_smoothness = lambda_smoothness
        self.lambda_baseline = lambda_baseline
        self.lambda_contrast = lambda_contrast
        self.lambda_fft = lambda_fft
        self.fft_use_log = fft_use_log
        self.fft_use_ttc_minus_one = fft_use_ttc_minus_one
        self.fft_r_min = fft_r_min
        self.fft_r_max = fft_r_max
        self.fft_eps = fft_eps

        self._fft_mask = None
        self._fft_mask_T = None

    def _get_fft_mask(self, T: int):
        if self._fft_mask is not None and self._fft_mask_T == T:
            return self._fft_mask
    
        y = tf.linspace(-1.0, 1.0, T)
        x = tf.linspace(-1.0, 1.0, T)
        yy, xx = tf.meshgrid(y, x, indexing="ij")
        rr = tf.sqrt(xx**2 + yy**2) / tf.sqrt(2.0)
    
        mask = tf.cast((rr >= self.fft_r_min) & (rr <= self.fft_r_max), tf.float32)  # [T,T]
        mask = tf.reshape(mask, (1, T, T))  # [1,T,T] float32
    
        self._fft_mask = mask
        self._fft_mask_T = T
        return mask


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

    def symmetry_loss(self, y_pred, band_width = 5):
        y = tf.squeeze(y_pred, axis=-1)        # [B, T, T]
        yT = tf.transpose(y, perm=[0, 2, 1])   # [B, T, T]

        diff = y - yT                          # antisymmetric part

        if band_width is not None:
            # build band mask once per batch shape
            T = tf.shape(y)[1]
            mask_2d = tf.linalg.band_part(
                tf.ones((T, T), dtype=tf.float32),
                band_width, band_width
            )                                  # [T, T]
            mask = tf.expand_dims(mask_2d, 0)  # [1, T, T]
            diff = diff * mask

        return tf.reduce_mean(tf.square(diff))
    
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
    
    def fft_loss(self, y_true, y_pred):
        """
        FFT loss comparing log-magnitude spectra of TTC.
        This version is WARNING-FREE and numerically stable.
        """
    
        yt = tf.squeeze(y_true, axis=-1)  # [B,T,T]
        yp = tf.squeeze(y_pred, axis=-1)
    
        if self.fft_use_ttc_minus_one:
            yt = yt - 1.0
            yp = yp - 1.0
    
        # ---- FFT ----
        Yt = tf.signal.fft2d(tf.cast(yt, tf.complex64))
        Yp = tf.signal.fft2d(tf.cast(yp, tf.complex64))
    
        # ---- EXPLICIT magnitude extraction (real tensor) ----
        Mt = tf.math.real(tf.abs(Yt))
        Mp = tf.math.real(tf.abs(Yp))
    
        # ---- OPTIONAL but RECOMMENDED: stop gradients through FFT ----
        Mt = tf.stop_gradient(Mt)
        Mp = tf.stop_gradient(Mp)
    
        if self.fft_use_log:
            Mt = tf.math.log1p(Mt + self.fft_eps)
            Mp = tf.math.log1p(Mp + self.fft_eps)
    
        T = int(yt.shape[1])
        mask = self._get_fft_mask(T)   # float32 [1,T,T]
    
        Mt = Mt * mask
        Mp = Mp * mask
    
        return tf.reduce_mean(tf.square(Mp - Mt))

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
        loss_fft = self.fft_loss(y_true, y_pred)
        total = (
            self.lambda_recon * loss_recon +
            self.lambda_1tcf * loss_1tcf +
            self.lambda_symmetry * loss_symmetry +
            self.lambda_siegert * loss_siegert +
            self.lambda_causality * loss_causality +
            self.lambda_boundary * loss_boundary +
            self.lambda_smoothness * loss_smooth +
            self.lambda_baseline  * loss_baseline +
            self.lambda_contrast * loss_contrast +
            self.lambda_fft * loss_fft
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
            "fft": loss_fft
        }


# ============================================================
# 3) Trainer
# ============================================================

class XPCSPINNTrainer:
    """
    Training class for XPCS PINN with custom training loop
    """

    def __init__(self, model, loss_fn, optimizer, checkpoint_dir="./checkpoints", run_name = "model"):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
        self.run_name = run_name

        self.train_loss_tracker = keras.metrics.Mean(name="train_loss")
        self.val_loss_tracker = keras.metrics.Mean(name="val_loss")

        self.loss_components_tracker = {
            name: keras.metrics.Mean(name=name)
            for name in ["reconstruction", "1tcf", "symmetry",
                         "siegert", "causality", "boundary", "smoothness", "baseline", "contrast", "fft"]
        }
        self._save_run_config()

    def _save_run_config(self):
        """
        Save a static run config once (model json + architecture details + loss config).
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        run_cfg = {
            "run_name": self.run_name,
            "created_at": datetime.datetime.now().isoformat(),
            "tensorflow_version": tf.__version__,
            "keras_version": keras.__version__ if hasattr(keras, "__version__") else None,
            "loss_config": extract_loss_config(self.loss_fn),
            "architecture_details": extract_architecture_details(self.model),
            "model_json": None,
        }

        # model.to_json() is serializable and useful for recreation
        try:
            run_cfg["model_json"] = self.model.to_json()
        except Exception as e:
            run_cfg["model_json"] = f"FAILED: {str(e)}"
        
        
        
        cfg_path = os.path.join(self.checkpoint_dir, "run_config.json")
        
        print(cfg_path)
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(run_cfg, f, indent=2)

        # Also store a readable summary
        export_model_summary(self.model, os.path.join(self.checkpoint_dir, "model_summary.txt"))


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
    
    
    def _save_checkpoint_bundle(self, epoch: int, datasetname: str = ""):
        """
        Save weights + metadata bundle including model json and loss config.
        """
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{self.run_name}_epoch_{epoch}_{ts}"

        weights_path = os.path.join(self.checkpoint_dir, f"{prefix}.weights.h5")
        model_json_path = os.path.join(self.checkpoint_dir, f"{prefix}.model.json")
        meta_json_path = os.path.join(self.checkpoint_dir, f"{prefix}.meta.json")
        summary_path = os.path.join(self.checkpoint_dir, f"{prefix}.summary.txt")

        # 1) weights
        self.model.save_weights(weights_path)

        # 2) model architecture JSON
        model_json_str = self.model.to_json()
        with open(model_json_path, "w", encoding="utf-8") as f:
            f.write(model_json_str)

        # 3) metadata JSON (loss weights + architecture detail)
        meta = {
            "run_name": self.run_name,
            "saved_at": datetime.datetime.now().isoformat(),
            "epoch": int(epoch),
            "loss_config": extract_loss_config(self.loss_fn),
            "architecture_details": extract_architecture_details(self.model),
            "train_loss": float(self.train_loss_tracker.result().numpy()),
            "val_loss": float(self.val_loss_tracker.result().numpy()),
            "loss_components_epoch": {k: float(v.result().numpy()) for k, v in self.loss_components_tracker.items()},
            "weights_file": os.path.basename(weights_path),
            "model_json_file": os.path.basename(model_json_path),
            "training_dataset_name": datasetname
        }

        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # 4) summary txt (snapshot)
        export_model_summary(self.model, summary_path)

        return weights_path, meta_json_path, model_json_path, summary_path
    
    
    def train(self, train_dataset, val_dataset, epochs, save_freq=10, verbose=1, datasetname=""):
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
    
            train_loss = float(self.train_loss_tracker.result().numpy())
            val_loss = float(self.val_loss_tracker.result().numpy())
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
    
            for k, tracker in self.loss_components_tracker.items():
                history["loss_components"][k].append(float(tracker.result().numpy()))
    
            if verbose:
                print(f"\nEpoch {epoch+1}/{epochs}")
                print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                print("Loss components:")
                for k, tracker in self.loss_components_tracker.items():
                    print(f"  {k}: {tracker.result().numpy():.6f}")
    
            # save bundle
            if (epoch + 1) % save_freq == 0:
                w_path, meta_path, mj_path, sm_path = self._save_checkpoint_bundle(epoch + 1, datasetname=datasetname)
                if verbose >= 1:
                    print(f"Checkpoint bundle saved in: {self.checkpoint_dir}")
                    print(f"  weights: {w_path}")
                    print(f"  meta:    {meta_path}")
                    print(f"  model:   {mj_path}")
                    print(f"  summary: {sm_path}")
    
        return history

    # def train(self, train_dataset, val_dataset, epochs, save_freq=10, verbose=1):
    #     history = {
    #         "train_loss": [],
    #         "val_loss": [],
    #         "loss_components": {k: [] for k in self.loss_components_tracker.keys()},
    #     }

    #     for epoch in range(epochs):
    #         self.train_loss_tracker.reset_state()
    #         self.val_loss_tracker.reset_state()
    #         for tracker in self.loss_components_tracker.values():
    #             tracker.reset_state()

    #         # train
    #         for noisy, target in train_dataset:
    #             self.train_step(noisy, target)

    #         # val
    #         for noisy, target in val_dataset:
    #             self.val_step(noisy, target)

    #         train_loss = self.train_loss_tracker.result().numpy()
    #         val_loss = self.val_loss_tracker.result().numpy()
    #         history["train_loss"].append(train_loss)
    #         history["val_loss"].append(val_loss)

    #         for k, tracker in self.loss_components_tracker.items():
    #             history["loss_components"][k].append(tracker.result().numpy())

    #         if verbose:
    #             print(f"\nEpoch {epoch+1}/{epochs}")
    #             print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    #             print("Loss components:")
    #             for k, tracker in self.loss_components_tracker.items():
    #                 print(f"  {k}: {tracker.result():.4f}")

    #         if (epoch + 1) % save_freq == 0:
    #             os.makedirs(self.checkpoint_dir, exist_ok=True)
    #             ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #             checkpoint_path = os.path.join(
    #                 self.checkpoint_dir,
    #                 f"epoch_{epoch+1}_{ts}.weights.h5"   # <-- important
    #             )
    #             self.model.save_weights(checkpoint_path)
    #             if verbose >= 1:
    #                 print(f"Checkpoint saved: {checkpoint_path}")

    #     return history


# ============================================================
# 4) Preprocessing + dataset
# ============================================================

class XPCSDataPreprocessor:
    """
    Preprocessing utilities for XPCS 2TCF data
    """

    def __init__(self, target_size=100):
        self.target_size = target_size
    def symmetrize_ttc(self, ttcf):
        # ttcf: 2D numpy array [T,T]
        return 0.5 * (ttcf + ttcf.T)
    
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
    
    def add_noise_schaetzel(self,
                        ttcf,
                        n_frames,
                        ns,
                        diag_factor=1.5,
                        tail_factor=0.7,
                        base_scale=0.3):
        """
        Schätzel-like noise with diagonal enhancement,
        but same statistics along the whole diagonal.
    
        ttcf      : pure TTCF (normalized 0..1, baseline 0)
        n_frames  : number of frames used in correlation
        ns        : effective speckle count
        diag_factor : relative noise amplitude at tau=0
        tail_factor : relative noise amplitude at largest tau
        """
    
        ttcf = np.asarray(ttcf, dtype=np.float32)
        N = ttcf.shape[0]
    
        # 1) base Gaussian noise
        noise = np.random.normal(0.0, 1.0, ttcf.shape).astype(np.float32)
    
        # 2) build a scale matrix depending on lag = |i-j|
        idx = np.arange(N, dtype=np.float32)
        tau = np.abs(idx[:, None] - idx[None, :])   # tau[i,j] = |i-j|
        tau_norm = tau / tau.max()                  # in [0,1]
    
        # largest amplitude at tau=0, smallest at tau_max
        scale = diag_factor - (diag_factor - tail_factor) * tau_norm
        # shape: [N, N], constant along each diagonal band
    
        noise *= scale
    
        # 3) Schätzel variance scaling ~ 1/sqrt(M * ns)
        denom = np.sqrt(max(n_frames * ns, 1.0))
        noise = base_scale * noise / denom

        return ttcf + noise

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
            x = self.symmetrize_ttc(x) 
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
    
    def create_training_pairs_v2(self, ttcf_list, noise_level=0.1, num_noise=50, save_data= True, dataset_name= "training_dataset"):
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
                n_frames = int(np.random.randint(100, 500 + 1))

                # ----- 1) Estimate number of speckles (Schätzel parameter) -----
                ns = self.estimate_ns_from_beta(target)
    
                # ----- 2) Add Schätzel noise (diagonal) -----
                noisy = self.add_noise_schaetzel(
                    target, 
                    n_frames=n_frames, 
                    ns=ns,
                    diag_factor= n,
                    tail_factor= n,
                    base_scale=n*0.5
                )
                noisy_list.append(noisy)
                target_list.append(target)
    
        noisy_array = np.array(noisy_list, dtype=np.float32)[..., np.newaxis]
        target_array = np.array(target_list, dtype=np.float32)[..., np.newaxis]
        if save_data:
            np.save(f'./training_datasets/{dataset_name}_pure.npy', target_array)
            np.save(f'./training_datasets/{dataset_name}_noisy.npy', noisy_array)

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
