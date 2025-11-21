# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 11:59:50 2025

@author: amirt
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 12:02:16 2025

@author: amirt
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime


class TTCPhysicsAutoencoder:
    """
    Autoencoder for TTC denoising with optional physics-informed loss:
      physics_type:
        - 'none'      : purely data-driven
        - 'diffusion' : diffusion-like PDE in (τ, q)
        - 'siegert'   : discrete Siegert + kinetic model (g2)

    Assumes input shape [T, Q, 1], where:
      T ~ delay time τ index
      Q ~ q index
    """

    # ---------- Physics loss inner classes ----------

    class DiffusionLikePhysicsLoss:
        """
        ∂C/∂τ = D ∂²C/∂q² - λ C
        Discretized with central differences in τ and q.

        y_pred shape: [batch, T, Q, 1]
        """
        def __init__(self,
                     D=1.0,
                     lambda_relax=0.0,
                     delta_tau=1.0,
                     delta_q=1.0,
                     weight=1.0):
            self.D = float(D)
            self.lambda_relax = float(lambda_relax)
            self.delta_tau = float(delta_tau)
            self.delta_q = float(delta_q)
            self.weight = float(weight)

        def __call__(self, y_true, y_pred):
            # y_true not used, we enforce physics on y_pred
            C = y_pred  # [B, T, Q, 1]

            # interior region: T-2, Q-2
            # dC/dτ (central diff over τ, axis=1)
            C_tau_minus = C[:, :-2, 1:-1, :]
            C_tau_plus  = C[:,  2:, 1:-1, :]
            dC_dTau = (C_tau_plus - C_tau_minus) / (2.0 * self.delta_tau)

            # d²C/dq² (second diff over q, axis=2)
            C_q_minus = C[:, 1:-1, :-2, :]
            C_q_mid   = C[:, 1:-1,  1:-1, :]
            C_q_plus  = C[:, 1:-1,  2:, :]
            d2C_dq2 = (C_q_minus - 2.0 * C_q_mid + C_q_plus) / (self.delta_q ** 2)

            C_center = C[:, 1:-1, 1:-1, :]

            residual = dC_dTau - self.D * d2C_dq2 + self.lambda_relax * C_center
            physics_loss = tf.reduce_mean(tf.square(residual))
            return self.weight * physics_loss

    class SiegertPhysicsLoss:
        """
        Physics loss based on Siegert relation and a simple kinetic model:

          g1(q, τ) = exp( - [Γ(q) τ]^α ),   Γ(q) = D q^2
          g2(q, τ) = 1 + β |g1(q, τ)|^2

        y_pred ~ g2 on grid [T x Q]:
          y_pred shape: [batch, T, Q, 1]

        tau_values: 1D array, length T
        q_values:   1D array, length Q
        """
        def __init__(self,
                     tau_values,
                     q_values,
                     D=1.0,
                     beta=1.0,
                     alpha=1.0,
                     weight=1.0):
            tau_values = np.asarray(tau_values, dtype=np.float32)
            q_values   = np.asarray(q_values, dtype=np.float32)

            # Broadcastable grids: [1,T,1,1] and [1,1,Q,1]
            self.tau_grid = tf.constant(tau_values.reshape(1, -1, 1, 1), dtype=tf.float32)
            self.q_grid   = tf.constant(q_values.reshape(1, 1, -1, 1), dtype=tf.float32)

            self.D = float(D)
            self.beta = float(beta)
            self.alpha = float(alpha)
            self.weight = float(weight)

        def __call__(self, y_true, y_pred):
            # theoretical Γ(q)
            Gamma = self.D * tf.square(self.q_grid)  # [1,1,Q,1]

            # argument = (Γ τ)^α, broadcast to [1,T,Q,1]
            arg = tf.pow(Gamma * self.tau_grid, self.alpha)
            g1 = tf.exp(-arg)

            g2_theory = 1.0 + self.beta * tf.square(g1)
            residual = y_pred - g2_theory
            physics_loss = tf.reduce_mean(tf.square(residual))
            return self.weight * physics_loss

    # ---------- main class ----------

    def __init__(self,
                 input_shape,
                 physics_type='none',
                 physics_config=None):
        """
        input_shape : (T, Q, 1)
        physics_type: 'none' | 'diffusion' | 'siegert'
        physics_config: dict with parameters for physics model:
            For 'diffusion':
                D, lambda_relax, delta_tau, delta_q, weight
            For 'siegert':
                tau_values, q_values, D, beta, alpha, weight
        """
        self.input_shape = input_shape
        self.physics_type = physics_type
        self.physics_config = physics_config or {}
        self.model = self.build_model()
        self._physics_loss_impl = self._build_physics_loss()

    # -------- build CNN autoencoder --------

    def build_model(self, show_model_summary=True):
        print("Building TTC Physics Autoencoder")
        inputs = layers.Input(shape=self.input_shape)

        # Encoder
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)

        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)

        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), strides=1, padding='same')(x)

        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)

        # Decoder
        x = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
        x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
        x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)

        outputs = layers.Conv2D(1, (3, 3), activation='linear', padding='same')(x)

        model = models.Model(inputs, outputs)
        if show_model_summary:
            model.summary()
        return model

    # -------- physics-loss factory --------

    def _build_physics_loss(self):
        if self.physics_type == 'diffusion':
            cfg = self.physics_config
            return TTCPhysicsAutoencoder.DiffusionLikePhysicsLoss(
                D=cfg.get('D', 1.0),
                lambda_relax=cfg.get('lambda_relax', 0.0),
                delta_tau=cfg.get('delta_tau', 1.0),
                delta_q=cfg.get('delta_q', 1.0),
                weight=cfg.get('weight', 1e-3)
            )
        elif self.physics_type == 'siegert':
            cfg = self.physics_config
            T, Q, _ = self.input_shape

            tau_values = cfg.get('tau_values', np.arange(T, dtype=np.float32))
            q_values   = cfg.get('q_values', np.arange(Q, dtype=np.float32))

            return TTCPhysicsAutoencoder.SiegertPhysicsLoss(
                tau_values=tau_values,
                q_values=q_values,
                D=cfg.get('D', 1.0),
                beta=cfg.get('beta', 0.8),
                alpha=cfg.get('alpha', 1.0),
                weight=cfg.get('weight', 1e-3)
            )
        else:
            return None  # no physics

    # -------- combined loss for Keras --------

    def get_combined_loss(self):
        physics_impl = self._physics_loss_impl

        def combined_loss(y_true, y_pred):
            # TV regularization
            dx = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]
            dy = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
            tv_loss = tf.reduce_sum(tf.abs(dx)) + tf.reduce_sum(tf.abs(dy))

            # SSIM + MSE
            ssim_l = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
            mse_l = tf.reduce_mean(tf.square(y_true - y_pred))
            data_loss = 0.5 * mse_l + 0.5 * ssim_l + 0.2 * tv_loss

            if physics_impl is not None:
                physics_loss = physics_impl(y_true, y_pred)
                return data_loss + physics_loss
            else:
                return data_loss

        return combined_loss

    # -------- training --------

    def fit(self,
            input_data,
            output_data,
            epochs=50,
            batch_size=16,
            validation_split=0.1,
            learning_rate=1e-4,
            save_model=True,
            save_dir='./Models/TTC_models/',
            model_prefix='TTC_PINN'):
        """
        input_data, output_data: arrays with shape [N, T, Q, 1]
        """

        # simple max-normalisation (adapt if you want pure g2 scaling)
        input_data = input_data / np.max(input_data)
        output_data = output_data / np.max(output_data)

        loss_fn = self.get_combined_loss()
        self.model.compile(optimizer=Adam(learning_rate=learning_rate),
                           loss=loss_fn,
                           metrics=['mae', 'mse'])

        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=3,
                                       restore_best_weights=True)

        hist = self.model.fit(
            input_data,
            output_data,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_split=validation_split,
            callbacks=[early_stopping]
        )

        now = datetime.now()
        model_version = f"{now.day:02d}{now.month:02d}{now.hour:02d}{now.minute:02d}"
        physics_tag = self.physics_type if self.physics_type is not None else "none"
        model_name = f"{model_prefix}_{physics_tag}_{model_version}_N{len(input_data)}"

        if save_model:
            import os
            os.makedirs(save_dir, exist_ok=True)
            path = save_dir + 'model_' + model_name
            self.model.save(path)
        else:
            path = None

        return hist, model_name, path

    # -------- loading & prediction --------

    def load_trained_model(self, filepath):
        """
        Reload a trained model with the correct custom loss attached.
        You must initialize TTCPhysicsAutoencoder with the same physics_type & config
        as used for training.
        """
        loss_fn = self.get_combined_loss()
        self.model = tf.keras.models.load_model(
            filepath,
            custom_objects={'combined_loss': loss_fn}
        )
        self.model.summary()

    def denoise_single(self, input_image):
        """
        input_image: 2D or 3D TTC [T, Q] or [T, Q, 1]
        returns: reconstructed TTC [T, Q, 1]
        """
        arr = np.array(input_image, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]  # [T, Q, 1]

        arr = arr / np.max(arr)
        arr = arr[np.newaxis, ...]  # [1, T, Q, 1]

        recon = self.model.predict(arr)
        return recon[0]
