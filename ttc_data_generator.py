import numpy as np


class SimulatedTTCTimeTimeNumpyQAging:
    """
    Pure NumPy simulator for TTC(τ1, τ2) images (T x T) for multiple q-rings,
    multiple sample systems, and complex aging.

    Physics summary
    ---------------
    For each system we draw system-level parameters from ranges:

      D_sys              : sets base time scale (diffusion-like)
      gamma_scale_sys    : extra scaling → Γ0(q) = gamma_scale_sys * D_sys * q^2
      beta_sys           : speckle contrast
      alpha_sys          : stretching exponent
      M_sys              : base number of independent pairs

      aging_rate         : power-law aging of Γ
      aging_exp          : exponent in power-law aging of Γ
      gamma_aging_rate   : extra exponential slowing of Γ

      beta_aging_rate, beta_aging_exp : aging of β (contrast decay)
      M_aging_rate, M_aging_exp       : aging of M (statistics grow)

    Aging model
    -----------
    Γ0(q) = gamma_scale_sys * D_sys * q^2

    For age index a = 0,1,...

      Γ(q,a)  = Γ0(q)
                * exp(-gamma_aging_rate * a)
                / (1 + aging_rate * a)^aging_exp

      β(a)    = beta_sys / (1 + beta_aging_rate * a)^beta_aging_exp

      M(a)    = M_sys * (1 + M_aging_rate * a)^M_aging_exp

    For each (system, age, q):

      g1(τ; q, a) = exp( - [Γ(q,a) τ]^alpha_sys )
      g2(τ; q, a) = 1 + β(a) |g1(τ; q, a)|^2

      TTC_q,a(τ1,τ2) = g2(|τ1 - τ2|; q, a)

    Schätzel-like noise:
      Var[g2(τ; q, a)] ≈ (g2(τ; q, a) - 1)^2 / M(a)
      σ(τ; q, a) = |g2(τ; q, a) - 1| / sqrt(M(a))

      Noise(τ1, τ2) ~ N(0, σ(Δτ; q, a)^2)
    """

    def __init__(self,
                 T=64,
                 dt=1.0,
                 # q-grid
                 q_values=None,
                 num_q_rings=20,
                 q_min=0.01,
                 q_max=0.2,
                 # GLOBAL ranges for system parameters
                 D_sys_range=(0.1, 1.0),
                 gamma_scale_range=(1.0, 1.0),      # NEW: system-specific scaling for Γ
                 beta_sys_range=(0.3, 0.9),
                 alpha_sys_range=(1.0, 1.0),
                 M_sys_range=(500, 5000),
                 # aging of Γ
                 aging_rate_range=(0.05, 0.5),      # power-law aging rate
                 aging_exp_range=(0.5, 1.5),        # power-law exponent
                 gamma_aging_rate_range=(0.0, 0.0), # exponential slowing of Γ
                 # aging of β
                 beta_aging_rate_range=(0.0, 0.0),
                 beta_aging_exp_range=(1.0, 1.0),
                 # aging of M
                 M_aging_rate_range=(0.0, 0.0),
                 M_aging_exp_range=(1.0, 1.0),
                 seed=None):

        self.T = int(T)
        self.dt = float(dt)

        # q-rings
        if q_values is not None:
            q_values = np.asarray(q_values, dtype=np.float32)
            self.q_values = q_values
        else:
            self.q_values = np.linspace(q_min, q_max, num_q_rings, dtype=np.float32)
        self.num_q_rings = len(self.q_values)

        # global ranges
        self.D_sys_range = D_sys_range
        self.gamma_scale_range = gamma_scale_range
        self.beta_sys_range = beta_sys_range
        self.alpha_sys_range = alpha_sys_range
        self.M_sys_range = M_sys_range

        self.aging_rate_range = aging_rate_range
        self.aging_exp_range = aging_exp_range
        self.gamma_aging_rate_range = gamma_aging_rate_range

        self.beta_aging_rate_range = beta_aging_rate_range
        self.beta_aging_exp_range = beta_aging_exp_range

        self.M_aging_rate_range = M_aging_rate_range
        self.M_aging_exp_range = M_aging_exp_range

        # RNG
        self.rng = np.random.default_rng(seed)

        # τ grid
        self.tau_values = np.arange(self.T, dtype=np.float32) * self.dt  # [T]

        # Δτ grid & index mapping
        tau1 = self.tau_values.reshape(self.T, 1)
        tau2 = self.tau_values.reshape(1, self.T)
        self.delta_tau_grid = np.abs(tau1 - tau2).astype(np.float32)     # [T,T]

        self.delta_indices = np.round(self.delta_tau_grid / self.dt).astype(int)
        self.delta_indices = np.clip(self.delta_indices, 0, self.T - 1)

    # ------------------------------------------------------------------
    # Sample one system
    # ------------------------------------------------------------------
    def _sample_system_params(self):
        """
        Draw one random system from the global parameter ranges.
        """
        D_sys = float(self.rng.uniform(*self.D_sys_range))
        gamma_scale_sys = float(self.rng.uniform(*self.gamma_scale_range))
        beta_sys = float(self.rng.uniform(*self.beta_sys_range))
        alpha_sys = float(self.rng.uniform(*self.alpha_sys_range))
        M_sys = float(self.rng.uniform(*self.M_sys_range))

        aging_rate = float(self.rng.uniform(*self.aging_rate_range))
        aging_exp = float(self.rng.uniform(*self.aging_exp_range))
        gamma_aging_rate = float(self.rng.uniform(*self.gamma_aging_rate_range))

        beta_aging_rate = float(self.rng.uniform(*self.beta_aging_rate_range))
        beta_aging_exp = float(self.rng.uniform(*self.beta_aging_exp_range))

        M_aging_rate = float(self.rng.uniform(*self.M_aging_rate_range))
        M_aging_exp = float(self.rng.uniform(*self.M_aging_exp_range))

        return {
            "D_sys": D_sys,
            "gamma_scale_sys": gamma_scale_sys,
            "beta_sys": beta_sys,
            "alpha_sys": alpha_sys,
            "M_sys": M_sys,
            "aging_rate": aging_rate,
            "aging_exp": aging_exp,
            "gamma_aging_rate": gamma_aging_rate,
            "beta_aging_rate": beta_aging_rate,
            "beta_aging_exp": beta_aging_exp,
            "M_aging_rate": M_aging_rate,
            "M_aging_exp": M_aging_exp,
        }

    # ------------------------------------------------------------------
    # Γ(q, age) with complex aging
    # ------------------------------------------------------------------
    def _gamma_q_age(self, q, age_idx, params):
        """
        Γ(q, age) = Γ0(q) * exp(-gamma_aging_rate * age)
                    / (1 + aging_rate * age)^aging_exp

        with Γ0(q) = gamma_scale_sys * D_sys * q^2
        """
        q = float(q)
        age = float(age_idx)

        D_sys = params["D_sys"]
        gamma_scale_sys = params["gamma_scale_sys"]
        aging_rate = params["aging_rate"]
        aging_exp = params["aging_exp"]
        gamma_aging_rate = params["gamma_aging_rate"]

        Gamma0 = gamma_scale_sys * D_sys * q * q

        # power-law + exponential aging
        power_factor = (1.0 + aging_rate * age) ** aging_exp
        exp_factor = np.exp(-gamma_aging_rate * age)

        Gamma = Gamma0 * exp_factor / power_factor
        return Gamma

    # ------------------------------------------------------------------
    # Age-dependent β and M
    # ------------------------------------------------------------------
    def _beta_age(self, age_idx, params):
        beta_sys = params["beta_sys"]
        beta_aging_rate = params["beta_aging_rate"]
        beta_aging_exp = params["beta_aging_exp"]

        age = float(age_idx)
        factor = (1.0 + beta_aging_rate * age) ** beta_aging_exp
        beta_a = beta_sys / factor
        return beta_a

    def _M_age(self, age_idx, params):
        M_sys = params["M_sys"]
        M_aging_rate = params["M_aging_rate"]
        M_aging_exp = params["M_aging_exp"]

        age = float(age_idx)
        factor = (1.0 + M_aging_rate * age) ** M_aging_exp
        M_a = M_sys * factor
        return M_a

    # ------------------------------------------------------------------
    # One TTC for (q, age, system)
    # ------------------------------------------------------------------
    def _generate_single_q_age(self, q, age_idx, params):
        q = float(q)
        alpha_sys = params["alpha_sys"]

        # age-dependent parameters
        Gamma = self._gamma_q_age(q, age_idx, params)
        beta_a = self._beta_age(age_idx, params)
        M_a = self._M_age(age_idx, params)

        # 1D g2(τ)
        g1_tau = np.exp(-(Gamma * self.tau_values) ** alpha_sys)  # [T]
        g2_tau = 1.0 + beta_a * (g1_tau ** 2)                     # [T]

        # Clean TTC
        TTC_clean = g2_tau[self.delta_indices]                    # [T,T]

        # Schätzel-like noise using M(a)
        var_tau = (g2_tau - 1.0) ** 2 / M_a                       # [T]
        sigma_tau = np.sqrt(var_tau).astype(np.float32)           # [T]
        sigma_grid = sigma_tau[self.delta_indices]                # [T,T]

        noise = self.rng.normal(loc=0.0, scale=1.0,
                                size=TTC_clean.shape).astype(np.float32)
        TTC_noisy = TTC_clean + noise * sigma_grid

        TTC_noisy = np.maximum(TTC_noisy, 0.0).astype(np.float32)
        TTC_clean = TTC_clean.astype(np.float32)

        return TTC_noisy, TTC_clean

    # ------------------------------------------------------------------
    # Public: one system (ages × q-rings)
    # ------------------------------------------------------------------
    def generate_system(self,
                        num_ages,
                        add_channel_dim=True,
                        return_params=False):
        """
        Generate TTCs for ONE system over multiple ages and q-rings.

        Output (without channel dim):
          X_noisy: [num_ages, num_q_rings, T, T]
          Y_clean: [num_ages, num_q_rings, T, T]
        """
        num_ages = int(num_ages)
        params = self._sample_system_params()

        X_noisy = np.zeros((num_ages, self.num_q_rings, self.T, self.T), dtype=np.float32)
        Y_clean = np.zeros_like(X_noisy)

        for a in range(num_ages):
            for iq, q in enumerate(self.q_values):
                n, c = self._generate_single_q_age(q, a, params)
                X_noisy[a, iq] = n
                Y_clean[a, iq] = c

        if add_channel_dim:
            X_noisy = X_noisy[..., np.newaxis]  # [A, Q, T, T, 1]
            Y_clean = Y_clean[..., np.newaxis]

        if return_params:
            return X_noisy, Y_clean, params
        else:
            return X_noisy, Y_clean

    # ------------------------------------------------------------------
    # Public: many systems, flattened for training
    # ------------------------------------------------------------------
    def generate_dataset(self,
                         num_systems,
                         num_ages,
                         add_channel_dim=True,
                         return_meta=False):
        """
        Generate TTC dataset for many systems and ages.

        total_samples = num_systems * num_ages * num_q_rings
        """
        num_systems = int(num_systems)
        num_ages = int(num_ages)

        all_noisy = []
        all_clean = []
        system_idx_list = []
        age_idx_list = []
        q_val_list = []
        params_list = []

        for s in range(num_systems):
            X_sys, Y_sys, params = self.generate_system(
                num_ages=num_ages,
                add_channel_dim=False,
                return_params=True
            )
            params_list.append(params)  # one dict per system

            A, Q_, T_, T__ = X_sys.shape
            assert A == num_ages and Q_ == self.num_q_rings

            X_flat = X_sys.reshape(A * Q_, T_, T_)
            Y_flat = Y_sys.reshape(A * Q_, T_, T_)

            all_noisy.append(X_flat)
            all_clean.append(Y_flat)

            for a in range(num_ages):
                for iq, q in enumerate(self.q_values):
                    system_idx_list.append(s)
                    age_idx_list.append(a)
                    q_val_list.append(float(q))

        X_noisy = np.concatenate(all_noisy, axis=0)  # [N, T, T]
        Y_clean = np.concatenate(all_clean, axis=0)  # [N, T, T]

        if add_channel_dim:
            X_noisy = X_noisy[..., np.newaxis]        # [N, T, T, 1]
            Y_clean = Y_clean[..., np.newaxis]

        if return_meta:
            meta = {
                "system_idx": np.array(system_idx_list, dtype=int),
                "age_idx": np.array(age_idx_list, dtype=int),
                "q_values": np.array(q_val_list, dtype=np.float32),
                "system_params": params_list
            }
            return X_noisy, Y_clean, meta
        else:
            return X_noisy, Y_clean

    # ------------------------------------------------------------------
    def get_tau(self):
        return self.tau_values.copy()

    def get_q_values(self):
        return self.q_values.copy()
    
class XPCSSyntheticGenerator_V2:
    def __init__(self, size=100, seed=0):
        self.size = size
        self.rng = np.random.default_rng(seed)

        t = np.arange(size, dtype=np.float32)
        t1, t2 = np.meshgrid(t, t)
        self.tau = np.abs(t1 - t2).astype(np.float32)

    @staticmethod
    def symmetrize(A):
        return 0.5 * (A + A.T)

    def make_pure_signal(self, gamma, alpha):
        # S = exp(-2 (gamma*tau)^alpha)  in [0,1]
        S = np.exp(-2.0 * (gamma * self.tau) ** alpha).astype(np.float32)
        return self.symmetrize(S)

    def schaetzel_like_noise(self, n_frames, ns, diag_factor=1.5, tail_factor=0.7, base_scale=0.3):
        N = self.size
        noise = self.rng.normal(0.0, 1.0, (N, N)).astype(np.float32)
        noise = self.symmetrize(noise)

        idx = np.arange(N, dtype=np.float32)
        tau = np.abs(idx[:, None] - idx[None, :])
        tau_norm = tau / max(tau.max(), 1.0)

        scale = diag_factor - (diag_factor - tail_factor) * tau_norm  # constant per diagonal band
        noise *= scale

        denom = np.sqrt(max(n_frames * ns, 1.0))
        noise = base_scale * noise / denom
        return noise.astype(np.float32)

    def sample_params(self):
        gamma = self.rng.uniform(0.01, 0.10)
        alpha = self.rng.uniform(0.5, 1.3)
        beta  = self.rng.uniform(0.02, 0.15)   # realistic XPCS contrast range (adjust)
        n_frames = int(self.rng.integers(200, 3000))  # averaging / frames
        ns = int(self.rng.integers(50, 2000))         # speckles
        base_scale = self.rng.uniform(0.2, 1.0)
        diag_factor = self.rng.uniform(1.2, 2.5)
        tail_factor = self.rng.uniform(0.4, 1.2)
        return gamma, alpha, beta, n_frames, ns, base_scale, diag_factor, tail_factor

    def generate(self, n_samples=1000):
        X = np.empty((n_samples, self.size, self.size, 1), dtype=np.float32)
        Y = np.empty((n_samples, self.size, self.size, 1), dtype=np.float32)
        meta = []

        for i in range(n_samples):
            gamma, alpha, beta, n_frames, ns, base_scale, diag_factor, tail_factor = self.sample_params()

            S_pure = self.make_pure_signal(gamma, alpha)               # target in [0,1]
            noise  = self.schaetzel_like_noise(n_frames, ns, diag_factor, tail_factor, base_scale)

            S_noisy = S_pure + noise                                   # can be negative ✅

            X[i, :, :, 0] = S_noisy
            Y[i, :, :, 0] = S_pure

            meta.append({
                "gamma": float(gamma),
                "alpha": float(alpha),
                "beta":  float(beta),
                "n_frames": int(n_frames),
                "ns": int(ns),
                "base_scale": float(base_scale),
                "diag_factor": float(diag_factor),
                "tail_factor": float(tail_factor),
            })

        return X, Y, meta