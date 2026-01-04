
#%%

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
#%%

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
        #print(np.max(noisy_data))
        denoised = self.model.predict(noisy_data, batch_size=batch_size, verbose=0)
        #denoised = denoised/np.max(denoised)
        #denoised = self.model.predict(denoised, batch_size=batch_size, verbose=0)
        #denoised = denoised/np.max(denoised)

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
        return denoised, target_data

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
        denoised_data,
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

        N = denoised_data.shape[0]
        num_examples = min(num_examples, N)
        indices = np.random.choice(N, size=num_examples, replace=False)

        # Denoise subset
        noisy_subset = noisy_data[indices]
        denoised_subset = denoised_data[indices]

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


#%%

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

evaluator = XPCSPINNEvaluator(model=model, loss_fn=loss_fn)
dummy_input = tf.random.normal([1, t_bins, t_bins, 1])
out = model(dummy_input)

model.summary()


weights_path = "./test_xpcs_pinn_checkpoints/ttc_AE_V1_PINN_fft_epoch_10_20251220_173002.weights.h5"
evaluator.load_weights(weights_path)

# 5. TTCF triplet plots
reconstructed_images, gt_images = evaluator.plot_ttc_examples(
    noisy_data=noisy_data,
    target_data=target_data,
    num_examples=6,
    save_path="ttcf_examples.png",
)

#%%

print(np.max(reconstructed_images))
print(np.max(gt_images))


evaluator.plot_g2_examples(
    noisy_data=noisy_data,
    target_data=gt_images,
    denoised_data=reconstructed_images,
    num_examples=3,
    save_path="g2_examples.png",
)


model = XPCS_PINN_Denoiser()
loss_fn = PhysicsInformedLoss(
    lambda_recon=5.0,
    lambda_1tcf=0.0,
    lambda_symmetry=5.3,
    lambda_siegert=0.5,
    lambda_causality=0.5,
    lambda_boundary=0.3,
    lambda_smoothness=0.05,
    lambda_baseline=1.0,
    lambda_contrast= 1.0,
    lambda_fft=1.0,          
    fft_use_log=True,
    fft_use_ttc_minus_one=True,
    fft_r_min=0.05,
    fft_r_max=0.95
)

evaluator = XPCSPINNEvaluator(model=model, loss_fn=loss_fn)
dummy_input = tf.random.normal([1, t_bins, t_bins, 1])
out = model(dummy_input)

model.summary()

reconstructed_images = reconstructed_images/np.max(reconstructed_images)
gt_images = gt_images /np.max(gt_images)

reconstructed_images_, gt_images_ = evaluator.plot_ttc_examples(
    noisy_data=reconstructed_images,
    target_data=gt_images,
    num_examples=6,
    save_path="ttcf_examples.png",
)


# 6. g2(τ) comparison plots
evaluator.plot_g2_examples(
    noisy_data=reconstructed_images,
    target_data=gt_images_,
    denoised_data=reconstructed_images_,
    num_examples=3,
    save_path="g2_examples.png",
)
#%%

def image_normalization( img):
    img_max = np.max(img)
    img_min = np.min(img)
    img_normalized = (img - img_min) / (img_max - img_min)
    return img_normalized


def image_padding(img, original_size= 100):
    img_p = np.zeros((100, 100))
    img_p[0:original_size, 0:original_size ] =  img[0:original_size, 0:original_size]
    return img_p

def extract_g2_from_ttc(ttc, contrast = 1.0):
    height, width = ttc.shape
    max_tau = min(height, width)
    ttc = ttc
    g2 = []
    for tau in range(max_tau):
        diag = np.diagonal(ttc, offset=tau)
        
        max_diag = np.max(diag)

        diag_normalized = diag / max_diag
        diag_mean = np.mean(diag)
        if np.isnan(diag_mean):
            diag_mean = 0
        g2.append(diag_mean)


    g2 = np.array(g2)
    g2_normalized = (g2 - np.min(g2)) / (np.max(g2) - np.min(g2))

    return g2_normalized*contrast

#%%

q_values = np.array([0.0075    , 0.0085    , 0.0095    , 0.0105    , 0.0115    ,
                    0.0125    , 0.0135    , 0.0145    , 0.0155    , 0.0165    ])

contrast = np.array([0.05696044, 0.05510423, 0.05531992, 0.05439782, 0.05282186,
       0.05300923, 0.05067948, 0.05072688, 0.04531849, 0.05046893])

ave_in = [1,5,15,50,100,200,400]
#%%


ttc_nimmi = np.zeros((len(ave_in),10, 64,64))

SE_ttc_nimmi = np.zeros((len(ave_in),10, 64, 64))

ttc_nimmi_denoised = np.zeros((len(ave_in),10, 64,64))
ttc_nimmi_denoised_2 = np.zeros((len(ave_in),10, 64,64))


for idx, ab in enumerate(ave_in):
    
    if ab == 400:
        file_name = f'./NIMMI_DATA/{ab}ttc_averaged/Avg_TTC_{0}_{ab}.npy'
        file_name_se = f'./NIMMI_DATA/{ab}ttc_averaged/SE_TTC_{0}_{ab}.npy'
    else:
        file_name =f'./NIMMI_DATA/{ab}ttc_averaged/Avg_TTC_{0+ab}_{ab*2}.npy'
        file_name_se =f'./NIMMI_DATA/{ab}ttc_averaged/SE_TTC_{0+ab}_{ab*2}.npy'

    ttc_nimmi_ = np.load(file_name)[:10]
    se_ttc_nimmi_ = np.load(file_name_se)[:10]

    for id, ttc in enumerate(ttc_nimmi_):
        for cont in contrast:
            ttc = image_padding(ttc , 64)
            np.fill_diagonal(ttc, cont)
            
            #ttc = image_normalization(ttc)
            re_ = evaluator.denoise(ttc.reshape(1,100,100,1))
            #re_ = re_/np.max(re_)
            #re_ = evaluator.denoise(re_)
            ttc_nimmi[idx, id] = ttc[0:64, 0:64]
            denoised_ttc = re_[0,0:64, 0:64].reshape(64,64)
            ttc_nimmi_denoised[idx, id] = denoised_ttc

            np.fill_diagonal(denoised_ttc, np.max(denoised_ttc))
            denoised_ttc = image_padding(denoised_ttc, 64)
            re_2 = evaluator.denoise(denoised_ttc.reshape(1,100,100,1))
            ttc_nimmi_denoised_2[idx, id] = re_2[0,0:64, 0:64].reshape(64,64)
            
            
    SE_ttc_nimmi[idx, 0:10, 0:64, 0:64] = se_ttc_nimmi_


#%%
def subtract_noise_np(image, noise, idx):
    """
    Pixel-wise noise subtraction with floor at zero.

    image : np.ndarray, shape (..., H, W) or (..., H, W, 1)
    noise : np.ndarray, same shape as image

    returns: np.ndarray, same shape
    """
    image = image_normalization(image)
    noise = image_normalization(noise)
    clean = image - noise
    clean = np.maximum(clean, 0.0)
    #np.fill_diagonal(clean, np.max(clean) )
    #clean = image_normalization(clean)
    return clean  

#%%
fig, axs = plt.subplots(3, 7, figsize=(50, 20))

q_idx = 5
for idx, ab in enumerate(ave_in):
    axs[0, idx].imshow(ttc_nimmi[idx,q_idx], origin='lower')
    axs[0, idx].set_title(f'{ab}TTCs', fontsize=40, fontweight='bold' )
    
    axs[1, idx].imshow(subtract_noise_np(ttc_nimmi[idx,q_idx, 5:59, 5:59],ttc_nimmi_denoised[idx,q_idx, 5:59, 5:59], idx), origin='lower')

    axs[1, idx].set_title(f'{ab}TTCs', fontsize=40, fontweight='bold' )
    
    
    
    axs[2, idx].imshow(np.max(ttc_nimmi_denoised_2[idx,q_idx])-ttc_nimmi_denoised_2[idx,q_idx], origin='lower')
    axs[2, idx].set_title(f'{ab}TTCs_2', fontsize=40, fontweight='bold' )
    

    axs[0, idx].axis('off')
    axs[1, idx].axis('off')
    axs[2, idx].axis('off')
#plt.tight_layout()
plt.savefig(f'./denoised_TTC_q_{q_idx}.jpg', dpi=300, bbox_inches='tight', transparent=False,format='jpg')

plt.show()






#%%
from ttc_analyzer import TTCAnalyzer

analyzer = TTCAnalyzer(eps=1e-15)

p0 = [0,-0.01,1e-6,1]
bound_lower=[0,-0.1,1e-8,0.1]
bound_higher=[10,1.5,1e3,2]

abs_id, q_id = 1,5

res_gt = analyzer.ttc_full_pipeline(ttc=ttc_nimmi[6,5], 
                                    std =SE_ttc_nimmi[6,5], 
                                    t_delay=0.222, 
                                    ttc_slices_num=4, 
                                    log_avg_n=64,
                                    p0=p0,
                                    bound_higher=bound_higher,
                                    bound_lower=bound_lower)



denoised_ttc_res = subtract_noise_np(ttc_nimmi[abs_id, q_id, 5:59, 5:59],ttc_nimmi_denoised[abs_id, q_id, 5:59, 5:59], abs_id)

res_de = analyzer.ttc_full_pipeline(ttc=denoised_ttc_res, 
                                    std =SE_ttc_nimmi[6,5], 
                                    t_delay=0.222, 
                                    ttc_slices_num=4, 
                                    log_avg_n=64,
                                    p0=p0,
                                    bound_higher=bound_higher,
                                    bound_lower=bound_lower)





#%%
def plot_g2(
    tdelay_section,
    g2_cuts,
    g2_cuts_err,
    t_w,
    sections=None,
    ax=None,
):
    """
    Plot g2(tau) with error bars for selected waiting-time sections.

    Parameters
    ----------
    tdelay_section : array (num_sections, num_bins)
        Delay times (µs)
    g2_cuts : array (num_sections, num_bins)
        g2 values
    g2_cuts_err : array (num_sections, num_bins)
        g2 uncertainties
    t_w : array (num_sections,)
        Waiting times
    sections : list or None
        Which sections to plot (default: all)
    ax : matplotlib axis or None
        Existing axis or new one
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    if sections is None:
        sections = range(len(t_w))

    for i in sections:
        tau = tdelay_section[i]
        g2 = g2_cuts[i]
        g2_err = g2_cuts_err[i]

        mask = ~np.isnan(g2)

        ax.errorbar(
            tau[mask],
            g2[mask],
            yerr=g2_err[mask],
            fmt="o-",
            ms=4,
            capsize=2,
            label=f"t_w = {t_w[i]:.2e}"
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Delay time $\tau$ (µs)")
    ax.set_ylabel(r"$g_2(\tau)$")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    return ax

def plot_global_g2(tau_us, g2, g2_err):
    mask = ~np.isnan(g2)
    plt.figure(figsize=(6,4))
    plt.errorbar(tau_us[mask], g2[mask], yerr=g2_err[mask], fmt="o-", ms=4, capsize=2)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Delay time $\tau$ (µs)")
    plt.ylabel(r"$g_2(\tau)$")
    plt.grid(True, which="both", alpha=0.3)
    plt.show()
    
    


