#%%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from ttc_ai_v2 import (
    XPCS_PINN_Denoiser,
    PhysicsInformedLoss
)
from datetime import datetime
import os
import importlib
import ttc_analyzer
importlib.reload(ttc_analyzer)
from ttc_analyzer import TTCAnalyzer




def plot_ttc_grid(
    *,
    columns,                 
    rows,                    
    data, 
    q=None,
    q_unit="nm⁻¹",                   
    cmap="viridis",
    vmax=0.05,
    origin="lower",
    figsize_per_cell=(3.2, 3.2),
    fontsize=8,
    add_colorbar=True,
    hide_axes=True,
    save=False,              
    save_path=".",           
    save_name=None,          
    dpi=300                  
):
    """
    Generic TTC grid plotter with column titles and optional saving.
    - Each column gets ONE title (shown on the top row).
    - Global figure title includes q if provided: "TTCF at q = ... nm⁻¹"
    - data format:
        data[row_key][col_index] = 2D array (or None to leave blank)

    Example:
        columns = ["GT - 1300", "5 avg", "25 avg"]
        rows = ["raw", "den1", "den2"]
        data = {
            "raw":  [ttc_gt, ttc_raw_5,  ttc_raw_25],
            "den1": [ttc_d1_1300, ttc_d1_5, ttc_d1_25],
            "den2": [ttc_d2_1300, ttc_d2_5, ttc_d2_25],
        }
        plot_ttc_grid(columns=columns, rows=rows, data=data, q=0.032, save=True)
    """

    if not isinstance(columns, (list, tuple)) or len(columns) == 0:
        raise ValueError("columns must be a non-empty list/tuple of column titles.")
    if not isinstance(rows, (list, tuple)) or len(rows) == 0:
        raise ValueError("rows must be a non-empty list/tuple of row keys.")
    if not isinstance(data, dict) or len(data) == 0:
        raise ValueError("data must be a non-empty dict: data[row_key] -> list of images.")

    n_cols = len(columns)
    n_rows = len(rows)

    # validate row lengths
    for rk in rows:
        if rk not in data:
            raise ValueError(f"Row key '{rk}' is missing from data.")
        if not isinstance(data[rk], (list, tuple)):
            raise ValueError(f"data['{rk}'] must be a list/tuple with length == len(columns).")
        if len(data[rk]) != n_cols:
            raise ValueError(
                f"data['{rk}'] length is {len(data[rk])}, but columns length is {n_cols}."
            )

    fig_w = figsize_per_cell[0] * n_cols
    fig_h = figsize_per_cell[1] * n_rows

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False
    )

    # global title (q)
    if q is not None:
        fig.suptitle(f"TTCF at q = {q} {q_unit}", fontsize=fontsize + 4, y=1.02)

    # column titles (top row only)
    for c, title in enumerate(columns):
        axes[0, c].set_title(str(title), fontsize=fontsize + 1)

    # plot all cells
    for r, row_key in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r, c]
            img = data[row_key][c]

            if img is None:
                ax.axis("off")
                continue

            im = ax.imshow(img, cmap=cmap, origin=origin, vmax= vmax)

            if hide_axes:
                ax.axis("off")

            if add_colorbar:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # leave room for suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # saving
    if save:
        os.makedirs(save_path, exist_ok=True)

        if save_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            q_part = f"_q{q}" if q is not None else ""
            save_name = f"TTC_grid{q_part}_{timestamp}.png"
        elif not save_name.lower().endswith((".png", ".pdf", ".svg")):
            save_name += ".png"

        full_path = os.path.join(save_path, save_name)
        fig.savefig(full_path, dpi=dpi, bbox_inches="tight")
        print(f"✅ Figure saved to: {full_path}")

    plt.show()
    return fig

def image_normalization( img,b= 0, c = 1):
    img_max = np.max(img)
    img_min = np.min(img)
    img = np.clip(img, b, c)
    img_normalized = (img - img_min) / (img_max - img_min)
    return img_normalized


def image_padding(img, original_size= 100, padding_size=100):
    p_window = int((padding_size - original_size) / 2)
    img_p = np.zeros((padding_size, padding_size))
    img_p[0:original_size, 0:original_size ] =  img[0:original_size, 0:original_size]
    return img_p

class XPCSPINNEvaluator:
    """
    Helper for loading a trained XPCS PINN, testing it on data,
    and plotting TTCF maps + g2(τ) curves.
    """

    def __init__(self, model, loss_fn=None):
        self.model = model
        self.loss_fn = loss_fn

    # ---------- weights ----------

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)
        print(f"✅ Loaded weights from: {weights_path}")

    def denoise(self, noisy_data, batch_size=16):
        noisy_data = noisy_data.astype("float32")
        denoised = self.model.predict(noisy_data, batch_size=batch_size, verbose=1)
        return denoised


#%%

########## Loading data (eggyolk)############################################

number_of_q_values_eggyolk = 20

q_values_eggyolk = np.array([0.095     , 0.105     , 0.115     , 0.125     , 0.135     ,
       0.145     , 0.155     , 0.165     , 0.175     , 0.185     ,
       0.195     , 0.205     , 0.215     , 0.225     , 0.235     ,
       0.245     , 0.255     , 0.265     , 0.275     , 0.28854377])

contrast_eggyolk =np.array([0.05188205, 0.05192311, 0.05136611, 0.05171926, 0.05108087,
       0.05104867, 0.05020785, 0.04957793, 0.0491428 , 0.04859964,
       0.04808463, 0.04736483, 0.04679808, 0.04616146, 0.04531015,
       0.04457282, 0.04359917, 0.04177554, 0.04105258, 0.0395628 ])


ave_eggyolk = [5,25,50,100,200,400,600,800,1000,1200,1300]

ttc_eggyolk = np.zeros((len(ave_eggyolk),number_of_q_values_eggyolk, 100,100))

SE_ttc_eggyolk = np.zeros((len(ave_eggyolk),number_of_q_values_eggyolk, 100, 100))


for idx, ab in enumerate(ave_eggyolk):
    s_ttc = 1
    print(f'Loading data for {ab} ttc averaging...')
    file_name =f'./yolkplasma/{ab}ttc_averaged/Avg_TTC_{s_ttc}_{ab+s_ttc}.npy'
    file_name_se =f'./yolkplasma/{ab}ttc_averaged/SE_TTC_{s_ttc}_{ab+s_ttc}.npy'
    ttc_eggyolk_ = np.load(file_name)[: number_of_q_values_eggyolk]
    se_ttc_eggyolk_ = np.load(file_name_se)[0:number_of_q_values_eggyolk]
    for id, ttc in enumerate(ttc_eggyolk_):
        np.fill_diagonal(ttc, contrast_eggyolk[id])
        ttc_eggyolk[idx, id] = ttc
    SE_ttc_eggyolk[idx, 0:number_of_q_values_eggyolk] = se_ttc_eggyolk_
    

#%%
############# loading model #################################################





model_res_egg = XPCS_PINN_Denoiser()
loss_fn_res_egg = PhysicsInformedLoss(
    lambda_recon=5.0,
    lambda_1tcf=0.0,
    lambda_symmetry=3.0,
    lambda_siegert=0.3,
    lambda_causality=0.3,
    lambda_boundary=0.0,
    lambda_smoothness=0.05,
    lambda_baseline=0.5,
    lambda_contrast= 0.0,
    lambda_fft=0.3,          
    fft_use_log=True,
    fft_use_ttc_minus_one=True,
    fft_r_min=0.08,
    fft_r_max=0.85
)
#ttc_AE_V1_PINN_fft_epoch_10_20260122_200848.weights
#ttc_AE_V1_PINN_fft_epoch_5_20260122_064154.weights
res_weights_path = "./test_xpcs_pinn_checkpoints/ttc_AE_V1_PINN_fft_epoch_10_20260122_200848.weights.h5"
evaluator_res_egg = XPCSPINNEvaluator(model_res_egg, loss_fn_res_egg)
dummy_input = tf.random.normal([1, 100, 100, 1])
out = model_res_egg(dummy_input)
evaluator_res_egg.load_weights(res_weights_path)
#%%

def image_normalization_v2( img, w = 10):
    img_max_in = np.max(img[w:-w, w:-w])
    img_min_in = np.min(img[w:-w, w:-w])
    img_max_o = np.max(img)
    img_min_o = np.min(img)
    
    img = (img - img_min_o) / (img_max_o - img_min_o)
    img = np.clip(img , img_min_o,img_max_in)
    img_max_in = np.max(img)
    img_min_in = np.min(img)
    img = (img - img_min_in) / (img_max_in - img_min_in)
    #img[:10, :10] = (img[:10, :10] - img_min_o) / (img_max_o - img_min_o)
    #img[0:7,0:7] = img[0:7,0:7] - abs(img_max_o-img_max_in)
    return img

def replace_border_with_interior(img, m=3, mode="nearest"):
    """
    Replace an m-pixel border (all 4 sides) with interior values.
    mode:
      - "nearest": copy the nearest interior row/col outward
      - "median": fill border with median of interior
    """
    out = img.copy()
    H, W = out.shape
    
    if 2*m >= H or 2*m >= W:
        raise ValueError("m too large for image size.")

    interior = out[m:H-m, m:W-m]

    if mode == "nearest":
        # top border
        out[:m, m:W-m] = out[m, m:W-m]
        # bottom border
        out[H-m:, m:W-m] = out[H-m-1, m:W-m]
        # left border
        out[:, :m] = out[:, m:m+1]
        # right border
        out[:, W-m:] = out[:, W-m-1:W-m]
    elif mode == "median":
        med = np.median(interior)
        out[:m, :] = med
        out[H-m:, :] = med
        out[:, :m] = med
        out[:, W-m:] = med
    else:
        raise ValueError("mode must be 'nearest' or 'median'")
        
    return out




#%%
def remove_bad_pixels(ttc, m, keep_percent=50):
    thr = np.percentile(m, keep_percent)  # keep best pixels
    good = m <= thr
    out = ttc.copy().astype(float)
    out[~good] = 0
    return out, good

def pinn_map_to_weights(m, eps=1e-6, power=1.0):
    """
    Convert PINN map to weights.
    Larger m = less trust.
    """
    w = 1.0 / (m+ eps)**power
    w /= w[10:90, 10:90].max()
    return w

def denoise_ttc(ttc_input, SE_input, contrast_id, size=100):
    ttc_input_n = image_normalization(ttc_input, -1,1)
    re_ = evaluator_res_egg.denoise(ttc_input_n.reshape(1,100,100,1))
    re_ = re_.reshape(100,100)
     #*contrast_eggyolk[contrast_id]
    #re_ = np.clip(re_ , contrast_eggyolk[contrast_id],1)
    #re_ = image_normalization_v2(re_)
    #re_ = pinn_map_to_weights(1-re_)
    re_ = image_normalization_v2(re_,40)#/contrast_eggyolk[contrast_id]
    #re_ = np.clip(re_ ,0,1)
    re_1 = ttc_input* (ttc_input_n *re_)
    #re_1 = image_normalization(re_1, -1,1) *contrast_eggyolk[contrast_id]
    #plt.imshow(re_1, cmap='viridis', vmax=0.05)
    #plt.title('1')
    #plt.colorbar()
    #plt.show()

    re_ = image_normalization(re_1, -1,1)
    re_ = evaluator_res_egg.denoise(re_.reshape(1,100,100,1))
    re_ = re_.reshape(100,100)
    re_ = image_normalization_v2(re_,40) 
    re_2 = ttc_input * ttc_input_n * re_#* contrast_eggyolk[contrast_id]
    #plt.imshow( re_2 , cmap='viridis')
    #plt.title('2')
    #plt.colorbar()

    #plt.show()
    #plt.imshow( ttc_input, cmap='viridis')
    #plt.title('TTCF')
    #plt.colorbar()

    #plt.show()
  
    return  re_1, re_2



for q_idx in range(number_of_q_values_eggyolk):
    ttc_eggyolk_denoised_q = np.zeros((len(ave_eggyolk), 100,100))
    ttc_eggyolk_denoised_2_q = np.zeros((len(ave_eggyolk),100,100))   
    for count, ab in enumerate(ave_eggyolk):
        print(ab)
        #count = count + 5
        ttc_input = ttc_eggyolk[count,q_idx]
        se_input = SE_ttc_eggyolk[count,q_idx]
        ttc_eggyolk_denoised_q[count], ttc_eggyolk_denoised_2_q[count] = denoise_ttc(ttc_input, se_input, q_idx)
        #ttc_eggyolk_denoised_2_q[count] = denoise_ttc(ttc_eggyolk_denoised_q[count], q_idx, size=90)
    np.save(f'./denoised_ttc/eggyolk/1step_denoising/ttc_eggyolk_denoised_q{q_idx+1}.npy', ttc_eggyolk_denoised_q)
    np.save(f'./denoised_ttc/eggyolk/2step_denoising/ttc_eggyolk_denoised_twice_q{q_idx+1}.npy', ttc_eggyolk_denoised_2_q)


#%% 
################################# Plotting and analysis######################################

def prepare_data(raw, denoised_1, denoised_2:None, average_s:int, average_e:int):

    raw = np.asarray(raw)
    den1 = np.asarray(denoised_1)

    if raw.ndim < 3:
        raise ValueError(f"`raw` should be (n_avg, H, W). Got shape {raw.shape}")
    if den1.ndim < 3:
        raise ValueError(f"`denoised_1` should be (n_avg, H, W). Got shape {den1.shape}")

    n_raw = raw.shape[0]
    n_den1 = den1.shape[0]
    if n_raw != n_den1:
        raise ValueError(f"raw n_avg ({n_raw}) != denoised_1 n_avg ({n_den1})")

    if average_e is None:
        average_e = n_raw - 1

    if average_s < 0 or average_e < 0:
        raise ValueError("average_s/average_e must be non-negative.")
    if average_s > average_e:
        raise ValueError("average_s must be <= average_e.")
    if average_e >= n_raw:
        raise ValueError(f"average_e={average_e} out of range for n_avg={n_raw}.")

    sl = slice(average_s, average_e + 1)  # inclusive end

    data = {
        "raw":  [raw[i]  for i in range(average_s, average_e + 1)],
        "den1": [den1[i] for i in range(average_s, average_e + 1)],
    }

    if denoised_2 is not None:
        den2 = np.asarray(denoised_2)
        if den2.ndim < 3:
            raise ValueError(f"`denoised_2` should be (n_avg, H, W). Got shape {den2.shape}")
        if den2.shape[0] != n_raw:
            raise ValueError(f"raw n_avg ({n_raw}) != denoised_2 n_avg ({den2.shape[0]})")

        data["den2"] = [den2[i] for i in range(average_s, average_e + 1)]

    return data




# select data to be plotted and analysis 
q_value_index = 10 
ttc_deoinsed = np.load(f'./denoised_ttc/eggyolk/1step_denoising/ttc_eggyolk_denoised_q{q_value_index+1}.npy')
ttc_deoinsed_2 = np.load(f'./denoised_ttc/eggyolk/2step_denoising/ttc_eggyolk_denoised_twice_q{q_value_index+1}.npy')
#%%
print(ttc_eggyolk[:, q_value_index].shape)
print(ttc_deoinsed.shape)
print(ttc_deoinsed_2.shape)
#%%
data = prepare_data(ttc_eggyolk[:, q_value_index], ttc_deoinsed, ttc_deoinsed_2, 0, 10)
#%%
plot_ttc_grid(columns=ave_eggyolk, rows=["raw", "den1", "den2"], data=data, q=q_values_eggyolk[q_value_index])

#%%

analyzer = TTCAnalyzer(eps=1e-8)