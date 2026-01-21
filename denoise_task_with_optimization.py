#%%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from ttc_ai_v2 import (
    XPCS_PINN_Denoiser,
    PhysicsInformedLoss
)
def image_normalization( img):
    img_max = np.max(img)
    img_min = np.min(img)
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
    
#%% Loading Data
number_of_q_values = 10
q_values = np.array([0.0075    , 0.0085    , 0.0095    , 0.0105    , 0.0115    ,
                    0.0125    , 0.0135    , 0.0145    , 0.0155    , 0.0165    ])

contrast = np.array([0.05696044, 0.05510423, 0.05531992, 0.05439782, 0.05282186,
       0.05300923, 0.05067948, 0.05072688, 0.04531849, 0.05046893])

ave_in = [1,5,15,50,100,200,400]

ttc_nimmi = np.zeros((len(ave_in),number_of_q_values, 64,64))

SE_ttc_nimmi = np.zeros((len(ave_in),number_of_q_values, 64, 64))

ttc_nimmi_denoised = np.zeros((len(ave_in),number_of_q_values, 64,64))


for idx, ab in enumerate(ave_in):
    print(f'Loading data for {ab} ttc averaging...')
    if ab == 400:
        file_name = f'./NIMMI_DATA/{ab}ttc_averaged/Avg_TTC_{0}_{ab}.npy'
        file_name_se = f'./NIMMI_DATA/{ab}ttc_averaged/SE_TTC_{0}_{ab}.npy'
    else:
        file_name =f'./NIMMI_DATA/{ab}ttc_averaged/Avg_TTC_{0+ab}_{ab*2}.npy'
        file_name_se =f'./NIMMI_DATA/{ab}ttc_averaged/SE_TTC_{0+ab}_{ab*2}.npy'
    ttc_nimmi_ = np.load(file_name)[:number_of_q_values]
    se_ttc_nimmi_ = np.load(file_name_se)[0:number_of_q_values]
    for id, ttc in enumerate(ttc_nimmi_):
        ttc_temp_2 = ttc.copy()
        np.fill_diagonal(ttc, contrast[id])
        ttc_nimmi[idx, id] = ttc
    SE_ttc_nimmi[idx, 0:number_of_q_values, 0:64, 0:64] = se_ttc_nimmi_

#%%  
import importlib
import ttc_analyzer
importlib.reload(ttc_analyzer)
from ttc_analyzer import TTCAnalyzer
analyzer = TTCAnalyzer(eps=1e-8)
from scipy import stats
#%%

model_res = XPCS_PINN_Denoiser()
loss_fn_res = PhysicsInformedLoss(
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
    fft_r_min=1.05,
    fft_r_max=0.95
)


res_weights_path = "./test_xpcs_pinn_checkpoints/ttc_AE_V1_PINN_fft_epoch_5_20251220_134054.weights.h5"

#res_weights_path = "./test_xpcs_pinn_checkpoints/ttc_AE_V1_PINN_fft_epoch_10_20251220_173002.weights.h5"
evaluator_res = XPCSPINNEvaluator(model_res, loss_fn_res)
dummy_input = tf.random.normal([1, 100, 100, 1])
out = model_res(dummy_input)
evaluator_res.load_weights(res_weights_path)

#%% Denoising and plotting results

ttc_gt = ttc_nimmi[6,4]
ttc_gt = image_padding(ttc_gt , 64)
ttc_gt = image_normalization(ttc_gt)

re_ = evaluator_res.denoise(ttc_gt.reshape(1,100,100,1))
gt_denoised_reshaped = re_[0,10:50,10:50].reshape(40,40)
gt_denoised_reshaped = gt_denoised_reshaped / contrast[4]
gt_denoised_reshaped = image_normalization(gt_denoised_reshaped)
gt_ttc_recostructed = ttc_gt[10:50,10:50] -(ttc_gt[10:50,10:50] * (1-gt_denoised_reshaped))



ttc_noisy = ttc_nimmi[1,4] 
ttc_noisy = image_padding(ttc_noisy , 64)
ttc_noisy = image_normalization(ttc_noisy)

re_ = evaluator_res.denoise(ttc_noisy.reshape(1,100,100,1))
denoised_reshaped = re_[0,10:50,10:50].reshape(40,40)
denoised_reshaped = denoised_reshaped / contrast[4]
denoised_reshaped = image_normalization(denoised_reshaped)
ttc_recostructed = ttc_noisy[10:50,10:50] -(ttc_noisy[10:50,10:50] * (1-denoised_reshaped))

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.title('Ground Truth TTCF', fontsize=14)
plt.imshow(image_padding(gt_ttc_recostructed, 40,64), cmap='viridis', origin='lower')
plt.colorbar()
#plt.axis('off')
plt.subplot(1,3,2)
plt.title('Noisy TTCF', fontsize=14)
plt.imshow(ttc_noisy[0:64,0:64],origin='lower', cmap='viridis')
plt.colorbar()
plt.axis('off')
plt.subplot(1,3,3)
plt.title('Denoised TTCF', fontsize=14)
plt.imshow(image_padding(ttc_recostructed, 40,64), cmap='viridis', origin='lower')
plt.colorbar()
plt.axis('off')
plt.show()

#%%
def correct_ttc(ttc, contrast_id):
    ttc_gt = image_padding(ttc , 64)
    ttc_gt = image_normalization(ttc_gt)
    re_ = evaluator_res.denoise(ttc_gt.reshape(1,100,100,1))
    gt_denoised_reshaped = re_[0,10:50,10:50].reshape(40,40)
    gt_denoised_reshaped = gt_denoised_reshaped / contrast[contrast_id]
    gt_denoised_reshaped = image_normalization(gt_denoised_reshaped)
    gt_ttc_recostructed = ttc_gt[10:50,10:50] -(ttc_gt[10:50,10:50] * (1-gt_denoised_reshaped))
    return gt_ttc_recostructed


for q_val in range(number_of_q_values):
    for count, ab in enumerate(ave_in):
        ttc_input = ttc_nimmi[count,q_val]
        ttc_nimmi_denoised[count,q_val] = correct_ttc(ttc_input, q_val)
        
    



p0 = [0.001,-0.000001,1e-6,1]
bound_lower=[0.0001,-0.00001,1e-8,0.9]
bound_higher=[10,1.5,1e3,1.1]

gamma_q_gt = []
std_gamma_q_gt = []
gamma_q_de_1 = []
std_gamma_q_de_1 = []
gamma_q_de_5 = []
std_gamma_q_de_5 = []
gamma_q_de_15 = []
std_gamma_q_de_15 = []
gamma_q_de_50 = []
std_gamma_q_de_50 = []
for q_val in range(number_of_q_values):
    ttc_gt = correct_ttc(ttc_nimmi[6,q_val], q_val)
    ttc_noisy_1 = correct_ttc(ttc_nimmi[0,q_val], q_val)
    ttc_noisy_5 = correct_ttc(ttc_nimmi[1,q_val], q_val)
    ttc_noisy_15 = correct_ttc(ttc_nimmi[2,q_val], q_val)
    ttc_noisy_50 = correct_ttc(ttc_nimmi[3,q_val], q_val)
    res_gt = analyzer.ttc_full_pipeline(ttc=image_normalization(ttc_gt), 
                                        std =SE_ttc_nimmi[6,q_val], 
                                        t_delay=0.222, 
                                        ttc_slices_num=1, 
                                        log_avg_n=30,
                                        p0=p0,
                                        bound_higher=bound_higher,
                                        bound_lower=bound_lower)
    gamma_q_gt.append(res_gt['gamma'][0]/1000000)
    std_gamma_q_gt.append(res_gt['gamma_err'][0]/(100000*400))
    res_de = analyzer.ttc_full_pipeline(ttc=image_normalization(ttc_noisy_1), 
                                        std =SE_ttc_nimmi[6,q_val], 
                                        t_delay=0.15, 
                                        ttc_slices_num=1, 
                                        log_avg_n=150,
                                        p0=p0,
                                        bound_higher=bound_higher,
                                        bound_lower=bound_lower)
    gamma_q_de_1.append(res_de['gamma'][0]/1000000)
    std_gamma_q_de_1.append(res_de['gamma_err'][0]/(100000*1))

    res_de = analyzer.ttc_full_pipeline(ttc=image_normalization(ttc_noisy_5), 
                                        std =SE_ttc_nimmi[6,q_val], 
                                        t_delay=0.16, 
                                        ttc_slices_num=1, 
                                        log_avg_n=150,
                                        p0=p0,
                                        bound_higher=bound_higher,
                                        bound_lower=bound_lower)
    gamma_q_de_5.append(res_de['gamma'][0]/1000000)
    std_gamma_q_de_5.append(res_de['gamma_err'][0]/(100000*5))

    res_de = analyzer.ttc_full_pipeline(ttc=image_normalization(ttc_noisy_15), 
                                        std =SE_ttc_nimmi[6,q_val], 
                                        t_delay=0.17, 
                                        ttc_slices_num=1, 
                                        log_avg_n=150,
                                        p0=p0,
                                        bound_higher=bound_higher,
                                        bound_lower=bound_lower)
    gamma_q_de_15.append(res_de['gamma'][0]/1000000)
    std_gamma_q_de_15.append(res_de['gamma_err'][0]/(100000*15))

    res_de = analyzer.ttc_full_pipeline(ttc=image_normalization(ttc_noisy_50), 
                                        std =SE_ttc_nimmi[6,q_val], 
                                        t_delay=0.18, 
                                        ttc_slices_num=1, 
                                        log_avg_n=150,
                                        p0=p0,
                                        bound_higher=bound_higher,
                                        bound_lower=bound_lower)
    gamma_q_de_50.append(res_de['gamma'][0]/1000000)
    std_gamma_q_de_50.append(res_de['gamma_err'][0]/(100000*50))
#%% Plotting gamma(q) with error bars
gamma_q_gt = np.array(gamma_q_gt)
std_gamma_q_gt = np.array(std_gamma_q_gt)
gamma_q_de_1 = np.array(gamma_q_de_1)
std_gamma_q_de_1 = np.array(std_gamma_q_de_1)
gamma_q_de_5 = np.array(gamma_q_de_5)
std_gamma_q_de_5 = np.array(std_gamma_q_de_5)
gamma_q_de_15 = np.array(gamma_q_de_15)
std_gamma_q_de_15 = np.array(std_gamma_q_de_15)
gamma_q_de_50 = np.array(gamma_q_de_50)
std_gamma_q_de_50 = np.array(std_gamma_q_de_50)
plt.errorbar(q_values[0:number_of_q_values], gamma_q_gt, yerr=std_gamma_q_gt, fmt='-o', label='GT gamma(q)')
plt.errorbar(q_values[0:number_of_q_values], gamma_q_de_1, yerr=std_gamma_q_de_1, fmt='o', label='Denoised gamma(q) - 1 avg')
plt.errorbar(q_values[0:number_of_q_values], gamma_q_de_5, yerr=std_gamma_q_de_5, fmt='o', label='Denoised gamma(q) - 5 avg')     
plt.errorbar(q_values[0:number_of_q_values], gamma_q_de_15, yerr=std_gamma_q_de_15, fmt='o', label='Denoised gamma(q) - 15 avg')
plt.errorbar(q_values[0:number_of_q_values], gamma_q_de_50, yerr=std_gamma_q_de_50, fmt='o', label='Denoised gamma(q) - 50 avg')
plt.xlabel('q (1/nm)')
plt.ylabel('gamma(q)')
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.show()
#%% Plotting variance of gamma(q) as bars vs averaging number
averaging_numbers = range(5)
std_gamma_de_1 = np.array(std_gamma_q_de_1)
std_gamma_de_5 = np.array(std_gamma_q_de_5)
std_gamma_de_15 = np.array(std_gamma_q_de_15)
std_gamma_de_50 = np.array(std_gamma_q_de_50)
std_gamma_de_400 = np.array(std_gamma_q_gt)
std_means = [np.mean(std_gamma_de_1)*100, np.mean(std_gamma_de_5)*100, np.mean(std_gamma_de_15)*100, np.mean(std_gamma_de_50)*100, np.mean(std_gamma_de_400)*100]
plt.bar(averaging_numbers, std_means, width=0.6)
xticks_labels = ['1', '5', '15', '50', '400']
plt.xticks(averaging_numbers, xticks_labels)
plt.xlabel('Number of TTC Averaging')
plt.ylabel('Mean Standard Deviation of gamma(q)')
plt.grid(True, which="both", alpha=0.3)
plt.show()



#%%

plt.plot(q_values[0:number_of_q_values], gamma_q_gt, 'o-', label='GT gamma(q)')
plt.plot(q_values[0:number_of_q_values], gamma_q_de_1, 'o-', label='Denoised gamma(q) - 1 avg')
plt.plot(q_values[0:number_of_q_values], gamma_q_de_5, 'o-', label='Denoised gamma(q) - 5 avg')     
plt.plot(q_values[0:number_of_q_values], gamma_q_de_15, 'o-', label='Denoised gamma(q) - 15 avg')
plt.plot(q_values[0:number_of_q_values], gamma_q_de_50, 'o-', label='Denoised gamma(q) - 50 avg')
plt.xlabel('q (1/nm)')
plt.ylabel('gamma(q)')
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.show()


#%%
######################################## EGGYOLK DATA PROCESSING ########################################

import importlib
import ttc_analyzer
importlib.reload(ttc_analyzer)
from ttc_analyzer import TTCAnalyzer
analyzer = TTCAnalyzer(eps=1e-8)
from scipy import stats
#%%

model_res_egg = XPCS_PINN_Denoiser()
loss_fn_res_egg = PhysicsInformedLoss(
    lambda_recon=5.0,
    lambda_1tcf=0.0,
    lambda_symmetry=5.3,
    lambda_siegert=0.5,
    lambda_causality=0.5,
    lambda_boundary=0.3,
    lambda_smoothness=0.05,
    lambda_baseline=10.0,
    lambda_contrast= 1.0,
    lambda_fft=1.0,          
    fft_use_log=True,
    fft_use_ttc_minus_one=True,
    fft_r_min=1.05,
    fft_r_max=0.95
)


#res_weights_path = "./test_xpcs_pinn_checkpoints/ttc_AE_V1_PINN_fft_epoch_5_20251220_134054.weights.h5"

res_weights_path = "./test_xpcs_pinn_checkpoints/ttc_AE_V1_PINN_fft_epoch_10_20251220_173002.weights.h5"
evaluator_res_egg = XPCSPINNEvaluator(model_res_egg, loss_fn_res_egg)
dummy_input = tf.random.normal([1, 100, 100, 1])
out = model_res_egg(dummy_input)
evaluator_res_egg.load_weights(res_weights_path)

#%%
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
plt.imshow(ttc_eggyolk[3,10], cmap='viridis', origin='lower')
plt.colorbar()
plt.show()
#%%

def image_padding_v2(img, original_size= 100, padding_size=100):
    p_window = int((padding_size - original_size) / 2)
    img_p = np.zeros((padding_size, padding_size))
    img_p[p_window:original_size+p_window, p_window:original_size+p_window] =  img[0:original_size, 0:original_size]
    return img_p

def denoise_ttc(ttc_input, contrast_id, size=100):
    #print(np.min(ttc_input), np.max(ttc_input))

    min_val =  np.min(ttc_input) 
    ttc_input = ttc_input #+ abs(min_val)
    #print(np.min(ttc_input), np.max(ttc_input))
    #plt.imshow(ttc_input, cmap='viridis')
    #plt.title('Input TTCF')
    #plt.colorbar()
    #plt.show()
    pixel_padding = 10
    #print(ttc_input.shape)
    re_ = evaluator_res_egg.denoise(ttc_input.reshape(1,100,100,1))
    re_ = re_.reshape(100,100) 
    re_ = image_normalization( re_[7:93,7:93]/np.mean(ttc_input))* contrast_eggyolk[contrast_id]
    #re_ = image_normalization(re_) #* contrast_eggyolk[contrast_id]
    #print(np.min(re_[7:93,7:93]), np.max(re_[7:93,7:93]))
    #plt.imshow( re_, cmap='viridis')
    #plt.title('Denoised TTCF')
    #plt.colorbar()
    #plt.show()
    return  re_
    #gt_denoised_reshaped = re_[0,int(pixel_padding/2):size+int(pixel_padding/2),int(pixel_padding/2):size+int(pixel_padding/2)].reshape(size,size)[0:size-10,0:size-10]
    #gt_denoised_reshaped = image_normalization(gt_denoised_reshaped)
    #ttc_input_normalized = image_normalization(ttc_input[0:size-10,0:size-10])
    #gt_ttc_recostructed = (ttc_input[0:size-10,0:size-10]) *gt_denoised_reshaped  #ttc_input_normalized - ((ttc_input_normalized) * (1-gt_denoised_reshaped))

    #return gt_denoised_reshaped +contrast_eggyolk[contrast_id] #abs((1+gt_ttc_recostructed) *contrast_eggyolk[contrast_id]) 


for q_idx in range(15,18):
    ttc_eggyolk_denoised_q = np.zeros((len(ave_eggyolk), 86,86))
    #ttc_eggyolk_denoised_2_q = np.zeros((len(ave_eggyolk),80,80))   
    for count, ab in enumerate(ave_eggyolk):
        ttc_input = ttc_eggyolk[count,q_idx]
        ttc_eggyolk_denoised_q[count] = denoise_ttc(ttc_input, q_idx)
        #ttc_eggyolk_denoised_2_q[count] = denoise_ttc(ttc_eggyolk_denoised_q[count], q_idx, size=90)
    np.save(f'./denoised_ttc/eggyolk_noNormalization/1step_denoising/ttc_eggyolk_denoised_q{q_idx+1}.npy', ttc_eggyolk_denoised_q)
    #np.save(f'./denoised_ttc/eggyolk_noNormalization/2step_denoising/ttc_eggyolk_denoised_twice_q{q_idx+1}.npy', ttc_eggyolk_denoised_2_q)
#%%
ttc_deoinsed = np.load(f'./denoised_ttc/eggyolk_noNormalization/1step_denoising/ttc_eggyolk_denoised_q17.npy')
plt.imshow(ttc_eggyolk[5,10], cmap='viridis')
plt.colorbar()
plt.show()  
plt.imshow(ttc_deoinsed[5], cmap='viridis')
plt.colorbar()
plt.show()  



#%%

p0 = [0.001,-0.000001,1e-6,1]
bound_lower=[0.0001,-0.00001,1e-8,0.9]
bound_higher=[10,1.5,1e3,1.1]
t_delay=0.44e-6 # between two pulses of p006996 beamtime
delays = np.arange(0,200) * t_delay # original time
t_delay_eggyolk = delays.reshape(100, 2).mean(axis=1) #after binning

t_delay_val = 0.02
#%%
q_idx = 17  
ttc_deoinsed = np.load(f'./denoised_ttc/eggyolk_noNormalization/1step_denoising/ttc_eggyolk_denoised_q{q_idx+1}.npy')

ttc_ave_400 = ttc_deoinsed[5]
res_de_400 = analyzer.ttc_full_pipeline(ttc=ttc_ave_400,
                                        std =SE_ttc_eggyolk[10,q_idx],
                                        t_delay=0.02, 
                                        ttc_slices_num=2,  
                                        log_avg_n=20,
                                        p0=p0,
                                        bound_higher=bound_higher,
                                        bound_lower=bound_lower) 
ttc_1300_gt = ttc_eggyolk[10,q_idx]
ttc_400_gt = ttc_eggyolk[5,q_idx]
res_gt = analyzer.ttc_full_pipeline(ttc=ttc_1300_gt[5:95,5:95],
                                        std =SE_ttc_eggyolk[5,q_idx],
                                        t_delay=0.02, 
                                        log_avg_n=20,
                                        ttc_slices_num=1, 
                                        p0=p0,
                                        bound_higher=bound_higher,
                                        bound_lower=bound_lower)

res_gt_400 = analyzer.ttc_full_pipeline(ttc=ttc_400_gt[0:100-10,0:100-10],
                                        std =SE_ttc_eggyolk[5,q_idx],
                                        t_delay=0.02, 
                                        log_avg_n=20,
                                        ttc_slices_num=1, 
                                        p0=p0,
                                        bound_higher=bound_higher,
                                        bound_lower=bound_lower)

#( g2- (g2[0]-contrast))/contrast

plt.figure(figsize=(10,7))
g_2 = res_gt['g2_extracted'][0]
g2_de_400 = res_de_400['g2_extracted'][0]
g2_de_400_gt = res_gt_400['g2_extracted'][0]
plt.plot(t_delay_eggyolk[:len(g_2)], (g_2 - (g_2[0]-contrast_eggyolk[q_idx]))/contrast_eggyolk[q_idx], 'o-', label='GT g2 - 1300 avg')
plt.plot(t_delay_eggyolk[:len(g2_de_400)], (g2_de_400 - (g2_de_400[0]-contrast_eggyolk[q_idx]))/contrast_eggyolk[q_idx], 'o-', label='Denoised g2 - 400 avg')   
plt.plot(t_delay_eggyolk[:len(g2_de_400_gt)], (g2_de_400_gt - (g2_de_400_gt[0]-contrast_eggyolk[q_idx]))/contrast_eggyolk[q_idx], 'o-', label='GT g2 - 400 avg')
plt.xlabel('Delay Time (s)')
plt.ylabel('g2(tau)')
plt.xscale('log')
plt.title(f'Comparison of g2 curves at q={q_values_eggyolk[q_idx]:.3f} 1/nm', fontsize=14)
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.show()
plt.figure(figsize=(10,7))
plt.plot(t_delay_eggyolk[:len(g_2)], g_2, 'o-', label='GT g2 - 1300 avg')
plt.plot(t_delay_eggyolk[:len(g2_de_400)], g2_de_400 , 'o-', label='Denoised g2 - 400 avg')   
plt.plot(t_delay_eggyolk[:len(g2_de_400_gt)], g2_de_400_gt , 'o-', label='GT g2 - 400 avg')



plt.xlabel('Delay Time (s)')
plt.ylabel('g2(tau)')
plt.xscale('log')
plt.title(f'Comparison of g2 curves at q={q_values_eggyolk[q_idx]:.3f} 1/nm', fontsize=14)
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.show()




#%%

ttc_de_5 = denoise_ttc(ttc_eggyolk[0,10],q_idx)
ttc_de_25 = denoise_ttc(ttc_eggyolk[1,10],q_idx)
ttc_de_50 = denoise_ttc(ttc_eggyolk[2,10],q_idx)
ttc_de_100 = denoise_ttc(ttc_eggyolk[3,10],q_idx)
ttc_de_200 = denoise_ttc(ttc_eggyolk[4,10],q_idx)
ttc_de_400 = denoise_ttc(ttc_eggyolk[5,10],q_idx)

ttc_de_5_2 = denoise_ttc(ttc_de_5,q_idx, size=90)
ttc_de_25_2 = denoise_ttc(ttc_de_25,q_idx, size=90)
ttc_de_50_2 = denoise_ttc(ttc_de_50,q_idx, size=90)
ttc_de_100_2 = denoise_ttc(ttc_de_100,q_idx, size=90)
ttc_de_200_2 = denoise_ttc(ttc_de_200,q_idx, size=90)
ttc_de_400_2 = denoise_ttc(ttc_de_400,q_idx, size=90)


ttc_gt = ttc_eggyolk[10,q_idx]
ttc_de_1300 = denoise_ttc(ttc_gt,q_idx)

ttc_de_1300_2 = denoise_ttc(ttc_de_1300,q_idx, size=90)



plt.figure(figsize=(20,7))
plt.subplot(3,7,1)
plt.title('GT TTCF - 1300 avg', fontsize=6)
plt.imshow(ttc_gt, cmap='viridis', origin='lower')
plt.colorbar()
plt.axis('off')

plt.subplot(3,7,8)
plt.title('Denoised TTCF - 1300 avg', fontsize=6)
plt.imshow(ttc_de_1300, cmap='viridis', origin='lower')
plt.axis('off')
plt.colorbar()

plt.subplot(3,7,15)
plt.title('Denoised TTCF - 1300 avg (2nd denoising)', fontsize=6)
plt.imshow(ttc_de_1300_2, cmap='viridis', origin='lower')
plt.axis('off')
plt.colorbar()

plt.subplot(3,7,9)
plt.title('Denoised TTCF - 5 avg', fontsize=6)
plt.imshow(ttc_de_5, cmap='viridis', origin='lower')
plt.axis('off')
plt.colorbar()

plt.subplot(3,7,2)
plt.title('TTCF - 5 avg', fontsize=6)
plt.imshow(ttc_eggyolk[0,10], cmap='viridis', origin='lower')
plt.axis('off')
plt.colorbar()


plt.subplot(3,7,16)
plt.title('Denoised TTCF - 5 avg (2nd denoising)', fontsize=6)
plt.imshow(ttc_de_5_2, cmap='viridis', origin='lower')
plt.axis('off')
plt.colorbar()

plt.subplot(3,7,10)
plt.title('Denoised TTCF - 25 avg', fontsize=6)
plt.imshow(ttc_de_25, cmap='viridis', origin='lower')
plt.axis('off')
plt.colorbar()

plt.subplot(3,7,3)
plt.title('TTCF - 25 avg', fontsize=6)
plt.imshow(ttc_eggyolk[1,10], cmap='viridis', origin='lower')
plt.axis('off') 
plt.colorbar()

plt.subplot(3,7,17)
plt.title('Denoised TTCF - 25 avg (2nd denoising)', fontsize=6)
plt.imshow(ttc_de_25_2, cmap='viridis', origin='lower')
plt.axis('off')
plt.colorbar()

plt.subplot(3,7,11)
plt.title('Denoised TTCF - 50 avg', fontsize=6)
plt.imshow(ttc_de_50, cmap='viridis', origin='lower')
plt.axis('off')
plt.colorbar()
plt.subplot(3,7,4)
plt.title('TTCF - 50 avg', fontsize=6)
plt.imshow(ttc_eggyolk[2,10], cmap='viridis', origin='lower')
plt.axis('off')
plt.colorbar() 

plt.subplot(3,7,18)
plt.title('Denoised TTCF - 50 avg (2nd denoising)', fontsize=6)
plt.imshow(ttc_de_50_2, cmap='viridis', origin='lower')
plt.axis('off')
plt.colorbar()

plt.subplot(3,7,12)
plt.title('Denoised TTCF - 100 avg', fontsize=6)
plt.imshow(ttc_de_100, cmap='viridis', origin='lower')
plt.axis('off')
plt.colorbar()
plt.subplot(3,7,5)
plt.title('TTCF - 100 avg', fontsize=6)
plt.imshow(ttc_eggyolk[3,10], cmap='viridis', origin='lower')
plt.axis('off')
plt.colorbar() 
plt.subplot(3,7,19)
plt.title('Denoised TTCF - 100 avg (2nd denoising)', fontsize=6)
plt.imshow(ttc_de_100_2, cmap='viridis', origin='lower')
plt.axis('off')
plt.colorbar()


plt.subplot(3,7,13)
plt.title('Denoised TTCF - 200 avg', fontsize=6)
plt.imshow(ttc_de_200, cmap='viridis', origin='lower')
plt.axis('off')
plt.colorbar()

plt.subplot(3,7,6)
plt.title('TTCF - 200 avg', fontsize=6)
plt.imshow(ttc_eggyolk[4,10], cmap='viridis', origin='lower')
plt.axis('off')
plt.colorbar()

plt.subplot(3,7,20)
plt.title('Denoised TTCF - 200 avg (2nd denoising)', fontsize=6)
plt.imshow(ttc_de_200_2, cmap='viridis', origin='lower')
plt.axis('off')
plt.colorbar()

plt.subplot(3,7,14)
plt.title('Denoised TTCF - 400 avg', fontsize=6)
plt.imshow(ttc_de_400, cmap='viridis', origin='lower')
plt.axis('off')
plt.colorbar()

plt.subplot(3,7,7)
plt.title('TTCF - 400 avg', fontsize=6)
plt.imshow(ttc_eggyolk[5,10], cmap='viridis', origin='lower')
plt.axis('off')
plt.colorbar()

plt.subplot(3,7,21)
plt.title('Denoised TTCF - 400 avg (2nd denoising)', fontsize=6)
plt.imshow(ttc_de_400_2, cmap='viridis', origin='lower')
plt.axis('off')
plt.colorbar()
plt.show()


plt.savefig('./denoised_ttc/eggyolk_noNormalization/ttc_eggyolk_denoising_comparison_q11.png', dpi=300)
#%%

print(np.max(ttc_eggyolk[5,10]))
print(np.max(ttc_de_400))
print(np.max(ttc_de_400_2))
#%%
def symmetrize(A):
    return 0.5*(A + A.T)

def extract_1tcf(ttc):
    T = ttc.shape[0]
    g2 = np.empty(T, dtype=np.float32)
    for lag in range(T):
        if lag == 0:
            d = np.diag(ttc)
        else:
            d1 = np.diag(ttc, k=lag)
            d2 = np.diag(ttc, k=-lag)
            d = 0.5*(d1 + d2)
        g2[lag] = d.mean()
    return g2

def calibrate_ttc_to_beta(ttc_denoised, beta_q, corner=10, eps=1e-8, exclude_lag0=False):
    """
    Enforce baseline ~ 1 and contrast ~ beta_q on TTC by rescaling TTC- baseline.
    """
    ttc = symmetrize(ttc_denoised)

    # baseline from far corner
    b = float(np.mean(ttc[-corner:, -corner:]))

    g2 = extract_1tcf(ttc)
    lag0 = 1 if exclude_lag0 else 0

    c_den = float(g2[lag0] - b)
    scale = beta_q / (c_den + eps)

    ttc_cal = b + scale * (ttc - b)
    return ttc_cal, b, c_den, scale

ttc_400_de_clibrated = calibrate_ttc_to_beta(ttc_de_400, contrast_eggyolk[10])[0]
#%%
plt.imshow(ttc_400_de_clibrated, cmap='viridis', origin='lower')
plt.colorbar()
plt.show()
plt.imshow(ttc_de_400, cmap='viridis', origin='lower')
plt.title('Denoised TTCF - 400 avg No Normalization', fontsize=6)
plt.colorbar()
plt.show()
plt.imshow(ttc_eggyolk[10,10], cmap='viridis', origin='lower')
plt.title('Original TTCF - 1300 avg', fontsize=6)
plt.colorbar()
plt.show()





#%%
t_delay=0.44e-6 # between two pulses of p006996 beamtime
delays = np.arange(0,200) * t_delay # original time
t_delay_eggyolk = delays.reshape(100, 2).mean(axis=1) #after binning

t_delay_val = 0.02


#%%
p0 = [0.001,-0.000001,1e-6,1]
bound_lower=[0.0001,-0.00001,1e-8,0.9]
bound_higher=[10,1.5,1e3,1.1]
gamma_q_gt = []
std_gamma_q_gt = []
gamma_q_de_5 = []
std_gamma_q_de_5 = []
gamma_q_de_25 = []
std_gamma_q_de_25 = []
gamma_q_de_50 = []
std_gamma_q_de_50 = []
gamma_q_de_100 = []
std_gamma_q_de_100 = []
gamma_q_de_200 = []
std_gamma_q_de_200 = []
gamma_q_de_400 = []
std_gamma_q_de_400 = []
for q_val in range(number_of_q_values_eggyolk):
    ttc_gt = ttc_eggyolk[10,q_val]
    
    
    res_gt = analyzer.ttc_full_pipeline(ttc=image_normalization(denoise_ttc(ttc_gt,q_val)), 
                                        std =SE_ttc_eggyolk[10,q_val], 
                                        t_delay=0.08, 
                                        ttc_slices_num=1, 
                                        log_avg_n=50,
                                        p0=p0,
                                        bound_higher=bound_higher,
                                        bound_lower=bound_lower)
    
    
    
    
    gamma_q_gt.append(res_gt['gamma'][0]/1000000)
    std_gamma_q_gt.append(res_gt['gamma_err'][0]/(100000*1300))
    ttc_de_400 = denoise_ttc(ttc_eggyolk[5,q_val],q_val)
    res_de_400 = analyzer.ttc_full_pipeline(ttc=image_normalization(ttc_de_400),
                                        std =SE_ttc_eggyolk[10,q_val],
                                        t_delay=0.08, 
                                        ttc_slices_num=1, 
                                        log_avg_n=50,
                                        p0=p0,
                                        bound_higher=bound_higher,
                                        bound_lower=bound_lower)
    gamma_q_de_400.append(res_de['gamma'][0]/1000000)
    std_gamma_q_de_400.append(res_de['gamma_err'][0]/(100000*400))
    continue
    ttc_de_200 = denoise_ttc(ttc_eggyolk[4,q_val],q_val)
    res_de = analyzer.ttc_full_pipeline(ttc=image_normalization(ttc_de_200),
                                        std =SE_ttc_eggyolk[10,q_val],
                                        t_delay=0.08, 
                                        ttc_slices_num=1, 
                                        log_avg_n=50,
                                        p0=p0,
                                        bound_higher=bound_higher,
                                        bound_lower=bound_lower)
    gamma_q_de_200.append(res_de['gamma'][0]/1000000)
    std_gamma_q_de_200.append(res_de['gamma_err'][0]/(100000*200))
    
    ttc_de_100 = denoise_ttc(ttc_eggyolk[3,q_val],q_val)
    res_de = analyzer.ttc_full_pipeline(ttc=image_normalization(ttc_de_100), 
                                        std =SE_ttc_eggyolk[10,q_val],
                                        t_delay=0.08, 
                                        ttc_slices_num=1, 
                                        log_avg_n=50,
                                        p0=p0,

                                        bound_higher=bound_higher,
                                        bound_lower=bound_lower)
    gamma_q_de_100.append(res_de['gamma'][0]/1000000)
    std_gamma_q_de_100.append(res_de['gamma_err'][0]/(100000*100))  


    ttc_de_5 = denoise_ttc(ttc_eggyolk[0,q_val],q_val)
    ttc_de_25 = denoise_ttc(ttc_eggyolk[1,q_val],q_val)
    ttc_de_50 = denoise_ttc(ttc_eggyolk[2,q_val],q_val)
    
   
    res_de = analyzer.ttc_full_pipeline(ttc=image_normalization(ttc_de_5), 
                                        std =SE_ttc_eggyolk[10,q_val], 
                                        t_delay=0.06, 
                                        ttc_slices_num=1, 
                                        log_avg_n=150,
                                        p0=p0,
                                        bound_higher=bound_higher,
                                        bound_lower=bound_lower)
    gamma_q_de_5.append(res_de['gamma'][0]/1000000)
    std_gamma_q_de_5.append(res_de['gamma_err'][0]/(100000*5))

    res_de = analyzer.ttc_full_pipeline(ttc=image_normalization(ttc_de_25), 
                                        std =SE_ttc_eggyolk[10,q_val],
                                        t_delay=0.08, 
                                        ttc_slices_num=1, 
                                        log_avg_n=150,
                                        p0=p0,  
                                        bound_higher=bound_higher,
                                        bound_lower=bound_lower)
    gamma_q_de_25.append(res_de['gamma'][0]/1000000)
    std_gamma_q_de_25.append(res_de['gamma_err'][0]/(100000*25))

    res_de = analyzer.ttc_full_pipeline(ttc=image_normalization(ttc_de_50), 
                                        std =SE_ttc_eggyolk[10,q_val],
                                        t_delay=0.08, 
                                        ttc_slices_num=1, 
                                        log_avg_n=50,
                                        p0=p0,
                                        bound_higher=bound_higher,
                                        bound_lower=bound_lower)
    gamma_q_de_50.append(res_de['gamma'][0]/1000000)
    std_gamma_q_de_50.append(res_de['gamma_err'][0]/(100000*50))    
    
   
    

#%%
# plotting g2 curves for eggyolk data
plt.figure(figsize=(10,7))
q_idx = 10
g_2 = res_gt['g2_extracted'][0]
g2_de_400 = res_de_400['g2_extracted'][0]

plt.plot(t_delay_eggyolk, g_2, 'o-', label='GT g2 - 1300 avg')
plt.plot(t_delay_eggyolk, g2_de_400, 'o-', label='Denoised g2 - 400 avg')   
plt.xlabel('Delay Time (s)')
plt.ylabel('g2(tau)')
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.show()

#%%
   
# Plotting gamma(q) curves for eggyolk data
    
plt.plot(q_values_eggyolk[0:number_of_q_values_eggyolk], gamma_q_gt, 'o-', label='GT gamma(q)')
#plt.plot(q_values_eggyolk[0:number_of_q_values_eggyolk], gamma_q_de_5, 'o-', label='Denoised gamma(q) - 5 avg')
#plt.plot(q_values_eggyolk[0:number_of_q_values_eggyolk], gamma_q_de_25, 'o-', label='Denoised gamma(q) - 25 avg')     
#plt.plot(q_values_eggyolk[0:number_of_q_values_eggyolk], gamma_q_de_50, 'o-', label='Denoised gamma(q) - 50 avg')
plt.plot(q_values_eggyolk[0:number_of_q_values_eggyolk], gamma_q_de_100, 'o-', label='Denoised gamma(q) - 100 avg')
plt.plot(q_values_eggyolk[0:number_of_q_values_eggyolk], gamma_q_de_200, 'o-', label='Denoised gamma(q) - 200 avg')
plt.plot(q_values_eggyolk[0:number_of_q_values_eggyolk], gamma_q_de_400, 'o-', label='Denoised gamma(q) - 400 avg')

plt.xlabel('q (1/nm)')
plt.ylabel('gamma(q)')
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.show()  
#%% Plotting gamma(q) with error bars for eggyolk data
gamma_q_gt = np.array(gamma_q_gt)
std_gamma_q_gt = np.array(std_gamma_q_gt)
gamma_q_de_5 = np.array(gamma_q_de_5)
std_gamma_q_de_5 = np.array(std_gamma_q_de_5)
gamma_q_de_25 = np.array(gamma_q_de_25)
std_gamma_q_de_25 = np.array(std_gamma_q_de_25)
gamma_q_de_50 = np.array(gamma_q_de_50)
std_gamma_q_de_50 = np.array(std_gamma_q_de_50) 
gamma_q_de_100 = np.array(gamma_q_de_100)
std_gamma_q_de_100 = np.array(std_gamma_q_de_100)
plt.errorbar(q_values_eggyolk[0:number_of_q_values_eggyolk], gamma_q_gt, yerr=std_gamma_q_gt, fmt='-o', label='GT gamma(q)')
plt.errorbar(q_values_eggyolk[0:number_of_q_values_eggyolk], gamma_q_de_5, yerr=std_gamma_q_de_5, fmt='o', label='Denoised gamma(q) - 5 avg')
plt.errorbar(q_values_eggyolk[0:number_of_q_values_eggyolk], gamma_q_de_25, yerr=std_gamma_q_de_25, fmt='o', label='Denoised gamma(q) - 25 avg')     
plt.errorbar(q_values_eggyolk[0:number_of_q_values_eggyolk], gamma_q_de_50, yerr=std_gamma_q_de_50, fmt='o', label='Denoised gamma(q) - 50 avg')
plt.errorbar(q_values_eggyolk[0:number_of_q_values_eggyolk], gamma_q_de_100, yerr=std_gamma_q_de_100, fmt='o', label='Denoised gamma(q) - 100 avg')
plt.xlabel('q (1/nm)')
plt.ylabel('gamma(q)')            
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.show()
#%% Plotting gamma(q) curves for eggyolk data
plt.plot(q_values_eggyolk[0:number_of_q_values_eggyolk], gamma_q_gt, 'o-', label='GT gamma(q)')
#plt.plot(q_values_eggyolk[0:number_of_q_values_eggyolk], gamma_q_de_5, 'o-', label='Denoised gamma(q) - 5 avg')
#plt.plot(q_values_eggyolk[0:number_of_q_values_eggyolk], gamma_q_de_25, 'o-', label='Denoised gamma(q) - 25 avg')     
plt.plot(q_values_eggyolk[0:number_of_q_values_eggyolk], gamma_q_de_50, 'o-', label='Denoised gamma(q) - 50 avg')
plt.plot(q_values_eggyolk[0:number_of_q_values_eggyolk], gamma_q_de_100, 'o-', label='Denoised gamma(q) - 100 avg')
plt.xlabel('q (1/nm)')
plt.ylabel('gamma(q)')
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.show()      
#%%
#%% Plotting variance of gamma(q) as bars vs averaging number
averaging_numbers = range(6)
#std_gamma_de_5 = np.array(std_gamma_q_de_5)
std_gamma_de_25 = np.array(std_gamma_q_de_25)
std_gamma_de_50 = np.array(std_gamma_q_de_50)
std_gamma_de_100 = np.array(std_gamma_q_de_100)
std_gamma_q_de_200 = np.array(std_gamma_q_de_200)
std_gamma_de_400 = np.array(std_gamma_q_de_400)
std_gamma_q_gt = np.array(std_gamma_q_gt)

std_means = [ 
             np.mean(std_gamma_de_25)*100, 
             np.mean(std_gamma_de_50)*100, 
             np.mean(std_gamma_de_100)*100, 
             np.mean(std_gamma_q_de_200)*100, 
             np.mean(std_gamma_q_de_400)*100, 
             np.mean(std_gamma_q_gt)*100]
plt.bar(averaging_numbers, std_means, width=0.6)
xticks_labels = [ '25', '50', '100', '200', '400', '1300']
plt.xticks(averaging_numbers, xticks_labels)
plt.xlabel('Number of TTC Averaging')
plt.ylabel('Mean Standard Deviation of gamma(q)')
plt.grid(True, which="both", alpha=0.3)
plt.show()