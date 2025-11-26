from ttc_data_generator import SimulatedTTCTimeTimeNumpyQAging
import matplotlib.pyplot as plt


T = 100
num_q_rings=2
sim = SimulatedTTCTimeTimeNumpyQAging(
    T=T,
    dt=1.0,
    num_q_rings=num_q_rings,
    q_min=0.01,
    q_max=0.2,
    D_sys_range=(1.3, 1.3),
    gamma_scale_range=(1.5, 1.6),
    beta_sys_range=(0.8, 0.9),
    alpha_sys_range=(1.0, 1.0),
    M_sys_range=(1,3),
    aging_rate_range=(0.05, 0.06),
    aging_exp_range=(0.5, 0.6),
    gamma_aging_rate_range=(0.0, 0.1),
    beta_aging_rate_range=(0.0, 0.2),
    beta_aging_exp_range=(1.0, 1.1),
    M_aging_rate_range=(0.0, 0.5),
    M_aging_exp_range=(1.5, 1.5),
    seed=123
)
num_ages=5
# One system, 5 aging steps
X_sys, Y_sys, params = sim.generate_system(num_ages=num_ages, add_channel_dim=False, return_params=True)
tau = sim.get_tau()
q_vals = sim.get_q_values()

print("System params:", params)



for a in range(num_ages):
    for b in range(num_q_rings):

        TTC_clean = Y_sys[a,b]
        plt.figure(figsize=(10,4))

        plt.subplot(1,2,1)
        plt.imshow(TTC_clean, cmap='inferno', origin='lower')
        plt.title("Clean TTC")
        plt.colorbar()
        TTC_noisy = X_sys[a,b]
        plt.subplot(1,2,2)
        plt.imshow(TTC_noisy, cmap='inferno', origin='lower')
        plt.title("Noisy TTC")
        plt.colorbar()
        
        plt.tight_layout()
        plt.show()


