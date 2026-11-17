import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, resample, firwin, lfilter, welch

# ==============================================================================
# 0. GENERACIÓN DE SEÑAL DE PRUEBA ASIMÉTRICA (fs_nyq = 192 GHz)
# ==============================================================================
BW_deseado = 96e9
fc = BW_deseado / 2       # 48 GHz
fs_nyq = 4 * fc           # 192 GHz
fs_sub = fs_nyq / 2       # 96 GHz (tasa tras diezmar por 2)

N = 16384
t = np.arange(N) / fs_nyq
rng = np.random.default_rng(42)
ruido = rng.standard_normal(N)

# Subbanda inferior [0, 48 GHz]: nivel alto + piloto en 15 GHz
b_inf = firwin(129, 44e9, fs=fs_nyq)
sub_inf = 1.0 * lfilter(b_inf, 1.0, ruido) + 0.35 * np.cos(2 * np.pi * 15e9 * t)

# Subbanda superior [48, 96 GHz]: nivel bajo (-10 dB) + piloto en 75 GHz
b_sup = firwin(129, [52e9, 92e9], pass_zero=False, fs=fs_nyq)
sub_sup = 0.32 * lfilter(b_sup, 1.0, ruido) + 0.14 * np.cos(2 * np.pi * 75e9 * t)

x_sampled = sub_inf + sub_sup

# ==============================================================================
# 1. DIEZMADO POR 2 (MUESTRAS PARES E IMPARES)
# ==============================================================================
x_e = x_sampled[0::2]
x_o = x_sampled[1::2]
M = min(len(x_e), len(x_o))
x_e, x_o = x_e[:M], x_o[:M]

# ==============================================================================
# 2. ALINEACIÓN DE MEDIO PERÍODO (-0.5 MUESTRAS) EN LA RAMA IMPAR
# ==============================================================================
f_norm = np.fft.rfftfreq(M)  # [0, 0.5] <-> [0, 48 GHz]
x_o_aligned = np.fft.irfft(np.fft.rfft(x_o) * np.exp(-1j * np.pi * f_norm), n=M)

# ==============================================================================
# 3. SEPARACIÓN POR SUMA Y RESTA
# ==============================================================================
y_inf = x_e + x_o_aligned          # Subbanda inferior limpia [0, 48 GHz]
y_sup_inv = x_e - x_o_aligned      # Subbanda superior limpia (invertida)

# ==============================================================================
# 4. REORDENAMIENTO ESPECTRAL DE LA SUBBANDA SUPERIOR
# ==============================================================================
k = np.arange(M)
y_sup = y_sup_inv * ((-1.0) ** k)  # Espeja [0, 48 GHz] -> deja y_sup ordenada

# ==============================================================================
# 5. RECONSTRUCCIÓN A 384 GHz Y MODULACIÓN SSB
# ==============================================================================
fs_plot = 384e9
N_plot = int(round(M * (fs_plot / fs_sub)))
y_inf_plot = resample(y_inf, N_plot)
y_sup_plot = resample(y_sup, N_plot)
t_plot = np.arange(N_plot) / fs_plot

hat_y_sup = np.imag(hilbert(y_sup_plot))
s2_ssb = y_sup_plot * np.cos(2 * np.pi * fc * t_plot) - hat_y_sup * np.sin(2 * np.pi * fc * t_plot)
s_total = y_inf_plot + s2_ssb

# ==============================================================================
# FUNCIÓN AUXILIAR PARA PSD SUAVIZADA EN dB
# ==============================================================================
def psd_db(sig, fs, ref_max, nperseg=1024):
    f, Pxx = welch(sig, fs=fs, nperseg=nperseg, return_onesided=True)
    return f / 1e9, 10 * np.log10(Pxx / ref_max + 1e-18)

_, Pxx_ref = welch(x_sampled, fs=fs_nyq, nperseg=1024, return_onesided=True)
ref_max = np.max(Pxx_ref)

# ==============================================================================
# GRÁFICOS PASO A PASO
# ==============================================================================
fig, axes = plt.subplots(5, 1, figsize=(11, 15), constrained_layout=True)

# --- Panel 1: Señal original a 192 GHz ---
f, P = psd_db(x_sampled, fs_nyq, ref_max)
axes[0].plot(f, P, color="#1f77b4", lw=1.4)
axes[0].axvline(48, color="k", ls="--", lw=1, label="Frontera fc = 48 GHz")
axes[0].annotate("Piloto L: 15 GHz", xy=(15, 0), xytext=(18, -8),
                 arrowprops=dict(arrowstyle="->", color="#1f77b4"), fontweight="bold")
axes[0].annotate("Piloto U: 75 GHz", xy=(75, -7.5), xytext=(60, -15),
                 arrowprops=dict(arrowstyle="->", color="#d62728"), fontweight="bold", color="#d62728")
axes[0].set_title("1. Señal original x_sampled (fs = 192 GHz) — Dos subbandas contiguas [0–48] y [48–96] GHz")
axes[0].set_xlim(0, 96)
axes[0].legend(loc="upper right")

# --- Panel 2: Diezmado por 2 (Aliasing en x_e y x_o) ---
f_e, P_e = psd_db(x_e, fs_sub, ref_max, nperseg=512)
f_o, P_o = psd_db(x_o, fs_sub, ref_max, nperseg=512)
axes[1].plot(f_e, P_e, color="#1f77b4", lw=1.5, label="|X_e(f)| (muestras pares)")
axes[1].plot(f_o, P_o, color="#ff7f0e", ls="--", lw=1.3, label="|X_o(f)| (muestras impares)")
axes[1].annotate("Piloto L: 15 GHz", xy=(15, 0), xytext=(5, -10),
                 arrowprops=dict(arrowstyle="->", color="#1f77b4"), fontweight="bold")
axes[1].annotate("Alias U: 96 - 75 = 21 GHz", xy=(21, -7.5), xytext=(25, -14),
                 arrowprops=dict(arrowstyle="->", color="#d62728"), fontweight="bold", color="#d62728")
axes[1].set_title("2. Tras diezmar por 2 (fs = 96 GHz) — Aliasing: [48–96 GHz] se pliega invertida sobre [0–48 GHz]")
axes[1].set_xlim(0, 48)
axes[1].legend(loc="upper right")

# --- Panel 3: Separación por Suma y Resta ---
f_i, P_i = psd_db(y_inf, fs_sub, ref_max, nperseg=512)
f_si, P_si = psd_db(y_sup_inv, fs_sub, ref_max, nperseg=512)
axes[2].plot(f_i, P_i, color="#2ca02c", lw=1.5, label="y_inf = x_e + x_o_aligned (Subbanda Inferior)")
axes[2].plot(f_si, P_si, color="#d62728", lw=1.5, label="y_sup_inv = x_e - x_o_aligned (Subbanda Superior invertida)")
axes[2].annotate("Solo L (15 GHz)\n(21 GHz cancelado)", xy=(15, 6), xytext=(4, -18),
                 arrowprops=dict(arrowstyle="->", color="#2ca02c"), fontweight="bold", color="#2ca02c")
axes[2].annotate("Solo U_inv (21 GHz)\n(15 GHz cancelado)", xy=(21, -1.5), xytext=(26, -18),
                 arrowprops=dict(arrowstyle="->", color="#d62728"), fontweight="bold", color="#d62728")
axes[2].set_title("3. Separación sin aliasing tras alinear x_o (-0.5 muestras) y sumar/restar")
axes[2].set_xlim(0, 48)
axes[2].legend(loc="upper right")

# --- Panel 4: Inversión espectral de y_sup ---
f_s, P_s = psd_db(y_sup, fs_sub, ref_max, nperseg=512)
axes[3].plot(f_si, P_si, color="#d62728", alpha=0.35, ls="--", lw=1.2, label="Antes: y_sup_inv (invertida)")
axes[3].plot(f_s, P_s, color="#9467bd", lw=1.6, label="Después: y_sup = y_sup_inv · (-1)^k (ordenada)")
axes[3].annotate("Piloto U reordenado:\n48 - 21 = 27 GHz (= 75 - 48)", xy=(27, -1.5), xytext=(29, -18),
                 arrowprops=dict(arrowstyle="->", color="#9467bd"), fontweight="bold", color="#9467bd")
axes[3].set_title("4. Reordenamiento espectral de y_sup (f -> 48 - f) para dejarla lista para SSB")
axes[3].set_xlim(0, 48)
axes[3].legend(loc="upper right")

# --- Panel 5: Reconstrucción total a 384 GHz ---
f_tot, P_tot = psd_db(s_total, fs_plot, ref_max, nperseg=2048)
axes[4].plot(f_tot, P_tot, color="#17becf", lw=1.5, label="s_total_FI (y_inf + s2_ssb)")
axes[4].axvline(48, color="k", ls="--", lw=1)
axes[4].annotate("Piloto L: 15 GHz", xy=(15, 6), xytext=(20, -10),
                 arrowprops=dict(arrowstyle="->", color="#1f77b4"), fontweight="bold")
axes[4].annotate("Piloto U trasladado:\n48 + 27 = 75 GHz", xy=(75, -1.5), xytext=(82, -16),
                 arrowprops=dict(arrowstyle="->", color="#9467bd"), fontweight="bold", color="#9467bd")
axes[4].set_title("5. Señal reconstruida a fs_plot = 384 GHz — Espectro original recuperado sin aliasing")
axes[4].set_xlim(0, 120)
axes[4].set_xlabel("Frecuencia [GHz]")
axes[4].legend(loc="upper right")

for ax in axes:
    ax.set_ylabel("PSD [dB]")
    ax.set_ylim(-65, 12)
    ax.grid(True, alpha=0.3)

plt.show()