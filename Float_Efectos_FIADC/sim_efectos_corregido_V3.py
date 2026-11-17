import numpy as np
import matplotlib.pyplot as plt
import random
import sys
from scipy import signal
from scipy.signal import hilbert, resample


# ==============================================================================
# 1. GENERACIÓN DE SEÑALES TRANSMITIDAS (Tx)
# ==============================================================================
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, resample

# ==============================================================================
# 1. GENERACIÓN DE LA SEÑAL ANALÓGICA CONTINUA
# ==============================================================================
M = 1000
simb_pam_total = np.random.choice([-3, -1, 1, 3], size=2 * M)

span = 6
alpha = 0.1
sps = 64  # Rejilla analógica continua

simb_upsampled = np.zeros(len(simb_pam_total) * sps)
simb_upsampled[::sps] = simb_pam_total

sys.path.append("C:/Users/Usuario/Desktop/Valentino/PROCOM/TP1")
from filtro import RaisedCosineFilter # type: ignore

rc = RaisedCosineFilter(alpha=alpha, span=span, sps=sps, rrc=False)
rc_filter = rc.get_coefficients()
senial = np.convolve(simb_upsampled, rc_filter, mode='full')       

BW_deseado = 96e9
fc = BW_deseado / 2  # 48 GHz
fs_nyq = 4 * fc      # 192 GHz (para desfase de 90° en Hilbert)

Rs = (2 * BW_deseado) / (1 + alpha)
fs = Rs * sps        # ~11.17 THz

# paso = int(round(fs / fs_nyq))  # paso ≈ 58
# x_sampled = senial[::paso]

N_nyq = int(round(len(senial) * (fs_nyq / fs)))
N_nyq -= (N_nyq % 4)  # Garantiza que x_e y x_o tengan la misma longitud par
x_sampled = resample(senial, N_nyq)

# Muestras pares e impares
x_e = x_sampled[0::2]
x_o = x_sampled[1::2]

min_len = min(len(x_e), len(x_o))
x_e = x_e[:min_len]
x_o = x_o[:min_len]

f_norm = np.fft.rfftfreq(min_len)
x_o_aligned = np.fft.irfft(np.fft.rfft(x_o) * np.exp(-1j * np.pi * f_norm), n=min_len)

# 2. Separación de subbandas
y_inf = 0.5 * (x_e + x_o_aligned)          # [0, 48 GHz] ordenada
y_sup_inv = 0.5 * (x_e - x_o_aligned)      # [48, 96 GHz] en banda base, pero invertida

# 3. Compensación de la inversión espectral de U
y_sup = y_sup_inv.copy()
y_sup[1::2] *= -1                  # [48, 96 GHz] en banda base y ordenada

# ==============================================================================
# 2. RECONSTRUCCIÓN A FS_PLOT = 384 GHz (EJE EXTENDIDO A ±192 GHz)
# ==============================================================================
fs_plot = 384e9  # Tasa ampliada para ver decaimientos hasta ±192 GHz
N_target = int(round(len(y_inf) * (fs_plot / (fs_nyq / 2))))

# Reconstrucción pasatodo sin distorsión
y_inf_plot = resample(y_inf, N_target)
y_sup_plot = resample(y_sup, N_target)

t_plot = np.arange(N_target) / fs_plot

# Modulación SSB de la subbanda superior hacia [48 GHz - 96 GHz]
hat_y_sup_plot = np.imag(hilbert(y_sup_plot))
s2_ssb_plot = y_sup_plot * np.cos(2 * np.pi * fc * t_plot) - hat_y_sup_plot * np.sin(2 * np.pi * fc * t_plot)

# Señal transmitida total en el canal
s_total_FI_plot = y_inf_plot + s2_ssb_plot

# Ejes frecuenciales
freqs_plot = np.fft.fftshift(np.fft.fftfreq(N_target, d=1/fs_plot)) / 1e9
freqs_analog = np.fft.fftshift(np.fft.fftfreq(len(senial), d=1/fs)) / 1e9

# Espectros normalizados
fft_senial = np.fft.fftshift(np.fft.fft(senial))
mag_senial = 20 * np.log10(np.abs(fft_senial) / np.max(np.abs(fft_senial)))

fft_y_inf = np.fft.fftshift(np.fft.fft(y_inf_plot))
mag_y_inf = 20 * np.log10(np.abs(fft_y_inf) / np.max(np.abs(fft_y_inf)))

fft_y_sup = np.fft.fftshift(np.fft.fft(y_sup_plot))
mag_y_sup = 20 * np.log10(np.abs(fft_y_sup) / np.max(np.abs(fft_y_sup)))

fft_s2_ssb = np.fft.fftshift(np.fft.fft(s2_ssb_plot))
mag_s2_ssb = 20 * np.log10(np.abs(fft_s2_ssb) / np.max(np.abs(fft_y_inf)))

fft_stotal = np.fft.fftshift(np.fft.fft(s_total_FI_plot))
mag_stotal = 20 * np.log10(np.abs(fft_stotal) / np.max(np.abs(fft_y_inf)))

# ==============================================================================
# 3. FIGURA 1: SEÑAL CONTINUA Y SUBBANDAS EN BANDA BASE (EXTENDIDAS A ±160 GHz)
# ==============================================================================
plt.figure(figsize=(12, 7))

plt.subplot(2, 2, (1, 2))
plt.plot(freqs_analog, mag_senial, color='m')
plt.xlabel('Frecuencia [GHz]')
plt.ylabel('Magnitud [dB]')
plt.xlim(-160, 160)
plt.ylim(-60, 5)
plt.title('Señal PAM-4 Total Continua Original (Decaimiento > 96 GHz)')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(freqs_plot, mag_y_inf, color='b')
plt.xlabel('Frecuencia [GHz]')
plt.ylabel('Magnitud [dB]')
plt.xlim(-160, 160)
plt.ylim(-60, 5)
plt.title('Subbanda Inferior $y_{inf}$ en Banda Base (0 a 48 GHz)')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(freqs_plot, mag_y_sup, color='r')
plt.xlabel('Frecuencia [GHz]')
plt.ylabel('Magnitud [dB]')
plt.xlim(-160, 160)
plt.ylim(-60, 5)
plt.title('Subbanda Superior $y_{sup}$ en Banda Base (0 a 48 GHz)')
plt.grid(True)
plt.tight_layout()

# ==============================================================================
# 4. FIGURA 2: MODULACIÓN Y CANAL TRANSMITIDO (EXTENDIDAS A ±160 GHz)
# ==============================================================================
plt.figure(figsize=(12, 7))

plt.subplot(2, 2, 1)
plt.plot(freqs_plot, mag_y_inf, color='b')
plt.xlabel('Frecuencia [GHz]')
plt.ylabel('Magnitud [dB]')
plt.xlim(-160, 160)
plt.ylim(-60, 5)
plt.title('Subbanda Inferior en el Canal (0 a 48 GHz)')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(freqs_plot, mag_s2_ssb, color='r')
plt.xlabel('Frecuencia [GHz]')
plt.ylabel('Magnitud [dB]')
plt.xlim(-160, 160)
plt.ylim(-60, 5)
plt.title('Subbanda Superior Modulada SSB (Decaimientos < 48 GHz y > 96 GHz)')
plt.grid(True)

plt.subplot(2, 2, (3, 4))
plt.plot(freqs_plot, mag_stotal, color='m')
plt.xlabel('Frecuencia [GHz]')
plt.ylabel('Magnitud [dB]')
plt.xlim(-160, 160)
plt.ylim(-60, 5)
plt.title('Señal PAM-4 Total Transmitida $s_{total\_FI}$ (Decaimiento suave > 96 GHz)')
plt.grid(True)
plt.tight_layout()

# ==============================================================================
# 5. FIGURA ADICIONAL: ESPECTROS DE LAS MUESTRAS PARES (x_e) E IMPARES (x_o)
# ==============================================================================
# Reconstrucción a fs_plot (384 GHz) para observar las colas de atenuación
x_e_plot = resample(x_e, N_target)
x_o_plot = resample(x_o, N_target)

# FFT de las ramas par e impar
fft_xe = np.fft.fftshift(np.fft.fft(x_e_plot))
mag_xe = 20 * np.log10(np.abs(fft_xe) / np.max(np.abs(fft_xe)))

fft_xo = np.fft.fftshift(np.fft.fft(x_o_plot))
mag_xo = 20 * np.log10(np.abs(fft_xo) / np.max(np.abs(fft_xo)))

plt.figure(figsize=(12, 6))

# Espectro Muestras Pares (x_e)
plt.subplot(1, 2, 1)
plt.plot(2*freqs_plot, mag_xe, color='c')
plt.xlabel('Frecuencia [GHz]')
plt.ylabel('Magnitud [dB]')
plt.xlim(-160, 160)
plt.ylim(-60, 5)
plt.title('Espectro Muestras Pares $x_e$ (Solapamiento / Aliasing de Subbandas)')
plt.grid(True)

# Espectro Muestras Impares (x_o)
plt.subplot(1, 2, 2)
plt.plot(2*freqs_plot, mag_xo, color='orange')
plt.xlabel('Frecuencia [GHz]')
plt.ylabel('Magnitud [dB]')
plt.xlim(-160, 160)
plt.ylim(-60, 5)
plt.title('Espectro Muestras Impares $x_o$ (Solapamiento / Aliasing de Subbandas)')
plt.grid(True)

plt.tight_layout()


# ==============================================================================
# 6. FIGURA ADICIONAL: DOMINIO DEL TIEMPO Y DIAGRAMA DE OJO (Tx)
# ==============================================================================
# Tasa de muestreo exacta de x_sampled (generada con resample(senial, N_nyq))
fs_nyq_real = fs * (len(x_sampled) / len(senial))
fs_plot_real = fs * (len(s_total_FI_plot) / len(senial))

# Normalización de amplitud (divide por 2.0 si sumaste x_e ± x_o_aligned sin 0.5)
s_tx_384 = s_total_FI_plot

# Interpolación suave de la señal reconstruida a la grilla continua de `senial`
# para evitar los tramos rectos poligonales de 384 GHz al graficar
s_tx_continua = resample(s_tx_384, len(senial))

t_analog_ns = (np.arange(len(senial)) / fs) * 1e9
t_sampled_ns = (np.arange(len(x_sampled)) / fs_nyq_real) * 1e9

plt.figure(figsize=(13, 5))

# --- Panel Izquierdo: Comparación en el dominio del tiempo ---
plt.subplot(1, 2, 1)
t_ini_ns = 1.5
t_fin_ns = 1.5 + 15 * (1e9 / Rs)

mask_an = (t_analog_ns >= t_ini_ns) & (t_analog_ns <= t_fin_ns)
mask_sa = (t_sampled_ns >= t_ini_ns) & (t_sampled_ns <= t_fin_ns)

plt.plot(t_analog_ns[mask_an], senial[mask_an], color='m', lw=1.8, alpha=0.65, label='Señal continua original')
plt.plot(t_analog_ns[mask_an], s_tx_continua[mask_an], 'b--', lw=1.5, label='$s_{total\\_FI}$ reconstruida (suavizada)')
plt.xlabel('Tiempo [ns]')
plt.ylabel('Amplitud [V]')
plt.title('Señal PAM-4 en el Canal (Dominio del Tiempo)')
plt.grid(True)
plt.legend(loc='upper right')

# --- Panel Derecho: Diagrama de Ojo PAM-4 ---
plt.subplot(1, 2, 2)
sps_eye = 16
N_eye = int(round(len(s_tx_384) * (Rs * sps_eye / fs_plot_real)))
s_eye = resample(s_tx_384, N_eye)

trim = span * sps_eye
s_eye_trim = s_eye[trim:-trim]

num_trazas = min(400, len(s_eye_trim) // (2 * sps_eye))
trazas = s_eye_trim[:num_trazas * 2 * sps_eye].reshape((num_trazas, 2 * sps_eye))
t_simb = np.linspace(0, 2, 2 * sps_eye, endpoint=False)

plt.plot(t_simb, trazas.T, color='b', alpha=0.12, lw=0.9)
for nivel in [-3, -1, 1, 3]:
    plt.axhline(nivel, color='r', ls=':', lw=1)

plt.xlabel('Tiempo normalizado [$t / T_{símbolo}$]')
plt.ylabel('Amplitud [V]')
plt.title('Diagrama de Ojo PAM-4 Transmitido ($s_{total\\_FI}$)')
plt.xlim(0, 2)
plt.grid(True)

plt.tight_layout()

# plt.show()

#Simulo efectos del canal
from simuladorADC import simuladorADC

y_inf_rx, y_sup_rx = simuladorADC(s_total_FI_plot, fs_plot, rolloff = 0.1, span = 6 , sps = 8)
# y_inf_rx, y_sup_rx = simuladorADC(s_total_FI_plot, fs_plot, rolloff = 0.1, span = 6 , sps = 8, fugaOL=30, fugaIO=35, epsilon=0.01, phi_deg=9, square_mixer=True, jitter_OL=8e-4, jitter_adc=300e-15)

# Re-inversión espectral de la subbanda superior
y_sup_inv_rx = y_sup_rx.copy()
y_sup_inv_rx[1::2] *= -1

# Desmezcla algebraica (suma y resta)
x_e_rx = y_inf_rx + y_sup_inv_rx
x_o_aligned_rx = y_inf_rx - y_sup_inv_rx

# Adelanto fraccionario de +0.5 muestras en x_o (signo + en la exponencial)
N_sub = len(x_e_rx)
f_norm_rx = np.fft.rfftfreq(N_sub)
x_o_rx = np.fft.irfft(np.fft.rfft(x_o_aligned_rx) * np.exp(+1j * np.pi * f_norm_rx), n=N_sub)

# Intercalado de muestras pares e impares -> señal reconstruida a 192 GHz
x_rec_192 = np.empty(2 * N_sub)
x_rec_192[0::2] = x_e_rx
x_rec_192[1::2] = x_o_rx

# Normalización de la señal reconstruida para la comparativa temporal
x_rec_norm = x_rec_192 * (np.std(x_sampled) / np.std(x_rec_192))

# Cálculo de la duración real y cantidad exacta de símbolos para evitar desincronización
T_rx = len(x_rec_norm) / fs_nyq 
num_symbols_rx = T_rx * Rs
sps_eye = 16

# Remuestreo a sps_eye basado en la cantidad real de símbolos
N_eye = int(round(num_symbols_rx * sps_eye))
s_eye_rx = resample(x_rec_norm, N_eye)

# Recorte de transitorios de los filtros (retardo de grupo)
trim_rx = int(span * sps_eye)

# Extracción de trazas sincronizadas (cada traza dura 2 símbolos)
n_trazas = min(400, int(num_symbols_rx) - span * 2)
trazas_rx = np.zeros((n_trazas, 2 * sps_eye))

for i in range(n_trazas):
    idx = trim_rx + i * sps_eye
    trazas_rx[i, :] = s_eye_rx[idx : idx + 2 * sps_eye]

# Normalización final a niveles PAM-4 teóricos (varianza = 5) para el diagrama de ojo
varianza_central = np.var(trazas_rx[:, 0])
trazas_rx = trazas_rx * np.sqrt(5.0 / varianza_central)

# ==============================================================================
# 2. GRÁFICAS TEMPORALES Y DIAGRAMA DE OJO (Figura 1)
# ==============================================================================
plt.figure(figsize=(14, 5))

# Subplot 1,2,1: Señales temporales Tx vs Rx
plt.subplot(1, 2, 1)
inicio, fin = 200, 300  # Segmento visible
plt.plot(x_sampled[inicio:fin], 'k-', lw=2, alpha=0.7, label='Tx ($x_{sampled}$)')
plt.plot(x_rec_norm[inicio:fin], 'r--', lw=1.5, label='Rx ($x_{rec}$ normalizada)')
plt.title('Comparación Temporal Tx vs Rx (192 GSa/s)')
plt.xlabel('Índice de muestra')
plt.ylabel('Amplitud [V]')
plt.grid(True, alpha=0.3)
plt.legend()

# Subplot 1,2,2: Diagrama de Ojo Recibido (Corregido)
plt.subplot(1, 2, 2)
t_eye = np.linspace(0, 2, 2 * sps_eye, endpoint=False)
plt.plot(t_eye, trazas_rx.T, color='b', alpha=0.15, lw=1)

for nivel in [-3, -1, 1, 3]:
    plt.axhline(nivel, color='r', ls=':', lw=1.5)
    
plt.title('Diagrama de Ojo Recibido Sincronizado')
plt.xlabel('Tiempo normalizado [$t / T_{símbolo}$]')
plt.ylabel('Amplitud normalizada')
plt.xlim(0, 2)
plt.grid(True, alpha=0.3)

plt.tight_layout()
# plt.show()

# ==============================================================================
# 3. GRÁFICAS ESPECTRALES (Figura 2)
# ==============================================================================
fs_sub = 96e9
fs_nyq = 192e9

# Cálculo de FFTs normalizadas al máximo (0 dB)
f_sub = np.fft.fftshift(np.fft.fftfreq(len(y_inf_rx), d=1/fs_sub)) / 1e9
Y_inf_mag = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(y_inf_rx))) / np.max(np.abs(np.fft.fft(y_inf_rx))) + 1e-12)
Y_sup_mag = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(y_sup_rx))) / np.max(np.abs(np.fft.fft(y_sup_rx))) + 1e-12)

f_rec = np.fft.fftshift(np.fft.fftfreq(len(x_rec_192), d=1/fs_nyq)) / 1e9
X_rec_mag = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(x_rec_192))) / np.max(np.abs(np.fft.fft(x_rec_192))) + 1e-12)

plt.figure(figsize=(14, 5))

# Subbandas recibidas
plt.subplot(1, 2, 1)
plt.plot(f_sub, Y_inf_mag, 'b-', label='$y_{inf}$ (Rx)', alpha=0.8)
plt.plot(f_sub, Y_sup_mag, 'r-', label='$y_{sup}$ (Rx)', alpha=0.8)
plt.title('Espectro de Subbandas (96 GSa/s)')
plt.xlabel('Frecuencia [GHz]')
plt.ylabel('Magnitud Normalizada [dB]')
plt.ylim(-60, 5)
plt.grid(True, alpha=0.3)
plt.legend()

# Espectro total reconstruido
plt.subplot(1, 2, 2)
plt.plot(f_rec, X_rec_mag, 'g-')
plt.title('Espectro Señal Reconstruida (192 GSa/s)')
plt.xlabel('Frecuencia [GHz]')
plt.ylabel('Magnitud Normalizada [dB]')
plt.ylim(-60, 5)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()