import numpy as np
import matplotlib.pyplot as plt
import random
import sys
from scipy import signal
from scipy.signal import hilbert, resample

def aplicar_lpf_butterworth(senial, fs, fc, orden, causal=False):
    sos = signal.butter(N=orden, Wn=fc, btype='lowpass', analog=False, fs=fs, output='sos')
    if causal:
        senial_filtrada = signal.sosfilt(sos, senial)
    else:
        senial_filtrada = signal.sosfiltfilt(sos, senial)
    return senial_filtrada, sos

def aplicar_hpf_butterworth(senial, fs, fc, orden, causal=False):
    sos = signal.butter(N=orden, Wn=fc, btype='highpass', analog=False, fs=fs, output='sos')
    if causal:
        senial_filtrada = signal.sosfilt(sos, senial)
    else:
        senial_filtrada = signal.sosfiltfilt(sos, senial)
    return senial_filtrada, sos

def graficar_diagrama_ojo(senial, sps, n_simbolos_pantalla=3, n_trazas=500, nueva_figura=True):
    muestras_por_traza = sps * n_simbolos_pantalla
    n_trazas_posibles = len(senial) // muestras_por_traza
    n_trazas = min(n_trazas, n_trazas_posibles)
    
    if nueva_figura:
        plt.figure(figsize=(8, 4))
    t_eje = np.arange(muestras_por_traza) / sps
    
    for i in range(n_trazas):
        idx_inicio = i * muestras_por_traza
        segmento = senial[idx_inicio : idx_inicio + muestras_por_traza]
        plt.plot(t_eje, segmento, color='blue', alpha=0.03, linewidth=3)

def graficar_par_ojos(sig_bb, sig_hb, sps, titulo_bb, titulo_hb):
    plt.figure(figsize=(10, 7))
    plt.subplot(2, 1, 1)
    graficar_diagrama_ojo(sig_bb, sps=sps, nueva_figura=False)
    plt.title(titulo_bb)
    plt.xlabel("Tiempo [Símbolos $T_s$]")
    plt.ylabel("Amplitud")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    graficar_diagrama_ojo(sig_hb, sps=sps, nueva_figura=False)
    plt.title(titulo_hb)
    plt.xlabel("Tiempo [Símbolos $T_s$]")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.tight_layout()

def graficar_par_constelaciones(tx_bb, rx_bb, tx_hb, rx_hb, titulo_bb, titulo_hb):
    plt.figure(figsize=(10, 7))
    plt.subplot(2, 1, 1)
    plt.scatter(np.real(tx_bb[::sps]), np.imag(tx_bb[::sps]), alpha=0.3, color='blue', marker='x', label='Tx Ideal')
    plt.scatter(np.real(rx_bb[delay::sps]), np.imag(rx_bb[delay::sps]), alpha=0.3, color='m', label='Rx Muestreada')
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.title(titulo_bb)
    plt.xlabel('En Fase (I)')
    plt.ylabel('Cuadratura (Q)')
    plt.ylim(-0.5, 0.5)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')

    plt.subplot(2, 1, 2)
    plt.scatter(np.real(tx_hb[::sps]), np.imag(tx_hb[::sps]), alpha=0.3, color='blue', marker='x', label='Tx Ideal')
    plt.scatter(np.real(rx_hb[delay::sps]), np.imag(rx_hb[delay::sps]), alpha=0.3, color='m', label='Rx Muestreada')
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.title(titulo_hb)
    plt.xlabel('En Fase (I)')
    plt.ylabel('Cuadratura (Q)')
    plt.ylim(-0.5, 0.5)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()


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

paso = int(round(fs / fs_nyq))  # paso ≈ 58
x_sampled = senial[::paso]

# Muestras pares e impares
x_e = x_sampled[0::2]
x_o = x_sampled[1::2]

min_len = min(len(x_e), len(x_o))
x_e = x_e[:min_len]
x_o = x_o[:min_len]

hat_x_o = np.imag(hilbert(x_o))

# Subbandas en Banda Base (0 a 48 GHz)
y_inf = x_e + hat_x_o
y_sup = x_e - hat_x_o

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
plt.plot(freqs_plot, mag_xe, color='c')
plt.xlabel('Frecuencia [GHz]')
plt.ylabel('Magnitud [dB]')
plt.xlim(-160, 160)
plt.ylim(-60, 5)
plt.title('Espectro Muestras Pares $x_e$ (Solapamiento / Aliasing de Subbandas)')
plt.grid(True)

# Espectro Muestras Impares (x_o)
plt.subplot(1, 2, 2)
plt.plot(freqs_plot, mag_xo, color='orange')
plt.xlabel('Frecuencia [GHz]')
plt.ylabel('Magnitud [dB]')
plt.xlim(-160, 160)
plt.ylim(-60, 5)
plt.title('Espectro Muestras Impares $x_o$ (Solapamiento / Aliasing de Subbandas)')
plt.grid(True)

plt.tight_layout()

plt.show()

# # Separación en muestras pares (Banda Inferior) e impares (Banda Superior)
# simb_pam = senial[0::2]
# simb_pam_hf = senial[1::2]

# # print(f"Primeros símbolos PAM-4 Total:  {simb_pam_total[0:20]}")
# # print(f"Primeros símbolos BB (Pares):   {simb_pam[0:10]}")
# # print(f"Primeros símbolos HB (Impares): {simb_pam_hf[0:10]}")

# # delay = (span * sps) // 2
# # senial_alineada = senial[delay : delay + len(simb_upsampled)]

# BW_deseado = 24e9
# Rs = (2 * BW_deseado) / (1 + alpha)
# fs = Rs * sps

# fc = 46e9
# t = np.arange(len(simb_pam_hf)) / fs
# s2_hilbert = hilbert(simb_pam_hf)
# s2_ssb = np.real(s2_hilbert * np.exp(1j * 2 * np.pi * fc * t)) # Banda superior modulada en SSB

# s_total_FI = simb_pam + s2_ssb

# # Señal PAM-4 total de referencia (0 a 96 GHz continuos, doble tasa de símbolos)
# sps_total = sps // 2  # 32 muestras por símbolo
# simb_upsampled_total = np.zeros(len(simb_pam_total) * sps_total)
# simb_upsampled_total[::sps_total] = simb_pam_total

# rc_total = RaisedCosineFilter(alpha=alpha, span=span, sps=sps_total, rrc=False)
# rc_filter_total = rc_total.get_coefficients()
# senial_total_tx = np.convolve(simb_upsampled_total, rc_filter_total, mode='full')

# delay_total = (span * sps_total) // 2
# senial_total_tx_alineada = senial_total_tx[delay_total : delay_total + len(simb_upsampled_total)]


# ==============================================================================
# 2. MODELADO DE CADA EFECTO AISLADO Y SUMA TOTAL AL FINAL
# ==============================================================================
# BW_tia = 100e9
# B_d = 48e9
# LO_Port = 10**(-30/20)
# RF_IF = 10**(-35/20)
# epsilon = 0.01
# phi_rad = np.deg2rad(9)
# f_lo = 48e9

# # Osciladores Locales
# LO_ideal = 2 * np.cos(2 * np.pi * f_lo * t)
# LO_estatico = (2 + epsilon/2) * np.sign(np.cos(2 * np.pi * f_lo * t + phi_rad/2))

# # Comportamiento ideal para respuesta temporal
# # Sin señal vecina (HPF perfecto)
# s2_ideal = s2_ssb * LO_ideal
# s2_ideal_filt, _ = aplicar_lpf_butterworth(s2_ideal, fs, fc=48e9, orden=3)
# # Con señal vecina 
# s2_filt, _ = aplicar_hpf_butterworth(s_total_FI, fs, fc=48e9, orden=3)
# s2_ideal_FI = s2_filt * LO_ideal
# s2_ideal_filt_FI, _ = aplicar_lpf_butterworth(s2_ideal_FI, fs, fc=48e9, orden=3)

# sigma_rw = 2 * np.pi * 8e-4
# ruido_fase_lo = np.cumsum(np.random.normal(0, sigma_rw, size=len(t)))
# LO_solo_jitter = 2 * np.cos(2 * np.pi * f_lo * t + ruido_fase_lo)
# LO_completo = (2 + epsilon/2) * np.sign(np.cos(2 * np.pi * f_lo * t + phi_rad/2 + ruido_fase_lo))

# # Jitter de Apertura (Vector de tiempo para el ADC)
# sigma_t_adc = 300e-15
# t_ideal = np.arange(len(t)) / fs
# t_jitter_adc = t_ideal + np.random.normal(0, sigma_t_adc, size=len(t))

# # --- Etapa 1: TIA ---
# senial_tia, _ = aplicar_lpf_butterworth(s_total_FI, fs, BW_tia, 3)        # Compuesta
# senial_tia_bb_aislada, _ = aplicar_lpf_butterworth(senial, fs, BW_tia, 3) # BB aislada
# senial_tia_hf_aislada, _ = aplicar_lpf_butterworth(s2_ssb, fs, BW_tia, 3) # HB aislada

# # --- Etapa 2: Separación de Bandas (LPF externo en BB y HPF externo en HB) ---
# senial_lpf_bb_con_solap, _ = aplicar_lpf_butterworth(senial_tia, fs, fc=48e9, orden=4)
# senial_hpf_con_solap, _ = aplicar_hpf_butterworth(senial_tia, fs, fc=48e9, orden=4)

# # --- Etapa 3: Mixer (Aislados vs Completo) ---
# RF_aislada, _ = aplicar_lpf_butterworth(senial_tia_hf_aislada, fs, 100e9, 1)
# RF_solap, _ = aplicar_lpf_butterworth(senial_hpf_con_solap, fs, 100e9, 1)

# # A) Solo Imperfecciones del Mixer (Fugas + Error estático LO, sobre señal aislada)
# post_mix_estatico = (RF_aislada + LO_Port * LO_estatico) * LO_estatico + RF_IF * RF_aislada + LO_Port * LO_estatico
# IF_hf_solo_mixer, _ = aplicar_lpf_butterworth(post_mix_estatico, fs, 100e9, 1)

# # B) Solo Solapamiento (HPF sobre señal compuesta + Mixer ideal sin fugas ni jitter)
# IF_hf_solo_solap, _ = aplicar_lpf_butterworth(RF_solap * LO_ideal, fs, 100e9, 1)

# # C) Solo Jitter LO (Señal aislada + Mixer ideal con LO con jitter)
# IF_hf_solo_jit_lo, _ = aplicar_lpf_butterworth(RF_aislada * LO_solo_jitter, fs, 100e9, 1)

# # D) Cadena Ideal (Para aislar el efecto del T&H y del Jitter de ADC)
# IF_hf_ideal, _ = aplicar_lpf_butterworth(RF_aislada * LO_ideal, fs, 100e9, 1)

# # E) SUMA TOTAL DE EFECTOS EN MIXER (Solapamiento + Fugas + Error estático + Jitter LO)
# post_mix_total = (RF_solap + LO_Port * LO_completo) * LO_completo + RF_IF * RF_solap + LO_Port * LO_completo
# IF_hf_total, _ = aplicar_lpf_butterworth(post_mix_total, fs, 100e9, 1)

# # --- Etapa 4: T&H y Muestreo ADC ---
# # 1. Solo T&H (Sin solapamiento, sin fugas de mixer, sin jitter)
# out_bb_solo_th, _ = aplicar_lpf_butterworth(senial_tia_bb_aislada, fs, B_d, 4)
# out_hf_solo_th, _ = aplicar_lpf_butterworth(IF_hf_ideal, fs, B_d, 4)

# # 2. Solo Imperfecciones de Mixer (En HB; BB no pasa por el mixer)
# out_hf_solo_mixer, _ = aplicar_lpf_butterworth(IF_hf_solo_mixer, fs, B_d, 4)

# # 3. Solo Solapamiento entre Bandas (Crosstalk mutuo: BB tras LPF externo y HB tras HPF externo)
# out_bb_solo_solap, _ = aplicar_lpf_butterworth(senial_lpf_bb_con_solap, fs, B_d, 4)
# out_hf_solo_solap, _ = aplicar_lpf_butterworth(IF_hf_solo_solap, fs, B_d, 4)

# # 4. Solo Jitter (En BB: Aperture Jitter ADC; En HB: Jitter LO + Aperture Jitter ADC)
# out_bb_solo_jitter = np.interp(t_jitter_adc, t_ideal, out_bb_solo_th)
# out_hf_pre_jit, _ = aplicar_lpf_butterworth(IF_hf_solo_jit_lo, fs, B_d, 4)
# out_hf_solo_jitter = np.interp(t_jitter_adc, t_ideal, out_hf_pre_jit)

# # 5. SUMA TOTAL DE TODOS LOS EFECTOS (T&H + Solapamiento + Mixer + Jitter)
# out_bb_total = np.interp(t_jitter_adc, t_ideal, out_bb_solo_solap)
# out_hf_pre_total, _ = aplicar_lpf_butterworth(IF_hf_total, fs, B_d, 4)
# out_hf_total = np.interp(t_jitter_adc, t_ideal, out_hf_pre_total)

# --- Etapa 5: Reconstrucción de la Señal PAM-4 Total (Rx) ---
# muestras_rx_bb = out_bb_total[delay : delay + len(simb_upsampled) : sps]
# muestras_rx_hb = out_hf_total[delay : delay + len(simb_upsampled_hf) : sps]

# simb_pam_rx_total = np.zeros(len(simb_pam_total))
# simb_pam_rx_total[0::2] = muestras_rx_bb
# simb_pam_rx_total[1::2] = muestras_rx_hb

# s2_hilbert = hilbert(out_hf_total)
# s2_rx = np.real(s2_hilbert * np.exp(1j * 2 * np.pi * fc * t)) # Banda superior modulada en SSB

# s_total_rx = out_bb_total + s2_rx
# senial_total_rx_alineada = s_total_rx[delay_total : delay_total + len(s_total_rx)]
# simb_pam_rx_total = senial_total_rx_alineada[::sps]
# Selector de método de reconstrucción para el tiempo y el ojo:
# USAR_FILTRO_RC = True

# if USAR_FILTRO_RC:
#     # OPCIÓN A: CON FILTRO RC
#     simb_rx_upsampled_total = np.zeros(len(simb_pam_rx_total) * sps_total)
#     simb_rx_upsampled_total[::sps_total] = simb_pam_rx_total

#     senial_total_rx = np.convolve(simb_rx_upsampled_total, rc_filter_total, mode='full')
#     senial_total_rx_alineada = senial_total_rx[delay_total : delay_total + len(simb_rx_upsampled_total)]
# else:
#     # OPCIÓN B: SIN FILTRO RC (Reconstrucción continua a 2 muestras/símbolo)
#     rx_bb_alin = out_bb_total[delay : delay + len(simb_upsampled)]
#     rx_hb_alin = out_hf_total[delay : delay + len(simb_upsampled_hf)]

#     osr_adc = 2  # 2 muestras por símbolo (1 en el pico + 1 en la transición)
#     paso_adc = sps // osr_adc

#     bb_adc = rx_bb_alin[::paso_adc]
#     hb_adc = rx_hb_alin[::paso_adc]

#     bb_bloques = bb_adc.reshape(M, osr_adc)
#     hb_bloques = hb_adc.reshape(M, osr_adc)

#     intercalado_adc = np.empty((2 * M, osr_adc))
#     intercalado_adc[0::2] = bb_bloques
#     intercalado_adc[1::2] = hb_bloques
#     muestras_adc_total = intercalado_adc.ravel()

#     senial_total_rx_alineada = resample(muestras_adc_total, len(simb_upsampled_total))



# # Espectro con Solapamiento vs Sin Solapamiento
# plt.figure(figsize=(10, 7))
# N = len(senial_tia_hf_aislada)
# freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/fs)) / 1e9
# esp_orig = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(senial_tia_hf_aislada))) / np.max(np.abs(np.fft.fft(senial_tia_hf_aislada))))
# esp_solap = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(senial_hpf_con_solap))) / np.max(np.abs(np.fft.fft(senial_hpf_con_solap))))
# plt.plot(freqs, esp_solap, color='r', alpha=0.8, label='Con Solapamiento (HPF 4º Orden)')
# plt.plot(freqs, esp_orig, color='b', linestyle='--', alpha=0.7, label='Sin Solapamiento (filtrado ideal)')
# plt.axvline(48, color='k', linestyle=':', label='Frecuencia de cruce (48 GHz)')
# plt.axvline(-48, color='k', linestyle=':')
# plt.xlabel('Frecuencia [GHz]')
# plt.ylabel('Magnitud Normalizada [dB]')
# plt.xlim(-110, 110)
# plt.ylim(-100, 5)
# plt.title('Espectro de entrada al Mixer: Efecto de Solapamiento de Banda Inferior')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()



# # ==============================================================================
# # 4. GRÁFICOS TEMPORALES EN SUBPLOTS (BANDA INFERIOR Y SUPERIOR)
# # ==============================================================================

# # 1. Señales Transmitidas (Tx): PAM-4 Original y División en Sub-bandas (Mismo intervalo temporal)
# n_simb_zoom = 20  # 20 símbolos totales (10 pares a BB y 10 impares a HB)
# muestras_zoom = n_simb_zoom * sps_total  # 640 muestras (mismo tiempo físico en las 3 gráficas)

# plt.figure(figsize=(10, 8))
# plt.subplot(3, 1, 1)
# plt.plot(senial_total_tx_alineada[:muestras_zoom], color='purple', alpha=0.5, label='Señal PAM-4 Total Conformada RC')
# plt.stem(simb_upsampled_total[:muestras_zoom], linefmt='purple', markerfmt='o', basefmt='r-', label='Símbolos PAM-4 Originales (Pares + Impares)')
# plt.xlabel('Muestras')
# plt.ylabel('Amplitud')
# plt.title('1. Señal PAM-4 Original Transmitida (Antes de dividir en sub-bandas)')
# plt.grid(True)
# plt.legend(loc='upper right')

# plt.subplot(3, 1, 2)
# plt.plot(senial_alineada[:muestras_zoom], color='b', alpha=0.5, label='Señal Conformada RC (BB)')
# plt.stem(simb_upsampled[:muestras_zoom], linefmt='b-', markerfmt='bo', basefmt='r-', label='Símbolos PAM-4 Pares (0, 2, 4...)')
# plt.xlabel('Muestras')
# plt.ylabel('Amplitud')
# plt.title('2. Señal Transmitida PAM-4: Banda Inferior (Símbolos Pares)')
# plt.grid(True)
# plt.legend(loc='upper right')

# plt.subplot(3, 1, 3)
# plt.plot(senial_alineada_hf[:muestras_zoom], color='g', alpha=0.5, label='Señal Conformada RC (HB)')
# plt.stem(simb_upsampled_hf[:muestras_zoom], linefmt='g-', markerfmt='go', basefmt='r-', label='Símbolos PAM-4 Impares (1, 3, 5...)')
# plt.xlabel('Muestras')
# plt.ylabel('Amplitud')
# plt.title('3. Señal Transmitida PAM-4: Banda Superior en Banda Base (Símbolos Impares)')
# plt.grid(True)
# plt.legend(loc='upper right')
# plt.tight_layout()
# plt.show()

# # 2. Efecto Aislado del TIA (Alineado en el mismo instante inicial delay)
# plt.figure(figsize=(10, 7))
# plt.subplot(2, 1, 1)
# plt.plot(senial[delay : delay + 2000], label="Entrada TIA (Banda Inferior)", alpha=0.6)
# plt.plot(senial_tia_bb_aislada[delay : delay + 2000], label="Salida TIA (LPF 3º Ord, 100 GHz)", linewidth=2)
# plt.xlabel('Muestras')
# plt.ylabel('Amplitud')
# plt.title('Efecto Aislado del TIA: Banda Inferior')
# plt.ylim(-6,6)
# plt.grid(True)
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(s2_ssb[delay : delay + 2000], label="Entrada TIA (Banda Superior SSB)", alpha=0.6)
# plt.plot(senial_tia_hf_aislada[delay : delay + 2000], label="Salida TIA (LPF 3º Ord, 100 GHz)", linewidth=2)
# plt.xlabel('Muestras')
# plt.ylabel('Amplitud')
# plt.title('Efecto Aislado del TIA: Banda Superior')
# plt.ylim(-6,6)
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# # 3. Efecto Aislado del Mixer (Exclusivo de Banda Superior)
# plt.figure(figsize=(10, 4))
# plt.plot(out_hf_solo_th[delay : delay + 2000], label="Banda Superior con Mixer Ideal (Sin fugas)", alpha=0.7)
# plt.plot(out_hf_solo_mixer[delay : delay + 2000], label="Banda Superior con Mixer Real (Fugas LO/RF + Error Fase/Ganancia)", linewidth=1.5)
# plt.xlabel('Muestras')
# plt.ylabel('Amplitud')
# plt.title('Efecto Aislado del Mezclador (Fugas y Error Estático LO): Banda Superior')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# # 4. Efecto Aislado del Track & Hold (T&H)
# plt.figure(figsize=(10, 7))
# plt.subplot(2, 1, 1)
# plt.plot(senial_tia_bb_aislada[delay : delay + 2000], label="Entrada T&H (Banda Inferior Aislada)", alpha=0.6)
# plt.plot(out_bb_solo_th[delay : delay + 2000], label="Salida T&H (LPF 4º Ord, 48 GHz)", linewidth=2)
# plt.xlabel('Muestras')
# plt.ylabel('Amplitud')
# plt.title('Efecto Aislado del T&H: Banda Inferior')
# plt.grid(True)
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(IF_hf_ideal[delay : delay + 2000], label="Entrada T&H (Banda Superior Demodulada Ideal)", alpha=0.6)
# plt.plot(out_hf_solo_th[delay : delay + 2000], label="Salida T&H (LPF 4º Ord, 48 GHz)", linewidth=2)
# plt.xlabel('Muestras')
# plt.ylabel('Amplitud')
# plt.title('Efecto Aislado del T&H: Banda Superior')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# # 5. Efecto Aislado del Solapamiento entre Bandas (Crosstalk)
# plt.figure(figsize=(10, 7))
# plt.subplot(2, 1, 1)
# plt.plot(out_bb_solo_th[delay : delay + 2000], label="Sin Solapamiento (Banda Inferior Aislada)", alpha=0.7)
# plt.plot(out_bb_solo_solap[delay : delay + 2000], label="Con Solapamiento (Fuga desde Banda Superior tras LPF 4º Ord)", linewidth=1.5)
# plt.xlabel('Muestras')
# plt.ylabel('Amplitud')
# plt.title('Efecto Aislado de Solapamiento (Crosstalk): Banda Inferior')
# plt.ylim(-4,4)
# plt.grid(True)
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(out_hf_solo_th[delay : delay + 2000], label="Sin Solapamiento (Banda Superior Aislada)", alpha=0.7)
# plt.plot(out_hf_solo_solap[delay : delay + 2000], label="Con Solapamiento (Fuga desde Banda Inferior tras HPF 4º Ord)", linewidth=1.5)
# plt.xlabel('Muestras')
# plt.ylabel('Amplitud')
# plt.title('Efecto Aislado de Solapamiento (Crosstalk): Banda Superior')
# plt.ylim(-4,4)
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# # 6. Efecto Aislado del Jitter
# plt.figure(figsize=(10, 7))
# plt.subplot(2, 1, 1)
# plt.plot(out_bb_solo_th[delay : delay + 2000], label="Sin Jitter (Banda Inferior)", alpha=0.7)
# plt.plot(out_bb_solo_jitter[delay : delay + 2000], label="Con Jitter de Apertura (ADC)", linewidth=1.5)
# plt.xlabel('Muestras')
# plt.ylabel('Amplitud')
# plt.title('Efecto Aislado de Jitter (Aperture Jitter ADC): Banda Inferior')
# plt.ylim(-4,4)
# plt.grid(True)
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(out_hf_solo_th[delay : delay + 2000], label="Sin Jitter (Banda Superior)", alpha=0.7)
# plt.plot(out_hf_solo_jitter[delay : delay + 2000], label="Con Jitter de Fase (LO) y Apertura (ADC)", linewidth=1.5)
# plt.xlabel('Muestras')
# plt.ylabel('Amplitud')
# plt.title('Efecto Aislado de Jitter (LO + ADC): Banda Superior')
# plt.grid(True)
# plt.legend()
# plt.ylim(-4,4)
# plt.tight_layout()
# plt.show()

# # 7. SUMA DE TODOS LOS EFECTOS (Comparación en el Tiempo por Sub-banda)
# plt.figure(figsize=(10, 7))
# plt.subplot(2, 1, 1)
# plt.plot(senial[delay : delay + 2000], color='r', alpha=0.5, label="Señal Tx Original (Banda Inferior)")
# plt.plot(out_bb_total[delay : delay + 2000], color='b', linewidth=1.5, label="Señal Rx a ecualizar (Banda Inferior)")
# plt.xlabel('Muestras')
# plt.ylabel('Amplitud')
# plt.title('Señal PAM-4 en el ecualizador considerando todos los efectos: Banda Inferior')
# plt.grid(True)
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(senial_hf[delay : delay + 2000], color='g', alpha=0.5, label="Señal Tx Original (Banda Superior)")
# plt.plot(out_hf_total[delay : delay + 2000], color='m', linewidth=1.5, label="Señal Rx a ecualizar (Banda Superior)")
# plt.xlabel('Muestras')
# plt.ylabel('Amplitud')
# plt.title('Señal PAM-4 en el ecualizador considerando todos los efectos: Banda Superior')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()


# # ==============================================================================
# # 5. DIAGRAMAS DE OJO EN SUBPLOTS (BANDA INFERIOR Y SUPERIOR)
# # ==============================================================================

# plt.figure(figsize=(10, 8))
# plt.subplot(2, 2, 1)
# graficar_diagrama_ojo(senial_alineada, sps=sps, nueva_figura=False)
# plt.title("Señal Tx Ideal (Banda Inferior)")
# plt.xlabel("Tiempo [Símbolos $T_s$]")
# plt.ylabel("Amplitud")
# plt.grid(True)

# plt.subplot(2, 2, 2)
# graficar_diagrama_ojo(senial_alineada_hf, sps=sps, nueva_figura=False)
# plt.title("Señal Tx Ideal (Banda Superior)")
# plt.xlabel("Tiempo [Símbolos $T_s$]")
# plt.ylabel("Amplitud")
# plt.grid(True)

# plt.subplot(2, 2, 3)
# graficar_diagrama_ojo(out_bb_total, sps=sps, nueva_figura=False)
# plt.title("Todos los Efectos (Banda Inferior)")
# plt.xlabel("Tiempo [Símbolos $T_s$]")
# plt.ylabel("Amplitud")
# plt.grid(True)

# plt.subplot(2, 2, 4)
# graficar_diagrama_ojo(out_hf_total, sps=sps, nueva_figura=False)
# plt.title("Todos los Efectos (Banda Superior)")
# plt.xlabel("Tiempo [Símbolos $T_s$]")
# plt.ylabel("Amplitud")
# plt.grid(True)
# plt.tight_layout()

# # Efecto Aislado: Solo T&H
# graficar_par_ojos(
#     out_bb_solo_th, out_hf_solo_th, sps=sps,
#     titulo_bb="Diagrama de Ojo: Efecto Aislado de T&H (Banda Inferior)",
#     titulo_hb="Diagrama de Ojo: Efecto Aislado de T&H (Banda Superior)"
# )

# # Efecto Aislado: Solo Mixer (Banda Superior)
# graficar_diagrama_ojo(out_hf_solo_mixer, sps=sps, nueva_figura=True)
# plt.title("Diagrama de Ojo: Efecto Aislado del Mixer con Fugas y Error LO (Banda Superior)")
# plt.xlabel("Tiempo [Símbolos $T_s$]")
# plt.ylabel("Amplitud")
# plt.grid(True)
# plt.tight_layout()

# # Efecto Aislado: Solo Solapamiento
# graficar_par_ojos(
#     out_bb_solo_solap, out_hf_solo_solap, sps=sps,
#     titulo_bb="Diagrama de Ojo: Efecto Aislado de Solapamiento (Banda Inferior)",
#     titulo_hb="Diagrama de Ojo: Efecto Aislado de Solapamiento (Banda Superior)"
# )

# # Efecto Aislado: Solo Jitter
# graficar_par_ojos(
#     out_bb_solo_jitter, out_hf_solo_jitter, sps=sps,
#     titulo_bb="Diagrama de Ojo: Efecto Aislado de Jitter ADC (Banda Inferior)",
#     titulo_hb="Diagrama de Ojo: Efecto Aislado de Jitter LO + ADC (Banda Superior)"
# )
# plt.show()


# # ==============================================================================
# # 6. DIAGRAMAS DE CONSTELACIÓN EN SUBPLOTS (BANDA INFERIOR Y SUPERIOR)
# # ==============================================================================

# # 1. Efecto Aislado: Solo T&H
# graficar_par_constelaciones(
#     senial_alineada, out_bb_solo_th, senial_alineada_hf, out_hf_solo_th,
#     titulo_bb="Constelación PAM-4: Efecto Aislado de T&H (Banda Inferior)",
#     titulo_hb="Constelación PAM-4: Efecto Aislado de T&H (Banda Superior)"
# )

# # 2. Efecto Aislado: Solo Mixer (Banda Superior)
# plt.figure(figsize=(10, 4))
# plt.scatter(np.real(senial_alineada_hf[::sps]), np.imag(senial_alineada_hf[::sps]), alpha=0.3, color='blue', marker='x', label='Tx Ideal')
# plt.scatter(np.real(out_hf_solo_mixer[delay::sps]), np.imag(out_hf_solo_mixer[delay::sps]), alpha=0.3, color='m', label='Rx Muestreada')
# plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
# plt.title('Constelación PAM-4: Efecto Aislado de Mixer (Banda Superior)')
# plt.xlabel('En Fase (I)')
# plt.ylabel('Cuadratura (Q)')
# plt.ylim(-0.5, 0.5)
# plt.grid(True, linestyle=':', alpha=0.6)
# plt.legend(loc='upper right')
# plt.tight_layout()

# # 3. Efecto Aislado: Solo Solapamiento
# graficar_par_constelaciones(
#     senial_alineada, out_bb_solo_solap, senial_alineada_hf, out_hf_solo_solap,
#     titulo_bb="Constelación PAM-4: Efecto Aislado de Solapamiento (Banda Inferior)",
#     titulo_hb="Constelación PAM-4: Efecto Aislado de Solapamiento (Banda Superior)"
# )

# # 4. Efecto Aislado: Solo Jitter
# graficar_par_constelaciones(
#     senial_alineada, out_bb_solo_jitter, senial_alineada_hf, out_hf_solo_jitter,
#     titulo_bb="Constelación PAM-4: Efecto Aislado de Jitter ADC (Banda Inferior)",
#     titulo_hb="Constelación PAM-4: Efecto Aislado de Jitter LO + ADC (Banda Superior)"
# )

# # 5. Suma Total de Efectos
# graficar_par_constelaciones(
#     senial_alineada, out_bb_total, senial_alineada_hf, out_hf_total,
#     titulo_bb="Constelación PAM-4 Final: Todos los Efectos Sumados (Banda Inferior)",
#     titulo_hb="Constelación PAM-4 Final: Todos los Efectos Sumados (Banda Superior)"
# )

# # 6. Resumen Final Comparativo en Subplots
# plt.figure(figsize=(10, 8))

# plt.subplot(2, 1, 1)
# niveles_bb = [0.15, 0.05, -0.05, -0.15]
# etiquetas_bb = [
#     '1. Solo T&H',
#     '2. Solo Solapamiento',
#     '3. Solo Jitter (ADC)',
#     '4. TODOS LOS EFECTOS'
# ]
# seniales_bb = [out_bb_solo_th, out_bb_solo_solap, out_bb_solo_jitter, out_bb_total]
# colores_bb = ['blue', 'green', 'orange', 'red']

# for y_val, sig, col, lab in zip(niveles_bb, seniales_bb, colores_bb, etiquetas_bb):
#     muestras = np.real(sig[delay::sps])
#     plt.scatter(muestras, np.zeros_like(muestras) + y_val, alpha=0.3, color=col, label=lab)

# plt.yticks(niveles_bb, etiquetas_bb)
# plt.title('Constelaciones PAM-4 para distintos efectos (Banda Inferior)')
# plt.xlabel('Amplitud en Fase (I)')
# plt.xlim(-4.5, 4.5)
# plt.ylim(-0.25, 0.25)
# plt.grid(True, linestyle=':', alpha=0.6)

# plt.subplot(2, 1, 2)
# niveles_hb = [0.2, 0.1, 0.0, -0.1, -0.2]
# etiquetas_hb = [
#     '1. Solo T&H',
#     '2. Solo Mixer',
#     '3. Solo Solapamiento',
#     '4. Solo Jitter',
#     '5. TODOS LOS EFECTOS'
# ]
# seniales_hb = [out_hf_solo_th, out_hf_solo_mixer, out_hf_solo_solap, out_hf_solo_jitter, out_hf_total]
# colores_hb = ['blue', 'purple', 'green', 'orange', 'red']

# for y_val, sig, col, lab in zip(niveles_hb, seniales_hb, colores_hb, etiquetas_hb):
#     muestras = np.real(sig[delay::sps])
#     plt.scatter(muestras, np.zeros_like(muestras) + y_val, alpha=0.3, color=col, label=lab)

# plt.yticks(niveles_hb, etiquetas_hb)
# plt.title('Constelaciones PAM-4 para distintos efectos (Banda Superior)')
# plt.xlabel('Amplitud en Fase (I)')
# plt.xlim(-4.5, 4.5)
# plt.ylim(-0.3, 0.3)
# plt.grid(True, linestyle=':', alpha=0.6)
# plt.tight_layout()
# plt.show()


# # ==============================================================================
# # 7. SEÑAL PAM-4 TOTAL RECONSTRUIDA (TIEMPO, OJO Y CONSTELACIÓN)
# # ==============================================================================

# # 1. Señal en el Tiempo: PAM-4 Transmitida vs Recibida Reconstruida (Mismo inicio t=0)
# plt.figure(figsize=(10, 6))
# plt.subplot(2, 1, 1)
# plt.plot(senial_total_tx_alineada[:2000], color='b', alpha=0.6, label='Señal PAM-4 Total Tx (Conformada RC)')
# #plt.stem(simb_upsampled_total[:2000], label='Símbolos PAM-4 Tx')
# plt.xlabel('Muestras')
# plt.ylabel('Amplitud')
# plt.title('Señal PAM-4 Total Transmitida (Antes de dividir en sub-bandas)')
# plt.grid(True)
# plt.legend(loc='upper right')

# plt.subplot(2, 1, 2)
# plt.plot(senial_total_tx_alineada[:2000], color='r', alpha=0.4, linestyle='--', label='Señal PAM-4 Total Tx (Referencia)')
# plt.plot(senial_total_rx_alineada[:2000], color='purple', linewidth=1.5, label='Señal PAM-4 Total Rx Reconstruida (Todos los efectos)')
# plt.xlabel('Muestras')
# plt.ylabel('Amplitud')
# plt.title('Señal PAM-4 Total Reconstruida en el Receptor (Con todos los efectos del canal)')
# plt.grid(True)
# plt.legend(loc='upper right')
# plt.tight_layout()
# plt.show()

# # 2. Diagrama de Ojo: PAM-4 Total Transmitida vs Recibida Reconstruida
# graficar_par_ojos(
#     senial_total_tx_alineada, senial_total_rx_alineada, sps=sps_total,
#     titulo_bb="Diagrama de Ojo: Señal PAM-4 Total Transmitida (Ideal)",
#     titulo_hb="Diagrama de Ojo: Señal PAM-4 Total Recibida y Reconstruida (Todos los Efectos)"
# )
# plt.show()

# # 3. Constelación: PAM-4 Total Transmitida vs Recibida Reconstruida
# plt.figure(figsize=(10, 4))
# plt.scatter(np.real(simb_pam_total), np.imag(simb_pam_total), alpha=0.3, color='blue', marker='x', s=60, label='Tx Ideal')
# plt.scatter(np.real(simb_pam_rx_total), np.imag(simb_pam_rx_total), alpha=0.3, color='m', label='Rx Reconstruida (Pares + Impares)')
# plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
# plt.title('Constelación PAM-4 Total: Transmitida vs Recibida Reconstruida (Todos los Efectos)')
# plt.xlabel('En Fase (I)')
# plt.ylabel('Cuadratura (Q)')
# plt.ylim(-0.5, 0.5)
# plt.grid(True, linestyle=':', alpha=0.6)
# plt.legend(loc='upper right')
# plt.tight_layout()
# plt.show()