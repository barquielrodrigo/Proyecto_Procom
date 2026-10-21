import numpy as np
import matplotlib.pyplot as plt
import random
import sys
from scipy import signal
from scipy.signal import hilbert

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
M = 1000
simb_pam = np.random.choice([-3, -1, 1, 3], size=M)
simb_pam_hf = np.random.choice([-3, -1, 1, 3], size=M)

print(f"Primeros símbolos BB: {simb_pam[0:10]}")
print(f"Primeros símbolos HB: {simb_pam_hf[0:10]}")

sys.path.append("C:/Users/Usuario/Desktop/Valentino/PROCOM/TP1")
from filtro import RaisedCosineFilter # type: ignore

span = 6
alpha = 0.1
sps = 64

simb_upsampled = np.zeros(len(simb_pam) * sps)
simb_upsampled[::sps] = simb_pam  
simb_upsampled_hf = np.zeros(len(simb_pam_hf) * sps)
simb_upsampled_hf[::sps] = simb_pam_hf  

rc = RaisedCosineFilter(alpha=alpha, span=span, sps=sps, rrc=False)
rc_filter = rc.get_coefficients()
senial = np.convolve(simb_upsampled, rc_filter, mode='full')       # Banda base
senial_hf = np.convolve(simb_upsampled_hf, rc_filter, mode='full') # Sub-banda superior

delay = (span * sps) // 2
senial_alineada = senial[delay : delay + len(simb_upsampled)]
senial_alineada_hf = senial_hf[delay : delay + len(simb_upsampled)]

BW_deseado = 48e9
Rs = (2 * BW_deseado) / (1 + alpha)
fs = Rs * sps

fc = 48e9
t = np.arange(len(senial)) / fs
s2_hilbert = hilbert(senial_hf)
s2_ssb = np.real(s2_hilbert * np.exp(1j * 2 * np.pi * fc * t))

s_total_FI = senial + s2_ssb


# ==============================================================================
# 2. MODELADO DE CADA EFECTO AISLADO Y SUMA TOTAL AL FINAL
# ==============================================================================
BW_tia = 100e9
B_d = 48e9
LO_Port = 10**(-30/20)
RF_IF = 10**(-35/20)
epsilon = 0.01
phi_rad = np.deg2rad(9)
f_lo = 48e9

# Osciladores Locales
LO_ideal = 2 * np.cos(2 * np.pi * f_lo * t)
LO_estatico = (2 + epsilon/2) * np.cos(2 * np.pi * f_lo * t + phi_rad/2)

sigma_rw = 2 * np.pi * 8e-4
ruido_fase_lo = np.cumsum(np.random.normal(0, sigma_rw, size=len(t)))
LO_solo_jitter = 2 * np.cos(2 * np.pi * f_lo * t + ruido_fase_lo)
LO_completo = (2 + epsilon/2) * np.cos(2 * np.pi * f_lo * t + phi_rad/2 + ruido_fase_lo)

# Jitter de Apertura (Vector de tiempo para el ADC)
sigma_t_adc = 300e-15
t_ideal = np.arange(len(t)) / fs
t_jitter_adc = t_ideal + np.random.normal(0, sigma_t_adc, size=len(t))

# --- Etapa 1: TIA ---
senial_tia, _ = aplicar_lpf_butterworth(s_total_FI, fs, BW_tia, 3)        # Compuesta
senial_tia_bb_aislada, _ = aplicar_lpf_butterworth(senial, fs, BW_tia, 3) # BB aislada
senial_tia_hf_aislada, _ = aplicar_lpf_butterworth(s2_ssb, fs, BW_tia, 3) # HB aislada

# --- Etapa 2: Separación HPF (Solapamiento de BB hacia HB) ---
senial_hpf_con_solap, _ = aplicar_hpf_butterworth(senial_tia, fs, fc=48e9, orden=4)

# --- Etapa 3: Mixer (Aislados vs Completo) ---
RF_aislada, _ = aplicar_lpf_butterworth(senial_tia_hf_aislada, fs, 100e9, 1)
RF_solap, _ = aplicar_lpf_butterworth(senial_hpf_con_solap, fs, 100e9, 1)

# A) Solo Imperfecciones del Mixer (Fugas + Error estático LO, sobre señal aislada)
post_mix_estatico = (RF_aislada + LO_Port * LO_estatico) * LO_estatico + RF_IF * RF_aislada + LO_Port * LO_estatico
IF_hf_solo_mixer, _ = aplicar_lpf_butterworth(post_mix_estatico, fs, 100e9, 1)

# B) Solo Solapamiento (HPF sobre señal compuesta + Mixer ideal sin fugas ni jitter)
IF_hf_solo_solap, _ = aplicar_lpf_butterworth(RF_solap * LO_ideal, fs, 100e9, 1)

# C) Solo Jitter LO (Señal aislada + Mixer ideal con LO con jitter)
IF_hf_solo_jit_lo, _ = aplicar_lpf_butterworth(RF_aislada * LO_solo_jitter, fs, 100e9, 1)

# D) Cadena Ideal (Para aislar el efecto del T&H y del Jitter de ADC)
IF_hf_ideal, _ = aplicar_lpf_butterworth(RF_aislada * LO_ideal, fs, 100e9, 1)

# E) SUMA TOTAL DE EFECTOS EN MIXER (Solapamiento + Fugas + Error estático + Jitter LO)
post_mix_total = (RF_solap + LO_Port * LO_completo) * LO_completo + RF_IF * RF_solap + LO_Port * LO_completo
IF_hf_total, _ = aplicar_lpf_butterworth(post_mix_total, fs, 100e9, 1)

# --- Etapa 4: T&H y Muestreo ADC ---
# 1. Solo T&H (Sin solapamiento, sin fugas de mixer, sin jitter)
out_bb_solo_th, _ = aplicar_lpf_butterworth(senial_tia_bb_aislada, fs, B_d, 4)
out_hf_solo_th, _ = aplicar_lpf_butterworth(IF_hf_ideal, fs, B_d, 4)

# 2. Solo Imperfecciones de Mixer (En HB; BB no pasa por el mixer)
out_hf_solo_mixer, _ = aplicar_lpf_butterworth(IF_hf_solo_mixer, fs, B_d, 4)

# 3. Solo Solapamiento entre Bandas (Crosstalk mutuo en BB y HB)
out_bb_solo_solap, _ = aplicar_lpf_butterworth(senial_tia, fs, B_d, 4)
out_hf_solo_solap, _ = aplicar_lpf_butterworth(IF_hf_solo_solap, fs, B_d, 4)

# 4. Solo Jitter (En BB: Aperture Jitter ADC; En HB: Jitter LO + Aperture Jitter ADC)
out_bb_solo_jitter = np.interp(t_jitter_adc, t_ideal, out_bb_solo_th)
out_hf_pre_jit, _ = aplicar_lpf_butterworth(IF_hf_solo_jit_lo, fs, B_d, 4)
out_hf_solo_jitter = np.interp(t_jitter_adc, t_ideal, out_hf_pre_jit)

# 5. SUMA TOTAL DE TODOS LOS EFECTOS (T&H + Solapamiento + Mixer + Jitter)
out_bb_total = np.interp(t_jitter_adc, t_ideal, out_bb_solo_solap)
out_hf_pre_total, _ = aplicar_lpf_butterworth(IF_hf_total, fs, B_d, 4)
out_hf_total = np.interp(t_jitter_adc, t_ideal, out_hf_pre_total)


# ==============================================================================
# 3. GRÁFICOS: ESPECTROS
# ==============================================================================
rc.plot(time_domain=True, freq_domain=True)

# Espectros Tx (Banda Inferior y Superior en Subplots)
plt.figure(figsize=(10, 7))
espectro_fft = np.fft.fftshift(np.fft.fft(senial))
magnitud_db = 20 * np.log10(np.abs(espectro_fft) / np.max(np.abs(espectro_fft)))
freqs = np.fft.fftshift(np.fft.fftfreq(len(senial), d=1/fs)) / 1e9
plt.plot(freqs, magnitud_db)
plt.xlabel('Frecuencia [GHz]')
plt.ylabel('Magnitud Normalizada [dB]')
plt.xlim(-110, 110)
plt.title('Espectro Tx: Sub-banda Inferior en Banda Base (0 a 48 GHz)')
plt.grid(True)



N = len(s2_ssb)
freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/fs)) / 1e9
espectro_fft = np.fft.fftshift(np.fft.fft(s2_ssb))
magnitud_db = 20 * np.log10(np.abs(espectro_fft) / np.max(np.abs(espectro_fft)))
plt.figure(figsize=(10, 7))
plt.plot(freqs, magnitud_db, color='g')
plt.xlabel('Frecuencia [GHz]')
plt.ylabel('Magnitud Normalizada [dB]')
plt.xlim(-110, 110)
plt.title('Espectro Tx: Sub-banda Superior Modulada en SSB-SC (48 a 96 GHz)')
plt.grid(True)
plt.tight_layout()

# Espectro Compuesto FI y Separación HPF
plt.figure(figsize=(10, 7))
N = len(s_total_FI)
freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/fs)) / 1e9
espectro_fft = np.fft.fftshift(np.fft.fft(s_total_FI))
magnitud_db = 20 * np.log10(np.abs(espectro_fft) / np.max(np.abs(espectro_fft)))
plt.plot(freqs, magnitud_db)
plt.xlabel('Frecuencia [GHz]')
plt.ylabel('Magnitud Normalizada [dB]')
plt.xlim(-110, 110)
plt.title('Espectro en Canal FI: Señal Compuesta s_total_FI (0 a 96 GHz)')
plt.grid(True)

plt.figure(figsize=(10, 7))
N = len(senial_tia_hf_aislada)
freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/fs)) / 1e9
esp_orig = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(senial_tia_hf_aislada))) / np.max(np.abs(np.fft.fft(senial_tia_hf_aislada))))
esp_solap = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(senial_hpf_con_solap))) / np.max(np.abs(np.fft.fft(senial_hpf_con_solap))))
plt.plot(freqs, esp_solap, color='r', alpha=0.8, label='Con Solapamiento (HPF 4º Orden sobre s_total_FI)')
plt.plot(freqs, esp_orig, color='b', linestyle='--', alpha=0.7, label='Sin Solapamiento (s2_ssb aislada)')
plt.axvline(48, color='k', linestyle=':', label='Frecuencia de cruce (48 GHz)')
plt.xlabel('Frecuencia [GHz]')
plt.ylabel('Magnitud Normalizada [dB]')
plt.xlim(-110, 110)
plt.ylim(-100, 5)
plt.title('Espectro en Puerto RF del Mixer: Efecto de Solapamiento de Banda Inferior')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ==============================================================================
# 4. GRÁFICOS TEMPORALES EN SUBPLOTS (BANDA INFERIOR Y SUPERIOR)
# ==============================================================================

# 1. Señales Transmitidas (Tx)
plt.figure(figsize=(10, 7))
plt.subplot(2, 1, 1)
plt.plot(senial_alineada[:2000], color='b', alpha=0.5, label='Señal Conformada RC')
plt.stem(simb_upsampled[:2000], label='Símbolos PAM-4')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.title('Señal Transmitida PAM-4: Banda Inferior (Banda Base)')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(senial_alineada_hf[:2000], color='g', alpha=0.5, label='Señal Conformada RC')
plt.stem(simb_upsampled_hf[:2000], label='Símbolos PAM-4')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.title('Señal Transmitida PAM-4: Banda Superior (en Banda Base)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 2. Efecto Aislado del TIA
plt.figure(figsize=(10, 7))
plt.subplot(2, 1, 1)
plt.plot(senial[:2000], label="Entrada TIA (Banda Inferior)", alpha=0.6)
plt.plot(senial_tia_bb_aislada[:2000], label="Salida TIA (LPF 3º Ord, 100 GHz)", linewidth=2)
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.title('Efecto Aislado del TIA: Banda Inferior')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(s2_ssb[:2000], label="Entrada TIA (Banda Superior SSB)", alpha=0.6)
plt.plot(senial_tia_hf_aislada[:2000], label="Salida TIA (LPF 3º Ord, 100 GHz)", linewidth=2)
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.title('Efecto Aislado del TIA: Banda Superior')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 3. Efecto Aislado del Mixer (Exclusivo de Banda Superior)
plt.figure(figsize=(10, 4))
plt.plot(out_hf_solo_th[:2000], label="Banda Superior con Mixer Ideal (Sin fugas)", alpha=0.7)
plt.plot(out_hf_solo_mixer[:2000], label="Banda Superior con Mixer Real (Fugas LO/RF + Error Fase/Ganancia)", linewidth=1.5)
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.title('Efecto Aislado del Mezclador (Fugas y Error Estático LO): Banda Superior')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 4. Efecto Aislado del Track & Hold (T&H)
plt.figure(figsize=(10, 7))
plt.subplot(2, 1, 1)
plt.plot(senial_tia_bb_aislada[:2000], label="Entrada T&H (Banda Inferior Aislada)", alpha=0.6)
plt.plot(out_bb_solo_th[:2000], label="Salida T&H (LPF 4º Ord, 48 GHz)", linewidth=2)
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.title('Efecto Aislado del T&H: Banda Inferior')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(IF_hf_ideal[:2000], label="Entrada T&H (Banda Superior Demodulada Ideal)", alpha=0.6)
plt.plot(out_hf_solo_th[:2000], label="Salida T&H (LPF 4º Ord, 48 GHz)", linewidth=2)
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.title('Efecto Aislado del T&H: Banda Superior')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 5. Efecto Aislado del Solapamiento entre Bandas (Crosstalk)
plt.figure(figsize=(10, 7))
plt.subplot(2, 1, 1)
plt.plot(out_bb_solo_th[:2000], label="Sin Solapamiento (Banda Inferior Aislada)", alpha=0.7)
plt.plot(out_bb_solo_solap[:2000], label="Con Solapamiento (Fuga desde Banda Superior)", linewidth=1.5)
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.title('Efecto Aislado de Solapamiento (Crosstalk): Banda Inferior')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(out_hf_solo_th[:2000], label="Sin Solapamiento (Banda Superior Aislada)", alpha=0.7)
plt.plot(out_hf_solo_solap[:2000], label="Con Solapamiento (Fuga desde Banda Inferior tras HPF 4º Ord)", linewidth=1.5)
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.title('Efecto Aislado de Solapamiento (Crosstalk): Banda Superior')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 6. Efecto Aislado del Jitter
plt.figure(figsize=(10, 7))
plt.subplot(2, 1, 1)
plt.plot(out_bb_solo_th[:2000], label="Sin Jitter (Banda Inferior)", alpha=0.7)
plt.plot(out_bb_solo_jitter[:2000], label="Con Jitter de Apertura (ADC)", linewidth=1.5)
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.title('Efecto Aislado de Jitter (Aperture Jitter ADC): Banda Inferior')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(out_hf_solo_th[:2000], label="Sin Jitter (Banda Superior)", alpha=0.7)
plt.plot(out_hf_solo_jitter[:2000], label="Con Jitter de Fase (LO) y Apertura (ADC)", linewidth=1.5)
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.title('Efecto Aislado de Jitter (LO + ADC): Banda Superior')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 7. SUMA DE TODOS LOS EFECTOS (Comparación Final en el Tiempo)
plt.figure(figsize=(10, 7))
plt.subplot(2, 1, 1)
plt.plot(senial[:2000], color='r', alpha=0.5, label="Señal Tx Original (Banda Inferior)")
plt.plot(out_bb_total[:2000], color='b', linewidth=1.5, label="Señal Rx con Todos los Efectos (T&H + Solapamiento + Jitter)")
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.title('Señal PAM-4 Final con Todos los Efectos Sumados: Banda Inferior')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(senial_hf[:2000], color='g', alpha=0.5, label="Señal Tx Original (Banda Superior)")
plt.plot(out_hf_total[:2000], color='m', linewidth=1.5, label="Señal Rx con Todos los Efectos (T&H + Mixer + Solapamiento + Jitter)")
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.title('Señal PAM-4 Final con Todos los Efectos Sumados: Banda Superior')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ==============================================================================
# 5. DIAGRAMAS DE OJO EN SUBPLOTS (BANDA INFERIOR Y SUPERIOR)
# ==============================================================================

# 1. Ideales Tx
graficar_par_ojos(
    senial_alineada, senial_alineada_hf, sps=sps,
    titulo_bb="Diagrama de Ojo: Señal Tx Ideal (Banda Inferior)",
    titulo_hb="Diagrama de Ojo: Señal Tx Ideal (Banda Superior)"
)

# 2. Efecto Aislado: Solo T&H
graficar_par_ojos(
    out_bb_solo_th, out_hf_solo_th, sps=sps,
    titulo_bb="Diagrama de Ojo: Efecto Aislado de T&H (Banda Inferior)",
    titulo_hb="Diagrama de Ojo: Efecto Aislado de T&H (Banda Superior)"
)

# 3. Efecto Aislado: Solo Mixer (Banda Superior)
graficar_diagrama_ojo(out_hf_solo_mixer, sps=sps, nueva_figura=True)
plt.title("Diagrama de Ojo: Efecto Aislado del Mixer con Fugas y Error LO (Banda Superior)")
plt.xlabel("Tiempo [Símbolos $T_s$]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.tight_layout()

# 4. Efecto Aislado: Solo Solapamiento
graficar_par_ojos(
    out_bb_solo_solap, out_hf_solo_solap, sps=sps,
    titulo_bb="Diagrama de Ojo: Efecto Aislado de Solapamiento (Banda Inferior)",
    titulo_hb="Diagrama de Ojo: Efecto Aislado de Solapamiento (Banda Superior)"
)

# 5. Efecto Aislado: Solo Jitter
graficar_par_ojos(
    out_bb_solo_jitter, out_hf_solo_jitter, sps=sps,
    titulo_bb="Diagrama de Ojo: Efecto Aislado de Jitter ADC (Banda Inferior)",
    titulo_hb="Diagrama de Ojo: Efecto Aislado de Jitter LO + ADC (Banda Superior)"
)

# 6. Suma Total de Efectos
graficar_par_ojos(
    out_bb_total, out_hf_total, sps=sps,
    titulo_bb="Diagrama de Ojo Final: Todos los Efectos Sumados (Banda Inferior)",
    titulo_hb="Diagrama de Ojo Final: Todos los Efectos Sumados (Banda Superior)"
)
plt.show()


# ==============================================================================
# 6. DIAGRAMAS DE CONSTELACIÓN EN SUBPLOTS (BANDA INFERIOR Y SUPERIOR)
# ==============================================================================

# 1. Efecto Aislado: Solo T&H
graficar_par_constelaciones(
    senial_alineada, out_bb_solo_th, senial_alineada_hf, out_hf_solo_th,
    titulo_bb="Constelación PAM-4: Efecto Aislado de T&H (Banda Inferior)",
    titulo_hb="Constelación PAM-4: Efecto Aislado de T&H (Banda Superior)"
)

# 2. Efecto Aislado: Solo Mixer (Banda Superior)
plt.figure(figsize=(10, 4))
plt.scatter(np.real(senial_alineada_hf[::sps]), np.imag(senial_alineada_hf[::sps]), alpha=0.3, color='blue', marker='x', label='Tx Ideal')
plt.scatter(np.real(out_hf_solo_mixer[delay::sps]), np.imag(out_hf_solo_mixer[delay::sps]), alpha=0.3, color='m', label='Rx Muestreada')
plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.title('Constelación PAM-4: Efecto Aislado de Mixer (Banda Superior)')
plt.xlabel('En Fase (I)')
plt.ylabel('Cuadratura (Q)')
plt.ylim(-0.5, 0.5)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()

# 3. Efecto Aislado: Solo Solapamiento
graficar_par_constelaciones(
    senial_alineada, out_bb_solo_solap, senial_alineada_hf, out_hf_solo_solap,
    titulo_bb="Constelación PAM-4: Efecto Aislado de Solapamiento (Banda Inferior)",
    titulo_hb="Constelación PAM-4: Efecto Aislado de Solapamiento (Banda Superior)"
)

# 4. Efecto Aislado: Solo Jitter
graficar_par_constelaciones(
    senial_alineada, out_bb_solo_jitter, senial_alineada_hf, out_hf_solo_jitter,
    titulo_bb="Constelación PAM-4: Efecto Aislado de Jitter ADC (Banda Inferior)",
    titulo_hb="Constelación PAM-4: Efecto Aislado de Jitter LO + ADC (Banda Superior)"
)

# 5. Suma Total de Efectos
graficar_par_constelaciones(
    senial_alineada, out_bb_total, senial_alineada_hf, out_hf_total,
    titulo_bb="Constelación PAM-4 Final: Todos los Efectos Sumados (Banda Inferior)",
    titulo_hb="Constelación PAM-4 Final: Todos los Efectos Sumados (Banda Superior)"
)

# 6. Resumen Final Comparativo en Subplots (Todos los efectos desglosados en BB y HB)
plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
niveles_bb = [0.15, 0.05, -0.05, -0.15]
etiquetas_bb = [
    '1. Solo T&H',
    '2. Solo Solapamiento (Fuga HB)',
    '3. Solo Jitter (ADC)',
    '4. TODOS LOS EFECTOS SUMADOS'
]
seniales_bb = [out_bb_solo_th, out_bb_solo_solap, out_bb_solo_jitter, out_bb_total]
colores_bb = ['blue', 'green', 'orange', 'red']

for y_val, sig, col, lab in zip(niveles_bb, seniales_bb, colores_bb, etiquetas_bb):
    muestras = np.real(sig[delay::sps])
    plt.scatter(muestras, np.zeros_like(muestras) + y_val, alpha=0.3, color=col, label=lab)

plt.yticks(niveles_bb, etiquetas_bb)
plt.title('Resumen de Constelaciones PAM-4: Efectos Aislados vs Suma Total (Banda Inferior)')
plt.xlabel('Amplitud en Fase (I)')
plt.ylim(-0.25, 0.25)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')

plt.subplot(2, 1, 2)
niveles_hb = [0.2, 0.1, 0.0, -0.1, -0.2]
etiquetas_hb = [
    '1. Solo T&H',
    '2. Solo Mixer (Fugas + Error LO)',
    '3. Solo Solapamiento (Fuga BB)',
    '4. Solo Jitter (LO + ADC)',
    '5. TODOS LOS EFECTOS SUMADOS'
]
seniales_hb = [out_hf_solo_th, out_hf_solo_mixer, out_hf_solo_solap, out_hf_solo_jitter, out_hf_total]
colores_hb = ['blue', 'purple', 'green', 'orange', 'red']

for y_val, sig, col, lab in zip(niveles_hb, seniales_hb, colores_hb, etiquetas_hb):
    muestras = np.real(sig[delay::sps])
    plt.scatter(muestras, np.zeros_like(muestras) + y_val, alpha=0.3, color=col, label=lab)

plt.yticks(niveles_hb, etiquetas_hb)
plt.title('Resumen de Constelaciones PAM-4: Efectos Aislados vs Suma Total (Banda Superior)')
plt.xlabel('Amplitud en Fase (I)')
plt.ylim(-0.3, 0.3)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()