import numpy as np
import matplotlib.pyplot as plt
import random
import sys
from scipy import signal
from scipy.signal import hilbert

def aplicar_lpf_butterworth(senial, fs, fc, orden, causal=False):
    """
    Parametros:
    - senial: Array 1D de la señal a filtrar.
    - fs: Frecuencia de muestreo de la simulación [Hz].
    - fc: Frecuencia de corte a -3 dB [Hz].
    - orden: Orden del filtro.
    - causal: Si True, usa 'sosfilt' (comportamiento físico real con retardo).
              Si False, usa 'sosfiltfilt' (sin retardo de grupo).
    """
    sos = signal.butter(N=orden, Wn=fc, btype='lowpass', analog=False, fs=fs, output='sos')
    
    if causal:
        senial_filtrada = signal.sosfilt(sos, senial)
    else:
        senial_filtrada = signal.sosfiltfilt(sos, senial)
        
    return senial_filtrada, sos

def aplicar_hpf_butterworth(senial, fs, fc, orden, causal=False):
    """
    Modelo del filtro c(t) previo al mixer (HPF a 48 GHz).
    
    Parametros:
    - senial: Entrada.
    - fs: Frecuencia de muestreo.
    - fc: Frecuencia de corte.
    - orden: Orden Butterworth (4 recomendado).
    - causal: Si True, usa 'sosfilt' (comportamiento físico real con retardo).
              Si False, usa 'sosfiltfilt' (sin retardo de grupo).
    """
    sos = signal.butter(N=orden, Wn=fc, btype='highpass', analog=False, fs=fs, output='sos')
    
    if causal:
        senial_filtrada = signal.sosfilt(sos, senial)
    else:
        senial_filtrada = signal.sosfiltfilt(sos, senial)
        
    return senial_filtrada, sos

def graficar_diagrama_ojo(senial, sps, n_simbolos_pantalla=3, n_trazas=500):
    muestras_por_traza = sps * n_simbolos_pantalla
    n_trazas_posibles = len(senial) // muestras_por_traza
    n_trazas = min(n_trazas, n_trazas_posibles)
    
    plt.figure(figsize=(8, 4))
    t_eje = np.arange(muestras_por_traza) / sps  # Eje en tiempos de símbolo (T_s)
    
    for i in range(n_trazas):
        idx_inicio = i * muestras_por_traza
        segmento = senial[idx_inicio : idx_inicio + muestras_por_traza]
        plt.plot(t_eje, segmento, color='blue', alpha=0.03)


M = 1000 # cant de simbolos PAM-4 a transmitir

simb_pam = np.random.choice([-3, -1, 1, 3], size=M)
simb_pam_hf = np.random.choice([-3, -1, 1, 3], size=M)

# print(f'{simb_pam.size}')
print(f'{simb_pam[0:10]}')
print(f'{simb_pam_hf[0:10]}')

# Aplico el coseno realzado
sys.path.append("C:/Users/Usuario/Desktop/Valentino/PROCOM/TP1")
from filtro import RaisedCosineFilter # type: ignore
# Upsampling a los simbolos
span=6
alpha=0.1
sps=64
simb_upsampled = np.zeros(len(simb_pam) * sps)
simb_upsampled[::sps] = simb_pam  
simb_upsampled_hf = np.zeros(len(simb_pam_hf) * sps)
simb_upsampled_hf[::sps] = simb_pam_hf  

rc = RaisedCosineFilter(alpha=alpha, span=span, sps=sps, rrc=False)
rc_filter = rc.get_coefficients()
senial = np.convolve(simb_upsampled,rc_filter,mode='full') # Banda base
senial_hf = np.convolve(simb_upsampled_hf,rc_filter,mode='full') # Sub-banda superior

delay = (span * sps) // 2
senial_alineada = senial[delay : delay + len(simb_upsampled)]
senial_alineada_hf = senial_hf[delay : delay + len(simb_upsampled)]

BW_deseado = 48e9  # 48 GHz
Rs = (2 * BW_deseado) / (1 + alpha)  # ~87.27 GBaud para alpha=0.1
fs = Rs * sps                       # Frecuencia de muestreo física (Hz)

# Se tiene la señal PAM4 a transmitir por cada sub-banda
espectro_fft = np.fft.fftshift(np.fft.fft(senial))
magnitud_db = 20 * np.log10(np.abs(espectro_fft) / np.max(np.abs(espectro_fft)))
freqs = np.fft.fftshift(np.fft.fftfreq(len(senial), d=1/fs)) / 1e9

# Verificación del ancho de banda
# idx_48ghz = np.argmin(np.abs(freqs - 48.0))
# atenuacion_48ghz = magnitud_db[idx_48ghz]

# print(f"Tasa de Símbolos (Rs): {Rs/1e9:.2f} GBaud")
# print(f"Frecuencia de Muestreo (fs): {fs/1e9:.2f} GHz")
# print(f"Nivel de potencia a 48 GHz: {atenuacion_48ghz:.2f} dB")

# if atenuacion_48ghz <= -20:
#     print("STATUS: VERIFICACIÓN EXITOSA (El espectro cae adecuadamente antes de 48 GHz)")
# else:
#     print("STATUS: ALERTA (La señal excede el ancho de banda de 48 GHz)")

plt.figure(figsize=(8,6))
plt.plot(freqs,magnitud_db)
plt.xlabel('Frecuencia [GHz]')
plt.ylabel('Amplitud')
plt.xlim(-110,110)
plt.title('Espectro de la señal PAM-4 a transmitir')

# Creo la sub-banda superior a partir de la misma señal PAM-4
fc = 48e9  # Portadora de entrelazado a 48 GHz
t = np.arange(len(senial)) / fs

# Generación de la banda lateral superior (USB: 48 a 96 GHz)
s2_hilbert = hilbert(senial_hf)
s2_ssb = np.real(s2_hilbert * np.exp(1j * 2 * np.pi * fc * t))
#Grafico
N = len(s2_ssb)
freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/fs)) / 1e9  # Eje en GHz
espectro_fft = np.fft.fftshift(np.fft.fft(s2_ssb))
magnitud_db = 20 * np.log10(np.abs(espectro_fft) / np.max(np.abs(espectro_fft)))
plt.figure(figsize=(8,6))
plt.plot(freqs,magnitud_db, color='g')
plt.xlabel('Frecuencia [GHz]')
plt.ylabel('Amplitud')
plt.xlim(-110,110)
plt.title('Espectro de la señal PAM-4 a transmitir (sub-banda superior)')

# Señal FI
s_total_FI = senial + s2_ssb # Esta seria la señal que viaja por el canal y arriba al TIA
# Grafico
N = len(s_total_FI)
freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/fs)) / 1e9  # Eje en GHz
espectro_fft = np.fft.fftshift(np.fft.fft(s_total_FI))
magnitud_db = 20 * np.log10(np.abs(espectro_fft) / np.max(np.abs(espectro_fft)))
plt.figure(figsize=(8,6))
plt.plot(freqs,magnitud_db)
plt.xlabel('Frecuencia [GHz]')
plt.ylabel('Amplitud')
plt.xlim(-110,110)
plt.title('Espectro de la señal a transmitir (dos sub-bandas de PAM-4)')
# plt.show()

# Primer bloque: TIA
BW_tia = 100e9
senial_tia, sos_tia = aplicar_lpf_butterworth(s_total_FI,fs,BW_tia,3)

# Parte 2: Mixer para llevar a banda base
# A la sub-banda superior le debo aplicar el mixer

LO_Port = 10**(-30/20) # fugas en veces
RF_IF = 10**(-35/20) # fuga en veces
epsilon = 0.01 # error de ganancia
phi_rad = np.deg2rad(9) # error de fase [rad]
f_lo = 48e9 #depende del centro de las subbandas
# LO = np.tanh((2 + epsilon/2) * np.cos(2 * np.pi * f_lo * t + phi_rad)) # senal cuadrada real
LO = (2 + epsilon/2) * np.cos(2 * np.pi * f_lo * t + phi_rad/2) # senal cuadrada perfecta

# #visualizo el LO
# plt.figure(figsize=(8,6))
# plt.plot(t,LO)
# plt.title('Señal cuadrada del oscilador local')

# Antes del mixer debo aplicar el filtrado para evitar solapamiento o crosstalk (Sistema real)
# senial_tia_hf, sos_hpf = aplicar_hpf_butterworth(senial_tia,fs,fc=48e9,orden=4)
# Directamente utilizo la sub-banda superior y le aplico el TIA                 (Sistema ideal)
senial_tia_hf, _ = aplicar_lpf_butterworth(s2_ssb,fs,BW_tia,3)

# Grafico espectral de la subbanda superior llevada a banda base
N = len(senial_tia_hf)
freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/fs)) / 1e9  # Eje en GHz
espectro_fft = np.fft.fftshift(np.fft.fft(senial_tia_hf))
magnitud_db = 20 * np.log10(np.abs(espectro_fft) / np.max(np.abs(espectro_fft)))
plt.figure(figsize=(8,6))
plt.plot(freqs,magnitud_db)
plt.xlabel('Frecuencia [GHz]')
plt.ylabel('Amplitud')
plt.xlim(-110,110)
plt.title('Espectro de la señal de subbanda superior luego de TIA y HPF')
plt.show()

# Modelo de mixer del paper
RF , sos_RF = aplicar_lpf_butterworth(senial_tia_hf,fs,100e9,1) #filtro de entrada
# RF = s2_ssb # para comprobar que la SSB tiene la señal PAM4
post_mix = (RF + LO_Port * LO) * LO #agrego la fuga y paso por el mixer
post_mix = post_mix + RF_IF * RF + LO_Port * LO #agrego las otras fugas

IF_hf, sos_mixed_hf = aplicar_lpf_butterworth(post_mix,fs,100e9,1) #filtrado de salida

# Parte 3: T&H y otros
B_d = 48e9           # Ancho de banda nominal de cada sub-banda: 48 GHz
out_bb, _ = aplicar_lpf_butterworth(senial_tia,fs,B_d,4) # Banda base no atraviesa el mixer, es la salida del TIA
out_hf, _ = aplicar_lpf_butterworth(IF_hf,fs,B_d,4) # Alta frecuencia pasa por mixer

# Graficos

# Filtro coseno realzado
rc.plot(time_domain=True, freq_domain=True)

# Señal transmitida PAM-4
plt.figure(figsize=(10,8))
plt.subplot(2,1,1)
plt.plot(senial_alineada[0:2000], color='b', alpha=0.4)
plt.stem(simb_upsampled[0:2000])
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.title('Señal PAM-4 transmitida (Banda Base)')
plt.subplot(2,1,2)
plt.plot(senial_alineada_hf[0:2000], color='b', alpha=0.4)
plt.stem(simb_upsampled_hf[0:2000])
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.title('Señal PAM-4 transmitida (Banda superior)')
plt.tight_layout()
# plt.show()

#Visualizo el resultado del TIA (solo retardo, no filtra nada de las señales utiles)
plt.figure(figsize=(10, 4))
plt.plot(s_total_FI[:2000], label="Entrada TIA (PAM-4 conformada)", alpha=0.6)
plt.plot(senial_tia[:2000], label="Salida TIA (Filtrada LPF 3º Ord)", linewidth=2)
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.title('Efecto del TIA sobre la señal transmitida')
plt.grid(True)
plt.legend()
# plt.show()

# Visualizo el resultado del mixer (deberia incrementar la frecuencia)
plt.figure(figsize=(10, 4))
plt.plot(senial_tia_hf[:2000], label="Entrada al Mixer", alpha=0.6)
plt.plot(IF_hf[:2000], label="Salida Mixer", linewidth=2)
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.title('Efecto del mixer sobre la señal transmitida')
plt.grid(True)
plt.legend()
# plt.show()

# Visualizo el resultado del T&H
plt.figure(figsize=(10, 4))
plt.subplot(2,1,1)
plt.plot(senial_tia[:2000], label="Entrada al T&H (Banda Base)", alpha=0.6)
plt.plot(out_bb[:2000], label="Salida T&H (Banda Base)", linewidth=2)
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()
plt.title('Efecto del T&H sobre la señal transmitida')
plt.subplot(2,1,2)
plt.plot(IF_hf[:2000], label="Entrada al T&H (Sub-banda superior)", alpha=0.6)
plt.plot(out_hf[:2000], label="Salida T&H (Sub-banda superior)", linewidth=2)
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()
# plt.show()

# Visualizo el resultado de toda la cadena
plt.figure(figsize=(10, 4))
plt.subplot(2,1,1)
plt.plot(senial[:2000], color='r', label="Señal original (BB)", alpha=0.6)
plt.plot(out_bb[:2000], color='r', label="Señal a ecualizar (BB)", linewidth=2)
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.title('Efecto de la cadena de procesamiento sobre la señal transmitida')
plt.grid(True)
plt.legend()
plt.subplot(2,1,2)
plt.plot(senial_hf[:2000], color='g', label="Señal original (HB)", alpha=0.6)
plt.plot(out_hf[:2000], color='g', label="Señal a ecualizar (HB)", linewidth=2)
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.title('Efecto de la cadena de procesamiento sobre la señal transmitida')
plt.grid(True)
plt.legend()
# plt.show()

# Diagramas de ojo en cada etapa

graficar_diagrama_ojo(senial_alineada, sps=sps)
plt.title("Diagrama de Ojo - Ideal (BB)")
plt.xlabel("Tiempo [Símbolos $T_s$]")
plt.ylabel("Amplitud")
plt.grid(True)

graficar_diagrama_ojo(senial_alineada_hf, sps=sps)
plt.title("Diagrama de Ojo - Ideal (HB)")
plt.xlabel("Tiempo [Símbolos $T_s$]")
plt.ylabel("Amplitud")
plt.grid(True)

graficar_diagrama_ojo(out_bb, sps=sps)
plt.title("Diagrama de Ojo - Resultado final (BB)")
plt.xlabel("Tiempo [Símbolos $T_s$]")
plt.ylabel("Amplitud")
plt.grid(True)

graficar_diagrama_ojo(out_hf, sps=sps)
plt.title("Diagrama de Ojo - Resultado final (HB)")
plt.xlabel("Tiempo [Símbolos $T_s$]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()

# Constelaciones
# Banda Base
plt.figure(figsize=(8, 4))

# Puntos recibidos (usamos alpha para visualizar la densidad de puntos con ruido/ISI)
plt.scatter(np.real(senial_alineada[::sps]), np.imag(senial_alineada[::sps]), alpha=0.3, color='blue', marker='x', edgecolors='none', label='Símbolos transmitidos')

plt.scatter(np.real(out_bb[delay::sps]), np.imag(out_bb[delay::sps]), alpha=0.3, color='m', edgecolors='none', label='Símbolos recibidos')


plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.title('Diagrama de Constelación PAM-4 (BB)')
plt.xlabel('En Fase (I)')
plt.ylabel('Cuadratura (Q)')
plt.ylim(-0.5, 0.5)  # Ajuste vertical para enfocar el eje I
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()

# Banda alta
plt.figure(figsize=(8, 4))

# Puntos recibidos (usamos alpha para visualizar la densidad de puntos con ruido/ISI)
plt.scatter(np.real(senial_alineada_hf[::sps]), np.imag(senial_alineada_hf[::sps]), alpha=0.3, color='blue', marker='x', label='Símbolos transmitidos')

plt.scatter(np.real(out_hf[delay::sps]), np.imag(out_hf[delay::sps]), alpha=0.3, color='m', label='Símbolos recibidos')

plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.title('Diagrama de Constelación PAM-4 (HB)')
plt.xlabel('En Fase (I)')
plt.ylabel('Cuadratura (Q)')
plt.ylim(-0.5, 0.5)  # Ajuste vertical para enfocar el eje I
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()