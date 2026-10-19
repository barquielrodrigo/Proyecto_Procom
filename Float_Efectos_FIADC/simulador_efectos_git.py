import numpy as np
import matplotlib.pyplot as plt
import random
import sys
from scipy import signal
from scipy.stats import gaussian_kde

# =============================================================================
# FUNCIONES DE PROCESAMIENTO Y GRAFICACIÓN
# =============================================================================

def aplicar_lpf_butterworth(senial, fs, fc, orden, causal=True):
    """
    Aplica filtro pasa-bajos Butterworth.
    Si causal=False, aplica filtrado de fase cero (sosfiltfilt) util para la referencia ideal.
    """
    sos = signal.butter(N=orden, Wn=fc, btype='lowpass', analog=False, fs=fs, output='sos')
    if causal:
        senial_filtrada = signal.sosfilt(sos, senial)
    else:
        senial_filtrada = signal.sosfiltfilt(sos, senial)
    return senial_filtrada, sos

def aplicar_skew_temporal(senial, fs, tau):
    """
    Aplica un retardo temporal fraccionario (time skew) en el dominio de la frecuencia
    multiplicando por exp(-j * 2 * pi * f * tau).
    """
    N = len(senial)
    f = np.fft.rfftfreq(N, d=1.0/fs)
    S_f = np.fft.rfft(senial)
    desfase = np.exp(-1j * 2 * np.pi * f * tau)
    return np.fft.irfft(S_f * desfase, n=N)

def obtener_instante_optimo(senial, ref_ideal):
    """
    Calcula el retardo de grupo entero (en muestras) introducido por los filtros analógicos
    mediante correlación cruzada con la señal ideal de referencia.
    """
    corr = np.correlate(senial, ref_ideal, mode='full')
    lag = int(np.argmax(np.abs(corr)) - (len(ref_ideal) - 1))
    return max(0, lag)

def obtener_muestras_simbolo(senial, sps, ref_ideal, lag_fijo=None):
    """
    Encuentra el retardo de grupo entero introducido por los filtros causales
    (o usa lag_fijo del reloj ideal) y diezma la señal a 1 muestra por símbolo (1 SPS).
    """
    if lag_fijo is None:
        lag = obtener_instante_optimo(senial, ref_ideal)
    else:
        lag = lag_fijo
    
    muestras = senial[lag::sps]
    margen = 15
    if len(muestras) > 2 * margen:
        muestras = muestras[margen:-margen]
    return muestras

def graficar_diagrama_ojo(senial, sps, titulo, ref_ideal=None, lag_fijo=None, n_simbolos_pantalla=3, n_trazas=400):
    """
    Grafica el diagrama de ojo en una figura propia sincronizando con el reloj ideal.
    """
    if lag_fijo is not None:
        lag = lag_fijo
    elif ref_ideal is not None:
        lag = obtener_instante_optimo(senial, ref_ideal)
    else:
        lag = 0

    muestras_por_traza = sps * n_simbolos_pantalla
    senial_estable = senial[10*sps + lag:]
    n_trazas_posibles = len(senial_estable) // muestras_por_traza
    n_trazas = min(n_trazas, n_trazas_posibles)
    
    plt.figure(figsize=(8, 4))
    t_eje = np.arange(muestras_por_traza) / sps
    
    for i in range(n_trazas):
        idx_inicio = i * muestras_por_traza
        segmento = senial_estable[idx_inicio : idx_inicio + muestras_por_traza]
        plt.plot(t_eje, segmento, color='blue', alpha=0.04, linewidth=2)
        
    plt.title(f"Diagrama de Ojo - {titulo}")
    plt.xlabel("Tiempo [Símbolos $T$]")
    plt.ylabel("Amplitud")
    plt.grid(True)

#con jitter
# def graficar_constelacion(senial_I, sps, ref_I, titulo, senial_Q=None, ref_Q=None, lag_fijo=None, cmap='jet'):
#     """
#     Grafica la constelación de los símbolos recibidos coloreados por densidad de puntos (Heatmap).
#     """
#     muestras_I = obtener_muestras_simbolo(senial_I, sps, ref_I, lag_fijo=lag_fijo)
    
#     plt.figure(figsize=(6.5, 6))
    
#     if senial_Q is not None:
#         if ref_Q is None:
#             ref_Q = ref_I
#         muestras_Q = obtener_muestras_simbolo(senial_Q, sps, ref_Q, lag_fijo=lag_fijo)
#         n_min = min(len(muestras_I), len(muestras_Q))
#         muestras_I, muestras_Q = muestras_I[:n_min], muestras_Q[:n_min]
        
#         # 1. Cálculo de Densidad 2D con Gaussian KDE
#         xy = np.vstack([muestras_I, muestras_Q])
#         z = gaussian_kde(xy)(xy)
        
#         # 2. Ordenar puntos para que los más densos (rojos) se dibujen al frente
#         idx = z.argsort()
#         x, y, z = muestras_I[idx], muestras_Q[idx], z[idx]
        
#         # 3. Scatter plot coloreado por densidad
#         sc = plt.scatter(x, y, c=z, cmap=cmap, s=8, edgecolors='none')
#         cbar = plt.colorbar(sc)
#         cbar.set_label('Densidad de símbolos')
        
#         plt.xlabel("In phase")
#         plt.ylabel("Quadrature")
        
#     else:
#         # Caso 1D (PAM-4): Agregamos jitter aleatorio en Y para formar nubes 2D de densidad
#         y_jitter = np.random.normal(0, 0.08, size=len(muestras_I))
        
#         xy = np.vstack([muestras_I, y_jitter])
#         z = gaussian_kde(xy)(xy)
        
#         idx = z.argsort()
#         x, y, z = muestras_I[idx], y_jitter[idx], z[idx]
        
#         sc = plt.scatter(x, y, c=z, cmap=cmap, s=12, edgecolors='none')
#         cbar = plt.colorbar(sc)
#         cbar.set_label('Densidad de símbolos')
        
#         for nivel in [-3, -1, 1, 3]:
#             plt.axvline(nivel, color='black', linestyle='--', alpha=0.5)
            
#         plt.xlabel("In phase")
#         plt.ylabel("Quadrature")
#         plt.ylim(-1, 1)

#     plt.title(f"Constelación Recibida - {titulo}")
#     plt.axhline(0, color='gray', linewidth=0.5)
#     plt.axvline(0, color='gray', linewidth=0.5)
#     plt.grid(True, linestyle=':', alpha=0.7)
#     plt.tight_layout()

#Sin jitter
def graficar_constelacion(senial_I, sps, ref_I, titulo, senial_Q=None, ref_Q=None, lag_fijo=None, cmap='jet'):
    """
    Grafica la constelación de los símbolos recibidos.
    """
    muestras_I = obtener_muestras_simbolo(senial_I, sps, ref_I, lag_fijo=lag_fijo)
    
    plt.figure(figsize=(6.5, 6))
    
    if senial_Q is not None:
        if ref_Q is None:
            ref_Q = ref_I
        muestras_Q = obtener_muestras_simbolo(senial_Q, sps, ref_Q, lag_fijo=lag_fijo)
        n_min = min(len(muestras_I), len(muestras_Q))
        muestras_I, muestras_Q = muestras_I[:n_min], muestras_Q[:n_min]
        
        # Cálculo de Densidad 2D con Gaussian KDE
        xy = np.vstack([muestras_I, muestras_Q])
        z = gaussian_kde(xy)(xy)
        
        idx = z.argsort()
        x, y, z = muestras_I[idx], muestras_Q[idx], z[idx]
        
        sc = plt.scatter(x, y, c=z, cmap=cmap, s=8, edgecolors='none')
        cbar = plt.colorbar(sc)
        cbar.set_label('Densidad de símbolos')
        
        plt.xlabel("In phase")
        plt.ylabel("Quadrature")
        
    else:
        # Caso 1D (PAM-4): Sin jitter, color uniforme sobre Q = 0 y sin barra de color
        y = np.zeros_like(muestras_I)
        plt.scatter(muestras_I, y, color='blue', s=20, alpha=0.6)
        
        for nivel in [-3, -1, 1, 3]:
            plt.axvline(nivel, color='black', linestyle='--', alpha=0.5)
            
        plt.xlabel("In phase")
        plt.ylabel("Quadrature")
        plt.ylim(-1, 1)

    plt.title(f"Constelación Recibida - {titulo}")
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()


# =============================================================================
# ETAPA 0: GENERACIÓN PAM-4 Y REFERENCIAS IDEALES EN FASE (I) Y CUADRATURA (Q)
# =============================================================================
M = 1500 # Cantidad de símbolos PAM-4 a transmitir
bits = np.random.randint(0, 2, size=2*M)

pares_bits = bits.reshape(-1, 2)
signo = 2*(pares_bits[:, 0] == 1) - 1
valor = 2*(pares_bits[:, 1] == 0) + 1
simb_pam = signo * valor

sys.path.append("C:/Users/Usuario/Desktop/Valentino/PROCOM/TP1")
from filtro import RaisedCosineFilter # type: ignore

span = 6
sps = 8
simb_upsampled = np.zeros(len(simb_pam) * sps)
simb_upsampled[::sps] = simb_pam  

rc = RaisedCosineFilter(alpha=0.1, span=span, sps=sps, rrc=False)
rc_filter = rc.get_coefficients()
senial = np.convolve(simb_upsampled, rc_filter, mode='full')

delay_rc = (span * sps) // 2
senial_alineada = senial[delay_rc : delay_rc + len(simb_upsampled)]

# Parámetros globales de operación (192 GBd)
f_B = 192e9          # 192 GBd (T ≈ 5.21 ps)
fs = sps * f_B       # 1.536 THz
f_lo = 48e9          # Oscilador local: 48 GHz
B_d = 48e9           # Ancho de banda nominal de cada sub-banda: 48 GHz
N_muestras = len(senial_alineada)
t = np.arange(N_muestras) / fs

# -----------------------------------------------------------------------------
# GENERACIÓN DE LAS SEÑALES I y Q IDEALES (Sin distorsiones del AFE/ADC)
# -----------------------------------------------------------------------------
LO_I_ideal = np.cos(2 * np.pi * f_lo * t)
LO_Q_ideal = np.sin(2 * np.pi * f_lo * t)

# 1) Salida del Mezclador Ideal (sin fugas, sin error de ganancia/fase, sin TIA)
I_mixer_ideal = senial_alineada * LO_I_ideal
Q_mixer_ideal = senial_alineada * LO_Q_ideal

# 2) Salida de Sub-banda Ideal del FI-ADC (filtrada a 48 GHz con fase cero, sin distorsión)
I_subbanda_ideal, _ = aplicar_lpf_butterworth(I_mixer_ideal, fs, B_d, 4, causal=False)
Q_subbanda_ideal, _ = aplicar_lpf_butterworth(Q_mixer_ideal, fs, B_d, 4, causal=False)


# =============================================================================
# CÁLCULO DE LAS ETAPAS SIGUIENDO EL DIAGRAMA DE BLOQUES (FIGURA 5-a)
# =============================================================================

# --- BLOQUE 1: TIA b(t) (LPF Butterworth Orden 3, BW = 100 GHz) ---
BW_tia = 100e9
senial_tia, _ = aplicar_lpf_butterworth(senial_alineada, fs, BW_tia, 3)

# --- BLOQUE 2: MEZCLADOR EN CUADRATURA (c_I, c_Q + Errores Ganancia/Fase + Fugas) ---
RF, _ = aplicar_lpf_butterworth(senial_tia, fs, 100e9, 1) # Filtro de entrada c(t)

LO_Port = 10**(-30 / 20) # -30 dB
RF_IF = 10**(-35 / 20)   # -35 dB
epsilon = 0.01           # Error de ganancia (1%)
phi_rad = np.deg2rad(9)  # Error de fase (9 grados)

LO_I = (1 + epsilon/2) * np.sign(np.cos(2 * np.pi * f_lo * t + phi_rad/2))
LO_Q = (1 - epsilon/2) * np.sign(np.sin(2 * np.pi * f_lo * t - phi_rad/2))

post_mix_I = (RF + LO_Port * LO_I) * LO_I + RF_IF * RF + LO_Port * LO_I
IF_I, _ = aplicar_lpf_butterworth(post_mix_I, fs, 100e9, 1)

post_mix_Q = (RF + LO_Port * LO_Q) * LO_Q + RF_IF * RF + LO_Port * LO_Q
IF_Q, _ = aplicar_lpf_butterworth(post_mix_Q, fs, 100e9, 1)

# --- BLOQUE 3: TRACK & HOLD CON BANDWIDTH MISMATCH d_I(t), d_Q(t) ---
delta_Bd_pct = 0.15                # Mismatch del +15% en rama Q
BW_th_I = B_d                      # 48.0 GHz
BW_th_Q = B_d * (1 + delta_Bd_pct) # 55.2 GHz

# A) Efecto Individual del BW Mismatch (actuando solo sobre I_mixer_ideal y Q_mixer_ideal)
I_solo_bw, _ = aplicar_lpf_butterworth(I_mixer_ideal, fs, BW_th_I, 4)
Q_solo_bw_nom, _ = aplicar_lpf_butterworth(Q_mixer_ideal, fs, BW_th_I, 4)
Q_solo_bw_mis, _ = aplicar_lpf_butterworth(Q_mixer_ideal, fs, BW_th_Q, 4)

# B) Efecto Acumulado en la cadena (TIA + Mixer + T&H con BW Mismatch)
out_I_th, _ = aplicar_lpf_butterworth(IF_I, fs, BW_th_I, 4)
out_Q_th_nom, _ = aplicar_lpf_butterworth(IF_Q, fs, BW_th_I, 4)
out_Q_th, _ = aplicar_lpf_butterworth(IF_Q, fs, BW_th_Q, 4)

# --- BLOQUE 4: TIME SKEW INTRABANDA (tau_IB) ---
tau_IB = 1.2e-12         # 1.2 ps de skew entre ramas I y Q
tau_I = -tau_IB / 2
tau_Q = +tau_IB / 2

# A) Efecto Individual del Skew (actuando sobre las sub-bandas ideales I y Q)
I_solo_skew = aplicar_skew_temporal(I_subbanda_ideal, fs, tau_I)
Q_solo_skew = aplicar_skew_temporal(Q_subbanda_ideal, fs, tau_Q)

# B) Efecto Acumulado Final (TIA + Mixer + T&H con BW Mismatch + Time Skew)
out_I_total = aplicar_skew_temporal(out_I_th, fs, tau_I)
out_Q_total = aplicar_skew_temporal(out_Q_th, fs, tau_Q)


# =============================================================================
# SECUENCIA DE GRÁFICOS EN EL ORDEN DEL DIAGRAMA DE BLOQUES
# =============================================================================

# -----------------------------------------------------------------------------
# PASO 0: SEÑAL TRANSMITIDA IDEAL Y RECONSTRUCCIÓN IDEAL
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(senial_alineada[:200], color='b', alpha=0.6, label='Señal PAM-4 conformada')
plt.stem(simb_upsampled[:200], linefmt='r-', markerfmt='ro', basefmt='k-', label='Símbolos ideales')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.title('Paso 0: Señal PAM-4 Transmitida Ideal (Banda Base Completa)')
plt.grid(True)
plt.legend()
plt.show()

graficar_diagrama_ojo(senial_alineada, sps, "Paso 0: Señal Ideal Transmitida", lag_fijo=0)
graficar_constelacion(senial_alineada, sps, senial_alineada, "Paso 0: Señal PAM-4 Ideal", lag_fijo=0)
plt.show()

# Reconstrucción Ideal desde sub-bandas de fase cero
reconst_ideal = 2 * (I_subbanda_ideal * np.cos(2 * np.pi * f_lo * t) + Q_subbanda_ideal * np.sin(2 * np.pi * f_lo * t))
graficar_diagrama_ojo(np.real(reconst_ideal), sps, "Paso 0: Reconstrucción Ideal", lag_fijo=0)
#graficar_constelacion(np.real(reconst_ideal), sps, senial_alineada, "Paso 0: Reconstrucción Ideal", lag_fijo=0)
plt.show()


# -----------------------------------------------------------------------------
# PASO 1: BLOQUE TIA - b(t) (Aún en Banda Base Completa antes del Mixer)
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(senial_alineada[:200], '--', label="Señal PAM-4 Ideal (Sin distorsión)", alpha=0.7)
plt.plot(senial_tia[:200], label="Salida del TIA (LPF 3º Orden, 100 GHz)", linewidth=2)
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.title('Paso 1: Efecto del TIA comparado contra la Señal PAM-4 Ideal')
plt.grid(True)
plt.legend()
plt.show()

lag_tia = obtener_instante_optimo(senial_tia, senial_alineada)
graficar_diagrama_ojo(senial_tia, sps, "Paso 1: Salida del TIA", lag_fijo=lag_tia)
#graficar_constelacion(senial_tia, sps, senial_alineada, "Paso 1: Salida del TIA", lag_fijo=lag_tia)
plt.show()


# -----------------------------------------------------------------------------
# PASO 2: BLOQUE MEZCLADOR EN CUADRATURA (Comparado contra I y Q Ideales)
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(I_mixer_ideal[:200], '--', label="Rama I Ideal (Mixer sin distorsión)", alpha=0.7)
plt.plot(IF_I[:200], label="Salida Mixer Real (Rama I con errores y fugas)", linewidth=1.8)
plt.title('Paso 2: Efecto del Mezclador en Cuadratura comparado con Ramas I y Q Ideales')
plt.ylabel('Amplitud (I)')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(Q_mixer_ideal[:200], '--', label="Rama Q Ideal (Mixer sin distorsión)", alpha=0.7)
plt.plot(IF_Q[:200], label="Salida Mixer Real (Rama Q con errores y fugas)", linewidth=1.8, color='orange')
plt.xlabel('Muestras')
plt.ylabel('Amplitud (Q)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

lag_mixer = obtener_instante_optimo(IF_I, I_mixer_ideal)
t_sinc_mix = t - (lag_mixer / fs)
reconst_paso2 = 2 * (IF_I * np.cos(2 * np.pi * f_lo * t_sinc_mix) + IF_Q * np.sin(2 * np.pi * f_lo * t_sinc_mix))

graficar_diagrama_ojo(IF_I, sps, "Paso 2: Salida del Mixer (Rama I)", lag_fijo=lag_mixer)
#graficar_constelacion(np.real(reconst_paso2), sps, senial_alineada, "Paso 2: Reconstrucción tras Mixer", lag_fijo=lag_mixer)
plt.show()


# -----------------------------------------------------------------------------
# PASO 3: BLOQUE TRACK & HOLD CON BANDWIDTH MISMATCH - d_I(t), d_Q(t)
# -----------------------------------------------------------------------------
# 3.A) Efecto Individual del Bandwidth Mismatch (Comparado contra Sub-bandas I/Q Ideales)
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(I_subbanda_ideal[:200], '--', label="Sub-banda I Ideal (Fase cero, 48 GHz)", alpha=0.7)
plt.plot(I_solo_bw[:200], label=f"Rama I T&H Causal (BW nominal = {BW_th_I/1e9:.1f} GHz)", linewidth=1.8)
plt.title(f'Paso 3.A: Efecto Individual del T&H y Bandwidth Mismatch ($\\Delta B_d = {delta_Bd_pct*100:+.0f}\\%$)')
plt.ylabel('Amplitud (I)')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(Q_subbanda_ideal[:200], ':', label="Sub-banda Q Ideal (Sin distorsión, 48 GHz)", alpha=0.7)
plt.plot(Q_solo_bw_nom[:200], '--', label=f"Rama Q T&H sin mismatch ({BW_th_I/1e9:.1f} GHz)", alpha=0.8)
plt.plot(Q_solo_bw_mis[:200], label=f"Rama Q T&H con mismatch ({BW_th_Q/1e9:.1f} GHz)", linewidth=1.8, color='orange')
plt.xlabel('Muestras')
plt.ylabel('Amplitud (Q)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

lag_th_solo = obtener_instante_optimo(I_solo_bw, I_subbanda_ideal)
t_sinc_3a = t - (lag_th_solo / fs)
reconst_3a = 2 * (I_solo_bw * np.cos(2 * np.pi * f_lo * t_sinc_3a) + Q_solo_bw_mis * np.sin(2 * np.pi * f_lo * t_sinc_3a))

graficar_diagrama_ojo(Q_solo_bw_mis, sps, f"Paso 3.A: Efecto Individual BW Mismatch (Rama Q, BW={BW_th_Q/1e9:.1f} GHz)", lag_fijo=lag_th_solo)
#graficar_constelacion(np.real(reconst_3a), sps, senial_alineada, "Paso 3.A: Reconstrucción Solo BW Mismatch", lag_fijo=lag_th_solo)
plt.show()

# 3.B) Efecto Acumulado hasta el T&H (Comparado contra Sub-bandas I/Q Ideales)
# plt.figure(figsize=(10, 5))
# plt.subplot(2, 1, 1)
# plt.plot(I_subbanda_ideal[:200], '--', label="Sub-banda I Ideal (Sin distorsiones)", alpha=0.7)
# plt.plot(out_I_th[:200], label=f"Salida T&H Acumulada I (BW={BW_th_I/1e9:.1f} GHz)", linewidth=2)
# plt.title('Paso 3.B: Efecto Acumulado hasta T&H (TIA + Mixer + BW Mismatch) vs Sub-bandas Ideales')
# plt.ylabel('Amplitud (I)')
# plt.grid(True)
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(Q_subbanda_ideal[:200], ':', label="Sub-banda Q Ideal (Sin distorsiones)", alpha=0.7)
# plt.plot(out_Q_th_nom[:200], '--', label=f"Salida T&H Acumulada Q sin mismatch ({BW_th_I/1e9:.1f} GHz)", alpha=0.8)
# plt.plot(out_Q_th[:200], label=f"Salida T&H Acumulada Q con mismatch ({BW_th_Q/1e9:.1f} GHz)", linewidth=2, color='orange')
# plt.xlabel('Muestras')
# plt.ylabel('Amplitud (Q)')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

lag_th_acum = obtener_instante_optimo(out_I_th, I_subbanda_ideal)
t_sinc_3b = t - (lag_th_acum / fs)
# reconst_3b = 2 * (out_I_th * np.cos(2 * np.pi * f_lo * t_sinc_3b) + out_Q_th * np.sin(2 * np.pi * f_lo * t_sinc_3b))

# graficar_diagrama_ojo(out_Q_th, sps, "Paso 3.B: Salida T&H Acumulada con BW Mismatch (Rama Q)", lag_fijo=lag_th_acum)
# graficar_constelacion(out_I_th, sps, I_subbanda_ideal, "Paso 3.B: Acumulada tras T&H (Plano I vs Q)", senial_Q=out_Q_th, ref_Q=Q_subbanda_ideal, lag_fijo=lag_th_acum)
# graficar_constelacion(np.real(reconst_3b), sps, senial_alineada, "Paso 3.B: Reconstrucción Acumulada tras T&H", lag_fijo=lag_th_acum)
# plt.show()


# -----------------------------------------------------------------------------
# PASO 4: BLOQUE TIME SKEW INTRABANDA (tau_IB) Y CADENA COMPLETA FINAL
# -----------------------------------------------------------------------------
# 4.A) Efecto Individual del Skew Intrabanda (Comparado contra Sub-bandas I/Q Ideales)
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(I_subbanda_ideal[:150], '--', label="Sub-banda I Ideal (Sin skew)", alpha=0.7)
plt.plot(I_solo_skew[:150], label=f"Sub-banda I con skew ({tau_I*1e12:.2f} ps)", linewidth=1.8)
plt.title(f'Paso 4.A: Efecto Individual del Skew Intrabanda ($\\tau_{{IB}} = {tau_IB*1e12:.2f}$ ps) sobre Sub-bandas Ideales')
plt.ylabel('Amplitud (I)')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(Q_subbanda_ideal[:150], '--', label="Sub-banda Q Ideal (Sin skew)", alpha=0.7)
plt.plot(Q_solo_skew[:150], label=f"Sub-banda Q con skew (+{tau_Q*1e12:.2f} ps)", linewidth=1.8, color='orange')
plt.xlabel('Muestras')
plt.ylabel('Amplitud (Q)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

reconst_4a = 2 * (I_solo_skew * np.cos(2 * np.pi * f_lo * t) + Q_solo_skew * np.sin(2 * np.pi * f_lo * t))
graficar_diagrama_ojo(I_solo_skew, sps, "Paso 4.A: Efecto Individual Skew Intrabanda (Rama I)", lag_fijo=0)
graficar_constelacion(I_solo_skew, sps, I_subbanda_ideal, "Paso 4.A: Solo Skew Intrabanda (Plano I vs Q)", senial_Q=Q_solo_skew, ref_Q=Q_subbanda_ideal, lag_fijo=0)
graficar_constelacion(np.real(reconst_4a), sps, senial_alineada, "Paso 4.A: Reconstrucción Solo Skew Intrabanda", lag_fijo=0)
plt.show()

# 4.B) Cadena Completa Final: Ramas I y Q con TODOS los efectos vs Ramas I y Q Ideales
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(I_subbanda_ideal[:200], '--', label="Sub-banda I Ideal (Sin distorsiones del ADC)", alpha=0.7)
plt.plot(out_I_total[:200], label="Señal Final a Ecualizar (Rama I - Todos los efectos)", linewidth=2)
plt.title('Paso 4.B: Comparación Final — Sub-bandas I y Q Ideales vs Señales con Todas las Distorsiones')
plt.ylabel('Amplitud (I)')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(Q_subbanda_ideal[:200], '--', label="Sub-banda Q Ideal (Sin distorsiones del ADC)", alpha=0.7)
plt.plot(out_Q_total[:200], label="Señal Final a Ecualizar (Rama Q - Todos los efectos)", linewidth=2, color='orange')
plt.xlabel('Muestras')
plt.ylabel('Amplitud (Q)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Ojo y Constelación Final de las Sub-bandas I y Q
graficar_diagrama_ojo(out_I_total, sps, "Paso 4.B: Señal Final a Ecualizar con TODOS los efectos (Rama I)", lag_fijo=lag_th_acum)
graficar_constelacion(out_I_total, sps, I_subbanda_ideal, "Paso 4.B: Señal Final solo Rama I vs Ideal", lag_fijo=lag_th_acum)

graficar_diagrama_ojo(out_Q_total, sps, "Paso 4.B: Señal Final a Ecualizar con TODOS los efectos (Rama Q)", lag_fijo=lag_th_acum)
graficar_constelacion(out_I_total, sps, I_subbanda_ideal, "Paso 4.B: Cadena Completa con TODOS los efectos (Plano I vs Q)", senial_Q=out_Q_total, ref_Q=Q_subbanda_ideal, lag_fijo=lag_th_acum)
plt.show()

# Reconstrucción Final PAM-4 al final de toda la cadena (con reloj ideal sincronizado)
reconst_final = 2 * (out_I_total * np.cos(2 * np.pi * f_lo * t_sinc_3b) + out_Q_total * np.sin(2 * np.pi * f_lo * t_sinc_3b))
graficar_diagrama_ojo(np.real(reconst_final), sps, "Paso 4.B: Reconstrucción Final (Todos los efectos)", lag_fijo=lag_th_acum)
graficar_constelacion(np.real(reconst_final), sps, senial_alineada, "Paso 4.B: Reconstrucción Final (Todos los efectos)", lag_fijo=lag_th_acum)
plt.show()