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
    Si causal=False, aplica filtrado de fase cero (sosfiltfilt) útil para la referencia ideal.
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
    mediante correlación cruzada con la referencia ideal.
    """
    corr = np.correlate(senial, ref_ideal, mode='full')
    lag = int(np.argmax(np.abs(corr)) - (len(ref_ideal) - 1))
    return max(0, lag)

def obtener_muestras_simbolo(senial, sps, ref_ideal, lag_fijo=None):
    """
    Muestrea la señal aplicando un reloj ideal periódico (período exacto T = sps muestras)
    sincronizado con el instante de punto máximo del símbolo.
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
    Grafica el diagrama de ojo sincronizando el reloj de trazado con el punto
    máximo del símbolo para centrar la apertura.
    """
    if lag_fijo is not None:
        lag = lag_fijo
    elif ref_ideal is not None:
        lag = obtener_instante_optimo(senial, ref_ideal)
    else:
        lag = 0

    muestras_por_traza = sps * n_simbolos_pantalla
    inicio_estable = 15 * sps + lag
    senial_estable = senial[inicio_estable:]
    
    n_trazas_posibles = len(senial_estable) // muestras_por_traza
    n_trazas = min(n_trazas, n_trazas_posibles)
    
    plt.figure(figsize=(8, 4))
    t_eje = np.arange(muestras_por_traza) / sps
    
    for i in range(n_trazas):
        idx_inicio = i * muestras_por_traza
        segmento = senial_estable[idx_inicio : idx_inicio + muestras_por_traza]
        plt.plot(t_eje, segmento, color='blue', alpha=0.04)
        
    for k in range(n_simbolos_pantalla):
        plt.axvline(k, color='red', linestyle='--', alpha=0.35, linewidth=0.9, label='Instante Clock Ideal' if k == 0 else "")
        
    plt.title(f"Diagrama de Ojo - {titulo}")
    plt.xlabel("Tiempo [Símbolos $T$]")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.legend(loc='upper right')

def graficar_constelacion(senial_I, sps, ref_I, titulo, senial_Q=None, ref_Q=None, lag_fijo=None, cmap='jet'):
    """
    Grafica la constelación de los símbolos recibidos coloreados por densidad de puntos (Heatmap con Gaussian KDE).
    """
    muestras_I = obtener_muestras_simbolo(senial_I, sps, ref_I, lag_fijo=lag_fijo)
    
    plt.figure(figsize=(6.5, 6))
    
    if senial_Q is not None:
        if ref_Q is None:
            ref_Q = ref_I
        muestras_Q = obtener_muestras_simbolo(senial_Q, sps, ref_Q, lag_fijo=lag_fijo)
        n_min = min(len(muestras_I), len(muestras_Q))
        muestras_I, muestras_Q = muestras_I[:n_min], muestras_Q[:n_min]
        
        # 1. Cálculo de Densidad 2D con Gaussian KDE
        xy = np.vstack([muestras_I, muestras_Q])
        z = gaussian_kde(xy)(xy)
        
        # 2. Ordenar puntos para que los más densos (rojos) se dibujen al frente
        idx = z.argsort()
        x, y, z = muestras_I[idx], muestras_Q[idx], z[idx]
        
        # 3. Scatter plot coloreado por densidad
        sc = plt.scatter(x, y, c=z, cmap=cmap, s=8, edgecolors='none')
        cbar = plt.colorbar(sc)
        cbar.set_label('Densidad de símbolos')
        
        plt.xlabel("In phase")
        plt.ylabel("Quadrature")
        
    else:
        # Caso 1D (PAM-4): Agregamos jitter aleatorio en Y para formar nubes 2D de densidad
        np.random.seed(42)
        y_jitter = np.random.normal(0, 0.08, size=len(muestras_I))
        
        xy = np.vstack([muestras_I, y_jitter])
        z = gaussian_kde(xy)(xy)
        
        idx = z.argsort()
        x, y, z = muestras_I[idx], y_jitter[idx], z[idx]
        
        sc = plt.scatter(x, y, c=z, cmap=cmap, s=12, edgecolors='none')
        cbar = plt.colorbar(sc)
        cbar.set_label('Densidad de símbolos')
        
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
    plt.show()


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
# RECONSTRUCCIÓN DE SEÑAL (SR) CON RELOJ IDEAL SINCRONIZADO AL MÁXIMO
# =============================================================================

# 1) Reconstrucción Ideal (Fase cero: instante óptimo lag = 0)
reconst_ideal = 2 * (I_subbanda_ideal * np.cos(2 * np.pi * f_lo * t) + 
                     Q_subbanda_ideal * np.sin(2 * np.pi * f_lo * t))

graficar_diagrama_ojo(np.real(reconst_ideal), sps, "Reconstrucción Ideal", lag_fijo=0)
plt.show()
graficar_constelacion(np.real(reconst_ideal), sps, senial_alineada, "Reconstrucción Ideal", lag_fijo=0)

# 2) Sincronización del Reloj Ideal para la cadena causal (TIA + Mixer + T&H)
lag_reloj_ideal = obtener_instante_optimo(I_solo_bw, I_subbanda_ideal)
t_sinc = t - (lag_reloj_ideal / fs)

# Reconstrucción final con distorsiones acumuladas bajo reloj ideal
reconst_final = 2 * (out_I_total * np.cos(2 * np.pi * f_lo * t_sinc) + 
                     out_Q_total * np.sin(2 * np.pi * f_lo * t_sinc))

graficar_diagrama_ojo(np.real(reconst_final), sps, "Reconstrucción Final (Clock Ideal)", lag_fijo=lag_reloj_ideal)
plt.show()
graficar_constelacion(np.real(reconst_final), sps, senial_alineada, "Reconstrucción Final (Clock Ideal)", lag_fijo=lag_reloj_ideal)