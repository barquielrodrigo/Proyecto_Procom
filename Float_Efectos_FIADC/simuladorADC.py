import numpy as np
import sys
sys.path.append("C:/Users/Usuario/Desktop/Valentino/PROCOM/TP1")
from filtro import RaisedCosineFilter # type: ignore
from scipy.signal import hilbert, resample
from scipy import signal

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

def simuladorADC(signal, fs, rolloff, span, sps, fugaOL=None, fugaIO=None, epsilon=0, phi_deg=0, square_mixer=False, jitter_OL=0, jitter_adc=0):
    """Simula distintos efectos de una cadena tipica de FI-ADC.

    Args:
        signal (float array M x 1): senal transmitida
        fs: frecuencia de muestreo de la señal
        rolloff (float): factor de roll off del filtro Raised Cosine.
        span: cantidad de simbolos involucrados con el pulso Raised Cosine.
        sps: muestras por simbolo dentro del Raised Cosine.
        fugaOL: fuga (valor positivo) del mixer hacia la entrada [dB].
        fugaIO: fuga (valor positivo) en el mixer de entrada hacia salida [dB].
        epsilon: error de ganancia en el mixer.
        phi_deg: error de fase del mixer [grados].
        square_mixer: Bandera para determinar si el mixer es de senal cuadrada o sinusoidal. (Por defecto en False)
        jitter_OL: tiempo del jitter para el oscilador local [segundos].
        jitter_adc: tiempo del jitter para el muestreo del ADC [segundos].

    Returns:
        y_inf: subbanda inferior.
        y_sup: subbanda superior.
    """

    s_total_FI = signal
    t = np.arange(len(s_total_FI)) / fs

    BW_tia = 100e9
    B_d = 48e9      # Dejo los valores fijos para que la transmision siempre sea 0-48GHz y 48-96GHz
    f_lo = 48e9

    LO_Port = 0 if fugaOL is None else 10**(-fugaOL/20)
    RF_IF   = 0 if fugaIO is None else 10**(-fugaIO/20)   # Fugas del mixer
    
    phi_rad = np.deg2rad(phi_deg)# Error de fase

    sigma_rw = 2 * np.pi * jitter_OL
    ruido_fase_lo = np.cumsum(np.random.normal(0, sigma_rw, size=len(t)))

    if square_mixer:
        LO = (2 + epsilon/2) * np.sign(np.cos(2 * np.pi * f_lo * t + phi_rad/2 + ruido_fase_lo))
    else:
        LO = (2 + epsilon/2) * np.cos(2 * np.pi * f_lo * t + phi_rad/2 + ruido_fase_lo)

    # --- Etapa 1: TIA ---
    senial_tia, _ = aplicar_lpf_butterworth(s_total_FI, fs, BW_tia, orden=3) # Compuesta

    # --- Etapa 2: Separación de Bandas (LPF externo en BB y HPF externo en HB) ---
    senial_lpf_bb_con_solap, _ = aplicar_lpf_butterworth(senial_tia, fs, fc=48e9, orden=4)
    senial_hpf_con_solap, _    = aplicar_hpf_butterworth(senial_tia, fs, fc=48e9, orden=4)

    # --- Etapa 3: Mixer en la banda superior ---
    RF_solap, _ = aplicar_lpf_butterworth(senial_hpf_con_solap, fs, 100e9, 1)
    post_mix_total = (RF_solap + LO_Port * LO) * LO + RF_IF * RF_solap + LO_Port * LO
    IF_hf_total, _ = aplicar_lpf_butterworth(post_mix_total, fs, 100e9, 1)

    # Jitter de Apertura (Vector de tiempo para el ADC)
    t_ideal = np.arange(len(t)) / fs
    t_jitter_adc = t_ideal + np.random.normal(0, jitter_adc, size=len(t))

    # --- Etapa 4: T&H y Muestreo ADC ---
    out_bb_solo_solap, _ = aplicar_lpf_butterworth(senial_lpf_bb_con_solap, fs, B_d, 4)
    out_bb_total = np.interp(t_jitter_adc, t_ideal, out_bb_solo_solap)
    out_hf_pre_total, _ = aplicar_lpf_butterworth(IF_hf_total, fs, B_d, 4)
    out_hf_total = np.interp(t_jitter_adc, t_ideal, out_hf_pre_total)

    out_bb_total = out_bb_total[::4]
    out_hf_total = out_hf_total[::4] # submuestreo las subbandas 

    return out_bb_total, out_hf_total