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

def simuladorADC(signal, rolloff, span, sps, fugaOL=0, fugaIO=0, epsilon=0, phi_deg=0, square_mixer=True, jitter_OL=0, jitter_adc=0):
    """Simula distintos efectos de una cadena tipica de FI-ADC.

    Args:
        signal (float array M x 1): senal PAM-4 transmitida
        rolloff (float): factor de roll off del filtro Raised Cosine.
        span: cantidad de simbolos involucrados con el pulso Raised Cosine.
        sps: muestras por simbolo dentro del Raised Cosine.
        fugaOL: fuga (valor positivo) del mixer hacia la entrada [dB].
        fugaIO: fuga (valor positivo) en el mixer de entrada hacia salida [dB].
        epsilon: error de ganancia en el mixer.
        phi_deg: error de fase del mixer [grados].
        square_mixer: Bandera para determinar si el mixer es de senal cuadrada o sinusoidal. (Por defecto en True)
        jitter_OL: tiempo del jitter para el oscilador local [segundos].
        jitter_adc: tiempo del jitter para el muestreo del ADC [segundos].




    Returns:
        float: El área total calculada (base por altura).
    """

    ## Transmisor ideal sin modelado en detalle

    # Separación en símbolos pares (Banda Inferior) e impares (Banda Superior)
    simb_pam = signal[0::2]
    simb_pam_hf = signal[1::2]

    # Sobremuestreo de cada sub-banda
    simb_upsampled = np.zeros(len(simb_pam) * sps)
    simb_upsampled[::sps] = simb_pam  
    simb_upsampled_hf = np.zeros(len(simb_pam_hf) * sps)
    simb_upsampled_hf[::sps] = simb_pam_hf  

    # Filtro conformador de cada sub-banda
    rc = RaisedCosineFilter(alpha=rolloff, span=span, sps=sps, rrc=False)
    rc_filter = rc.get_coefficients()
    senial = np.convolve(simb_upsampled, rc_filter, mode='full')       # Banda base
    senial_hf = np.convolve(simb_upsampled_hf, rc_filter, mode='full') # Sub-banda superior

    BW_deseado = 48e9
    Rs = (2 * BW_deseado) / (1 + rolloff)
    fs = Rs * sps

    fc = 48e9
    t = np.arange(len(senial)) / fs
    s2_hilbert = hilbert(senial_hf)
    s2_ssb = np.real(s2_hilbert * np.exp(1j * 2 * np.pi * fc * t)) # Banda superior modulada en SSB

    s_total_FI = senial + s2_ssb

    ## Fin del transmisor

    BW_tia = 100e9
    B_d = 48e9      # Dejo los valores fijos para que la transmision siempre sea 0-48GHz y 48-96GHz
    f_lo = 48e9

    LO_Port = 10**(-fugaOL/20)
    RF_IF = 10**(-fugaIO/20)   # Fugas del mixer
    
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
