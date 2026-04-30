"""
    rdirsupTE -- Radiation integral directivity for supercritical gusts (Trailing Edge Noise Model)

    Computes the radiation integral directivity for supercritical gusts in Amiet's trailing-edge noise model,
    extended with leading-edge backscattering corrections following Roger & Moreau (2005, 2009).

    The far-field acoustic pressure spectral density contribution is related to the radiation integral:

    .. math::

        L = L_1 + L_2

    where:

    - :math:`L_1` corresponds to the **trailing-edge contribution**,
    - :math:`L_2` corresponds to the **leading-edge scattering correction**.

    The intermediary terms are defined as:

    .. math::

        L_2 = H \\left[ (1 - (1+i) \\overline{e^{4ik \\kappa}} \\, F(4\\kappa)) - e^{2iD} + i(D + K_b + \\mu M - \\kappa) G \\right]

    .. math::

        G = (1 + \\epsilon) \\frac{e^{i(D+2\\kappa)} \\sin(D-2\\kappa)}{D-2\\kappa} + (1 - \\epsilon) \\frac{e^{i(D-2\\kappa)} \\sin(D+2\\kappa)}{D+2\\kappa} + \\text{(higher-order corrections)}

    where:
    
    - :math:`D = \\kappa - \\mu \\frac{x_1}{S_0}`,
    - :math:`\\kappa` is the square root of modified aerodynamic wave numbers,
    - :math:`F(\\cdot)` is the complex Fresnel integral.

    The **trailing-edge term** :math:`L_1` is:

    .. math::

        L_1 = i \\, \\frac{e^{2iC}}{C} \\left[ 1 - (1+i) \\overline{F(2B)} + (1+i) e^{-2iC} \\sqrt{\\frac{B}{B-C}} \\overline{F(2(B-C))} \\right]

    where:
    
    - :math:`B = \\alpha K_b + \\mu M + \\kappa`,
    - :math:`C = \\alpha K_b - \\mu \\left( \\frac{x_1}{S_0} - M \\right)`.

    The corrected distance :math:`S_0` is:

    .. math::

        S_0 = \\sqrt{ x_1^2 + \\beta^2 (x_2^2 + x_3^2) }

    with :math:`\\beta^2 = 1 - M^2`.

    :param freq: Frequency array or scalar (Hz).
    :type freq: float or ndarray
    :param x1: Streamwise observer coordinate array or scalar (m).
    :type x1: float or ndarray
    :param x2: Spanwise observer coordinate array or scalar (m).
    :type x2: float or ndarray
    :param x3: Vertical observer coordinate array or scalar (m).
    :type x3: float or ndarray
    :param chord: Airfoil chord (m).
    :type chord: float
    :param Uinf: Free-stream velocity (m/s).
    :type Uinf: float
    :param alpha: Ratio of free-stream velocity to convective velocity (Uinf/Uc).
    :type alpha: float
    :param cinf: Free-stream speed of sound (m/s).
    :type cinf: float

    :return: Radiation integral contribution (complex array matching input shapes).
    :rtype: complex or ndarray of complex

    :notes:
        - Inputs must be **dimensional** (m, m/s, Hz).
        - Fully vectorized: supports scalar or array inputs for frequency and observer coordinates.
        - Correctly handles leading-edge backscatter and trailing-edge correction.
        - Based on Amiet's trailing-edge model extended by Roger and Moreau.

    :references:
        - M. Rozenberg, Ph.D. thesis, 2007-44, École Centrale de Lyon.
        - Roger, M., Moreau, S., "Back-scattering correction and further extensions of Amiet's trailing edge noise model. Part I: Theory", Journal of Sound and Vibration, 286 (2005), 477–506.
        - Roger, M., Moreau, S., "Back-scattering correction and further extensions of Amiet's trailing edge noise model. Part II: Applications", Journal of Sound and Vibration, 323 (2009), 397–425.

    :author:
        GENEAU Dominic (original Fortran code)  
        GAMET Lionel (original Fortran code)  
        DENG Patrick (Python translation, modularization, vectorization)

    :history:
        - 2009-08-25 -- Original FORTRAN file creation.
        - 2024-01-15 -- First Python file adaptation.
        - 2025-04-28 -- Modular rewrite and full vectorization.
"""

import numpy as np
from .calc_fresnel import FresCS2

def rdirsupTE(freq, x1:float, x2:float, x3:float, chord:float=0.3048, Uinf:float=30, Uc:float=24, cinf:float=343) -> complex:
    """
    Vectorized computation of the radiation integral directivity for supercritical gusts
    in trailing-edge noise model, allowing arrays of frequencies and observer locations.

    Args:
        freq (float or ndarray): Frequency (Hz).
        x1 (float or ndarray): Streamwise observer coordinate (m).
        x2 (float or ndarray): Spanwise observer coordinate (m).
        x3 (float or ndarray): Vertical observer coordinate (m).
        chord (float): Airfoil chord (m).
        Uinf (float): Free-stream velocity (m/s).
        Uc (float): Ratio of free-stream velocity to convective velocity (Uinf/Uc).
        cinf (float): Free-stream speed of sound (m/s).

    Returns:
        ndarray: Radiation integral contribution (complex array matching input shape).
    """

    # Convert inputs to arrays
    # freq = np.atleast_1d(freq)
    # x1 = np.atleast_1d(x1)
    # x2 = np.atleast_1d(x2)
    # x3 = np.atleast_1d(x3)

    # # Broadcasting setup to handle different shapes (either scalar or array)
    # freq, x1, x2, x3 = np.broadcast_arrays(freq, x1, x2, x3)
    #===================================================
    # Prepare input variables
    M = Uinf / cinf                         # Mach number
    be2 = 1.0 - M**2                        # 1 - M^2
    omega = 2.0 * np.pi * freq              # Angular frequency
    k = omega / cinf                        # Convective wavenumber
    kc = k * chord                          # Helmholtz convective wavenumber
    Ka = omega / Uinf                       # Acoustic wavenumber
    Kb = Ka * chord / 2.0                   # Aerodynamic wavenumber
    Kby = 0.0                               # No spanwise aerodynamic wavenumber
    mu = Kb * M / be2                       # Mach number in the convective frame
    kapa = np.sqrt(complex(mu**2 - Kby**2 / be2, 0.0))  # Complex square root
    exp4ikapa = np.exp(4.0j * kapa)             # Exponential term
    alpha = Uc / Uinf   # ≈ 0.6–0.8, per Roger & Moreau (2005)                       # Ratio of free-stream velocity to convective velocity (NOTE: NOT ANGLE OF ATTACK!)

    epsi = 1.0 / np.sqrt(1.0 + 1.0 / (4.0 * mu))    # Epsilon term
    S0 = np.sqrt(x1**2 + be2 * (x2**2 + x3**2))     # Distance from the source to the observer
    #===================================================
    # Constructing the variable G from B, C, D
    BB = alpha * Kb + mu * M + kapa
    CC = alpha * Kb - mu * (x1 / S0 - M)
    DD = kapa - mu * x1 / S0

    Theta1_2 = BB / (Kb + mu * M + kapa)
    HH = (1.0 + 1.0j) * (1.0 - Theta1_2) * np.conj(exp4ikapa) / (2.0 * np.sqrt(BB * np.pi) * (alpha - 1.0) * Kb)
    
    # G Term as the sum of two parts
    GG = (1.0 + epsi) * np.exp(1j * (DD + 2.0 * kapa)) * np.sin(DD - 2.0 * kapa) / (DD - 2.0 * kapa)
    GG += (1.0 - epsi) * np.exp(1j * (DD - 2.0 * kapa)) * np.sin(DD + 2.0 * kapa) / (DD + 2.0 * kapa)
    
    zfre = FresCS2(4.0 * kapa)
    ztmp = (1.0 + 1.0j) * np.conj(exp4ikapa) * zfre
    # Adding G term 
    GG += (1.0 + epsi) * 0.5 * np.conj(ztmp) / (DD - 2.0 * kapa) - (1.0 - epsi) * 0.5 * ztmp / (DD + 2.0 * kapa)

    ztmp = (1.0 - epsi) * (1.0 + 1.0j) / (DD + 2.0 * kapa) - (1.0 + epsi) * (1.0 - 1.0j) / (DD - 2.0 * kapa)
    GG += 0.5 * np.exp(2.0j * DD) * np.sqrt(2.0 * kapa / DD) * np.conj(FresCS2(2.0 * DD)) * ztmp
    #===================================================
    # Leading edge contribution L2
    ztmp = exp4ikapa * (1.0 - (1.0 + 1.0j) * np.conj(zfre))
    xtmp = np.real(ztmp)
    ytmp = np.imag(ztmp) * epsi
    ztmp = complex(xtmp, ytmp)

    L2 = HH * (ztmp - np.exp(2.0j * DD) + 1j * (DD + Kb + mu * M - kapa) * GG)
    #===================================================
    # # Trailing edge contribution L1
    zfre = np.conj(FresCS2(2.0 * BB))
    L1   = 1.0 - (1.0 + 1j) * zfre

    # Branch on BB == CC
    if np.isclose(BB - CC, 0.0):
        L1 += (1.0 + 1j) \
            * np.exp(-2.0j * CC) \
            * np.sqrt(BB) \
            * 2.0/ np.sqrt(np.pi)
    else:
        zfre2 = np.conj(FresCS2(2.0 * (BB - CC)))
        L1 += (1.0 + 1j) \
            * np.exp(-2.0j * CC) \
            * np.sqrt(BB / (BB - CC)) \
            * zfre2
    #===================================================
    # Finalizing computation
    L1 = L1 * 1j * np.exp(2.0j * CC) / CC
    #===================================================
    # The final result is the sum of L1 and L2 for the Leading Edge and Trailing Edge contributions
    
    rdirsupTE_out =  L1 + L2

    return rdirsupTE_out
