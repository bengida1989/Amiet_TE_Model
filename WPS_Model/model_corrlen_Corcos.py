"""
    corrlen_Corcos -- Spanwise correlation length - Corcos model

    Computes the spanwise correlation length using the Corcos model.

    :param freq: Frequency (Hz) at which to compute the correlation length.
    :type freq: float
    :param bc: Corcos constant (dimensionless).
    :type bc: float
    :param Uc: Convection velocity (m/s).
    :type Uc: float

    :return: Spanwise correlation length according to the Corcos model.
    :rtype: float

    :notes:
        - Input variables must be dimensional (i.e., not non-dimensionalized).
        - Pulsation (angular frequency) is computed as: ``omega = 2π * freq``.
        - The correlation length decreases with increasing frequency.

    :references:
        - M. Rozenberg, PhD thesis, 2007-44, ECL, page 161.
        - Corcos, G.M., "Resolution of pressure in turbulence",
            J. Acoust. Soc. Am., 35(2), 192–199, 1963.
        - Corcos, G.M., "The structure of turbulent pressure field in
            boundary-layer flows", J. Fluid Mech., 18, 353–378, 1964.

    :author:
        GENEAU Dominic (Original Fortran code from GAMET Lionel)
        DENG Patrick (Adapation, translation and modularization to Python)

    :history:
        - 2009-08-25 -- Fortran file creation.
        - 2024-02-16 -- Python file creation.
"""

import math
import numpy as np
def corrlen_Corcos(freq: float, bc: float, Uc: float) -> float:
    """
    Computes the spanwise correlation length using the Corcos model.
    The correlation length is computed based on the frequency, Corcos constant,
    Args:
        freq (float): Frequency (Hz) at which to compute the correlation length.
        bc (float): Corcos constant (dimensionless).
        Uc (float): Convection velocity (m/s).

    Returns:
        float: _description_
    """
    omega = 2.0 * math.pi * freq[1:]
    if (omega <= 0).any():
        raise ValueError("All frequency values except the first must be positive and nonzero.")
    corrlen_Corcos_out = bc * Uc / omega
    
    if freq[0] == 0:
        corrlen_Corcos_out = np.concatenate(([0], corrlen_Corcos_out), axis=0)
    else:
        corrlen_Corcos_out = np.concatenate((np.array([bc * Uc/(2*math.pi*freq[0])]) , corrlen_Corcos_out), axis=0)
        
    return corrlen_Corcos_out