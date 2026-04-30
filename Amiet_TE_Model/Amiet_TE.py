"""
    AmietTE -- Trailing-edge noise prediction using Amiet's analytical model

    Implements Amiet's trailing-edge noise model with extensions to account for 
    coherence effects and empirical/analytical wall-pressure spectrum models. This class 
    calculates the far-field sound pressure spectral density (PSD) and radiation directivity 
    due to turbulent boundary layer–trailing edge interaction for a finite-span airfoil.

    The total acoustic power spectral density at a given observer location is computed as:

    .. math::

        S_{pp}(f, \mathbf{x}) = \\left( \\frac{k_c z}{4 \\pi S_0} \\right)^2 (2 L_y) \\left| D(f, \\mathbf{x}) \\right|^2 \\, \\Phi_{pp}^{\\mathrm{total}}(f) \\, L_y(f)

    where the total wall-pressure spectrum sums both airfoil sides in linear (power) units before
    conversion to dB:

    .. math::

        \\Phi_{pp}^{\\mathrm{total}}(f) = \\Phi_{pp}^{\\mathrm{SS}}(f) + \\Phi_{pp}^{\\mathrm{PS}}(f)

    with SS denoting the suction side and PS the pressure side.  Each side may use an independent
    model or data file (analytical models, experiment, or simulation).  When only one side is
    specified the contribution from the missing side is assumed to be zero.

    Parameters:
        - :math:`k_c` is the chordwise acoustic wavenumber,
        - :math:`S_0` is the retarded distance between the airfoil and observer,
        - :math:`D(f, \\mathbf{x})` is the complex-valued directivity,
        - :math:`\\Phi_{pp}^{\\mathrm{total}}(f)` is the summed (SS + PS) wall-pressure spectrum,
        - :math:`L_y(f)` is the spanwise coherence length.

    The model includes:
        - Independent suction-side and pressure-side WPS inputs (analytical or file-based),
        - Linear summation of both sides prior to Spp computation,
        - User-selectable wall-pressure spectrum models (Goody, Rozenberg, Lee, or file-based),
        - Coherence models (Corcos, experimental, or user-defined),
        - Support for finite span corrections,
        - Frequency interpolation when required.

    :param observers: List of 3D observer coordinates [[x1, x2, x3], ...].
    :type observers: list
    :param data: One-row pandas Series containing boundary layer parameters (e.g., Uref, delta, tau_w).
    :type data: pandas.Series
    :param freq: Optional frequency array (Hz). If None, defaults to logspace from 1–30000 Hz.
    :type freq: list or ndarray
    :param phipp_method: Method to compute the **suction-side** wall-pressure spectrum.
        Options: ['rozenberg', 'goody', 'lee', 'simulation', 'experiment'].
    :type phipp_method: str
    :param phipp_method_PS: Method to compute the **pressure-side** wall-pressure spectrum.
        Same options as *phipp_method*.  If ``None`` the pressure-side contribution is set to zero.
    :type phipp_method_PS: str or None
    :param coh_method: Method to compute spanwise coherence. Options: ['corcos', 'simulation', 'experiment'].
    :type coh_method: str
    :param spectra: Path to .csv or .dat file with suction-side WPS data
        (required when *phipp_method* is 'experiment' or 'simulation').
    :type spectra: str or None
    :param spectra_PS: Path to .csv or .dat file with pressure-side WPS data
        (required when *phipp_method_PS* is 'experiment' or 'simulation').
    :type spectra_PS: str or None
    :param coherence: Path to file with coherence length data (if using 'experiment' or 'simulation').
    :type coherence: str or None
    :param cinf: Speed of sound (m/s). Default is 343 m/s.
    :type cinf: float
    :param span: Airfoil span (m). Default is 0.5715 m.
    :type span: float
    :param chord: Airfoil chord (m). Default is 0.3048 m.
    :type chord: float

    :attributes:
        - freq (ndarray): Frequency array used for computation.
        - Spp (ndarray): PSD results in dB, shape (n_freqs, n_observers).
        - Directivity (ndarray): Complex-valued directivity matrix (same shape).
        - Phipp_SS (ndarray): Suction-side wall-pressure spectrum in dB.
        - Phipp_PS (ndarray): Pressure-side wall-pressure spectrum in dB (zeros if not provided).
        - Phipp (ndarray): Total (SS + PS, linear-summed) wall-pressure spectrum in dB.
        - Ly (ndarray): Coherence lengths across frequency.

    :raises:
        - ValueError: If required files do not exist or if input validation fails.

    :references:
        - Amiet, R.K., “Noise due to turbulent flow past a trailing edge,” J. Sound Vib., 47(3), 387–393, 1976.
        - Rozenberg, M., “PhD Thesis 2007-44,” École Centrale de Lyon.
        - Roger, M. and Moreau, S., “Back-scattering correction and further extensions of Amiet's trailing-edge noise model,” JSV, 2005–2009.
        - Goody, M. “Empirical spectral model of surface pressure fluctuations,” AIAA J, 42(9), 1788–1794, 2004.

    :author:
        GENEAU Dominic (original implementation)  
        GAMET Lionel (Fortran extensions)  
        DENG Patrick (Python adaptation, spectral model integration, vectorization)

    :history:
        - 2009–2010 -- Original FORTRAN implementation.
        - 2024-01-15 -- Initial Python rewrite.
        - 2025-04-28 -- Modularization, model integration, and documentation refinement.
        - 2025-04-29 -- Two-sided WPS summation (suction + pressure side) added.
"""

import math
import os
import numpy as np
import pandas as pd
from .calc_rdirsupTE import rdirsupTE
from .utils_checkdata import check_data
from WPS_Model import phipp_models

class AmietTE:
    def __init__(self, observers:list, data:pd.Series, freq:list = None, \
        phipp_method:str='rozenberg', phipp_method_PS:str=None,
        coh_method:str='corcos', spectra:str=None, spectra_PS:str=None, coherence:str=None,\
        cinf:float = 343, span:float = 0.5715, chord:float = 0.3048, Uc:float = 24, struhal:bool = True):
        """
        Initialize the AmietTE class.

        Parameters:
            observers (list): List of observer coordinates.
            data (pd.Series): Series containing the boundary layer input parameters.
            freq (list): Frequency array or scalar (Hz).
            phipp_method (str): WPS method for the suction side.
                Options: 'goody', 'rozenberg', 'lee', 'experiment', 'simulation'.
            phipp_method_PS (str or None): WPS method for the pressure side.
                Same options as phipp_method.  Pass None to ignore pressure-side contribution.
            coh_method (str): Method to be used for coherence calculations.
            spectra (str): Path to the suction-side WPS file (used when phipp_method is
                'simulation' or 'experiment').
            spectra_PS (str): Path to the pressure-side WPS file (used when phipp_method_PS is
                'simulation' or 'experiment').
            coherence (str): Path to the coherence file (used when coh_method is
                'simulation' or 'experiment').
            cinf (float): Free-stream speed of sound (m/s), default is 343 m/s.
            span (float): Span of the airfoil (m), default is 0.5715 m.
            chord (float): Airfoil chord (m), default is 0.3048 m.
            Uc (float): Convection velocity (m/s), default is 24 m/s.
        """
        print(f"\n{'Computing Amiets Trailing Edge Model':=^100}\n")
        # Store the user-supplied frequency array as the fallback for purely analytical runs.
        # self.freq is NOT set here — compute_spectra always defines it from the loaded data
        # or from this default when both sides are analytical.
        if freq is None:
            freq = np.logspace(0, 4.5, num=100, base=10.0)  # 1 Hz – ~30 kHz
        self._freq_default = np.asarray(freq, dtype=float)
        self.observers = observers
        self.phipp_method = phipp_method.lower()
        self.phipp_method_PS = phipp_method_PS.lower() if phipp_method_PS is not None else None
        self.coh_method = coh_method.lower()
        self.data = data
        self.spectra_path = spectra if self.phipp_method in ['simulation', 'experiment'] else None
        self.spectra_path_PS = spectra_PS if (self.phipp_method_PS is not None and self.phipp_method_PS in ['simulation', 'experiment']) else None
        self.coherence_path = coherence if self.coh_method in ['simulation', 'experiment'] else None
        self.struhal = struhal
        print('Model Settings:')
        print(f"    Default frequency range: {self._freq_default[0]:.2f} Hz to {self._freq_default[-1]:.2f} Hz "
              f"(overridden by file data when experiment/simulation WPS is used)")
        print(f"    Chord: {chord:.4f} m")
        print(f"    Span: {span:.4f} m")
        print(f"    Speed of sound: {cinf:.2f} m/s")
        print(f"    Wall-pressure spectrum method (suction side): {phipp_method}")
        if self.phipp_method_PS is not None:
            print(f"    Wall-pressure spectrum method (pressure side): {phipp_method_PS}")
        else:
            print(f"    Wall-pressure spectrum method (pressure side): not provided — contribution set to zero")
        print(f"    Coherence method: {coh_method}")
        if self.phipp_method in ['simulation', 'experiment']:
            print(f"    Spectra file (suction side): {spectra}")
        if self.phipp_method_PS in ['simulation', 'experiment']:
            print(f"    Spectra file (pressure side): {spectra_PS}")
        if self.coh_method in ['simulation', 'experiment']:
            print(f"    Coherence file: {coherence}")

        # Checking input data validity
        try:
            check_data(data)
        except ValueError as e:
            raise ValueError(f"Data check failed: {e}")
        
        self.chord = data['chord'] if 'chord' in data else chord
        self.span = data['span'] if 'span' in data else span
        self.cinf = data['cinf'] if 'cinf' in data else cinf
        self.Uc = data['Uc'] if 'Uc' in data else Uc
        
    def compute_spectra(self):
        """
        Computes the wall-pressure spectra and spanwise coherence for both the suction side (SS)
        and pressure side (PS) of the airfoil, then sums them in linear (power) units.

        The summation is performed before the dB conversion so that the total spectrum correctly
        represents the incoherent addition of both side contributions:

        .. math::

            \\Phi_{pp}^{\\mathrm{total}}(f) = \\Phi_{pp}^{\\mathrm{SS}}(f) + \\Phi_{pp}^{\\mathrm{PS}}(f)

        Both ``Phipp_SS`` and ``Phipp_PS`` are stored in **dB** for diagnostic inspection;
        ``Phipp`` (used in ``calculate_Spp``) holds the **linear-summed** value converted to dB.

        Raises:
            ValueError: If 'experiment'/'simulation' mode is selected but the corresponding
                file path is missing or does not exist.
        """
        # ------------------------------------------------------------------ #
        # Instantiate analytical models (only when at least one side needs it)
        # ------------------------------------------------------------------ #
        need_analytical_SS = self.phipp_method not in ['simulation', 'experiment']
        need_analytical_PS = (self.phipp_method_PS is not None and
                              self.phipp_method_PS not in ['simulation', 'experiment'])
        need_coh_analytical = self.coh_method not in ['simulation', 'experiment']

        if need_analytical_SS or need_coh_analytical:
            phipp_model_SS = phipp_models.PhippModels(
                self.phipp_method, self.data,
                normalization='frequency', spectral_normalization=2e-5, verbose=True)
        if need_analytical_PS:
            phipp_model_PS = phipp_models.PhippModels(
                self.phipp_method_PS, self.data,
                normalization='frequency', spectral_normalization=2e-5, verbose=True)

        print(f"\n{'Extracting phipp and coherence':.^60}\n")

        # ------------------------------------------------------------------ #
        # Suction-side WPS
        # Defines the master frequency grid: file data takes priority over the
        # default logspace.  Analytical models evaluate on self._freq_default.
        # ------------------------------------------------------------------ #
        if self.phipp_method in ['simulation', 'experiment']:
            print(f'----> Extracting suction-side phipp from {self.phipp_method} data.')
            if not os.path.exists(self.spectra_path):
                raise ValueError(f"Suction-side spectra file does not exist: {self.spectra_path}")
            freq_phipp_SS, Phipp_SS_dB = self.read_spectra(self.spectra_path)
            # Convert Strouhal number (St = f·chord/Uc) → Hz
            freq_phipp_SS = freq_phipp_SS * self.Uc / self.chord if self.struhal else freq_phipp_SS
            print(f'    Suction-side: {len(freq_phipp_SS)} points, '
                  f'{freq_phipp_SS[0]:.2f}–{freq_phipp_SS[-1]:.2f} Hz (converted from Strouhal).')
        else:
            print(f'----> Extracting suction-side phipp from {self.phipp_method} model.')
            freq_phipp_SS = self._freq_default.copy()
            Phipp_SS_dB   = phipp_model_SS.compute_phipp(freq_phipp_SS)

        # ------------------------------------------------------------------ #
        # Finalise the master frequency grid immediately after SS is resolved,
        # so the PS branch (analytical or file) always works on the same axis.
        # ------------------------------------------------------------------ #
        self.freq = freq_phipp_SS.copy()
        if math.isclose(self.freq[0], 0.0):
            self.freq[0] = 1e-3

        # ------------------------------------------------------------------ #
        # Pressure-side WPS — always brought onto self.freq before summing
        # ------------------------------------------------------------------ #
        if self.phipp_method_PS is None:
            print('----> No pressure-side WPS method provided — pressure-side contribution set to zero.')
            Phipp_PS_dB = np.full_like(Phipp_SS_dB, -np.inf)

        elif self.phipp_method_PS in ['simulation', 'experiment']:
            print(f'----> Extracting pressure-side phipp from {self.phipp_method_PS} data.')
            if not os.path.exists(self.spectra_path_PS):
                raise ValueError(f"Pressure-side spectra file does not exist: {self.spectra_path_PS}")
            freq_phipp_PS, Phipp_PS_dB_raw = self.read_spectra(self.spectra_path_PS)
            freq_phipp_PS = freq_phipp_PS * self.Uc / self.chord  if self.struhal else freq_phipp_PS
            print(f'    Pressure-side: {len(freq_phipp_PS)} points, '
                  f'{freq_phipp_PS[0]:.2f}–{freq_phipp_PS[-1]:.2f} Hz (converted from Strouhal).')
            # Interpolate onto master (SS) grid in linear power units
            if not np.array_equal(freq_phipp_PS, self.freq):
                print('    Interpolating pressure-side spectra onto suction-side frequency grid.')
                PS_lin = np.where(np.isfinite(Phipp_PS_dB_raw),
                                  (2e-5)**2 * 10.0**(Phipp_PS_dB_raw / 10.0), 0.0)
                PS_lin_interp = np.interp(self.freq, freq_phipp_PS, PS_lin,
                                          left=0.0, right=0.0)
                Phipp_PS_dB = np.where(PS_lin_interp > 0,
                                       10.0 * np.log10(PS_lin_interp / (2e-5)**2),
                                       -np.inf)
            else:
                Phipp_PS_dB = Phipp_PS_dB_raw

        else:
            # Analytical PS: evaluate directly on the already-finalised master grid
            print(f'----> Extracting pressure-side phipp from {self.phipp_method_PS} model.')
            Phipp_PS_dB = phipp_model_PS.compute_phipp(self.freq)

        # ------------------------------------------------------------------ #
        # Coherence — computed on self.freq (master grid, already finalised)
        # ------------------------------------------------------------------ #
        if self.coh_method in ['simulation', 'experiment']:
            print(f'----> Extracting coherence from {self.coh_method} data.')
            if not os.path.exists(self.coherence_path):
                raise ValueError(f"Coherence file does not exist: {self.coherence_path}")
            freq_Ly, Ly_raw = self.read_spectra(self.coherence_path)
            freq_Ly = freq_Ly * self.Uc / self.chord  if self.struhal else freq_Ly
            Ly = np.interp(self.freq, freq_Ly, Ly_raw)
            print(f'    Coherence interpolated onto {len(self.freq)}-point frequency grid.')
        else:
            print(f'----> Extracting coherence from {self.coh_method} model.')
            Ly = phipp_model_SS.compute_corrlen(self.freq, bc=0.52, Uc=self.Uc)
            print(f'    Coherence computed on {len(self.freq)}-point frequency grid.')

        # ------------------------------------------------------------------ #
        # Linear summation of suction-side and pressure-side WPS
        # ------------------------------------------------------------------ #
        # Convert from dB back to linear power (Pa²/Hz), sum, then back to dB.
        p_ref_sq = (2e-5)**2
        Phipp_SS_lin = np.where(np.isfinite(Phipp_SS_dB),
                                p_ref_sq * 10.0**(Phipp_SS_dB / 10.0), 0.0)
        Phipp_PS_lin = np.where(np.isfinite(Phipp_PS_dB),
                                p_ref_sq * 10.0**(Phipp_PS_dB / 10.0), 0.0)

        Phipp_total_lin = Phipp_SS_lin + Phipp_PS_lin
        Phipp_total_lin = np.where(Phipp_total_lin > 0, Phipp_total_lin, np.finfo(float).tiny)
        Phipp_total_dB  = 10.0 * np.log10(Phipp_total_lin / p_ref_sq)

        print('----> Suction-side and pressure-side WPS summed in linear (power) units.')

        # ------------------------------------------------------------------ #
        # Store results — self.freq already set above after Strouhal conversion
        # ------------------------------------------------------------------ #
        self.Phipp_SS        = Phipp_SS_dB       # suction-side  [dB]
        self.Phipp_PS        = Phipp_PS_dB       # pressure-side [dB]
        self.Phipp           = Phipp_total_dB    # total SS+PS   [dB]
        self._Phipp_total_lin = Phipp_total_lin  # total SS+PS   [Pa²/Hz], shape (n_freq,)
        self.Ly              = Ly                # coherence     [m],      shape (n_freq,)

        print(f'    All arrays shape: freq {self.freq.shape}, '
              f'Phipp {self.Phipp.shape}, Ly {self.Ly.shape}')
        print(f"\n{'phipp and coherence extraction complete':.^60}\n")

    def calculate_Spp(self):
        """Main function to calculate the far-field sound pressure spectral density (Spp)
        based on Amiet's model with Roger & Moreau (2005) leading-edge back-scattering correction.

        The far-field PSD is (Roger & Moreau 2005, eq. 1):

        .. math::

            S_{pp}(\\mathbf{x}, f) = \\left(\\frac{\\omega b x_3}{4\\pi c_\\infty S_0^2}\\right)^2
                                     2 b_s \\, |\\mathcal{L}(f,\\mathbf{x})|^2 \\, \\Phi_{pp}(f) \\, l_y(f)

        where:
            - ``b = chord / 2`` is the aeroacoustic half-chord [m],
            - ``b_s = span / 2`` is the half-span [m], so ``2 b_s = span`` [m],
            - ``x3`` is the wall-normal observer coordinate [m],
            - ``S0 = sqrt(x1² + β²(x2² + x3²))`` is the retarded (Prandtl-Glauert) distance [m],
            - ``mathcal{L} = L1 + L2`` is the complex radiation integral (trailing + leading edge),
              **not** a length — this is dimensionless,
            - ``Φ_pp`` is the linear wall-pressure spectrum [Pa²/Hz],
            - ``l_y`` is the spanwise coherence length [m].

        Note: ``mathcal{L}`` (the radiation integral) and ``b_s`` (the half-span) are entirely
        separate quantities. The variable ``D_ij`` in the code holds ``|mathcal{L}(f_i, x_j)|``.

        Uses the linearly-summed wall-pressure spectrum (suction side + pressure side) stored in
        ``self._Phipp_total_lin`` so that both airfoil surfaces contribute correctly.

        Returns:
            Spp (ndarray): Sound pressure spectral density in dB re (2e-5 Pa)²/Hz,
                shape (n_freqs, n_observers).
        """
        print(f"\n{'Calculating Spp':.^60}\n")
        self.Spp = np.zeros((len(self.freq), len(self.observers)))
        self.Directivity = np.zeros((len(self.freq), len(self.observers)), dtype=complex)
        Uinf = self.data['Uref']
        Ma = Uinf / self.cinf
        beta2 = 1.0 - Ma**2      # β² = 1 − M²
        p_ref_sq = (2e-5)**2     # p_ref² [Pa²]

        # self.Ly has shape (n_freq,) and is already aligned to self.freq by compute_spectra.
        # Assert here so any future grid mismatch raises immediately rather than silently
        # producing wrong results via out-of-bounds or misaligned indexing.
        assert len(self.Ly) == len(self.freq), (
            f"Ly length {len(self.Ly)} does not match freq length {len(self.freq)}. "
            "Ensure compute_spectra() has been called before calculate_Spp().")

        print('----> Looping over {0:3d} frequencies and {1:3d} observers:'.format(
            len(self.freq), len(self.observers)))

        for i, freq_i in enumerate(self.freq):
            phippi_lin = self._Phipp_total_lin[i]  # Φ_pp(f_i) [Pa²/Hz]
            l_y_i      = self.Ly[i]                # l_y(f_i)  [m] — index i into the (n_freq,) array
            omega_i    = 2.0 * math.pi * freq_i    # ω_i       [rad/s]
            for j, obs in enumerate(self.observers):
                x1, x2, x3 = obs[0], obs[1], obs[2]
                # Retarded (Prandtl-Glauert) distance [m]
                S0 = np.sqrt(x1**2 + beta2 * (x2**2 + x3**2))
                # mathcal{L}(f_i, x_j) = L1 + L2: complex radiation integral [dimensionless]
                # NOTE: mathcal{L} is NOT a length — do not confuse with span or half-span.
                self.Directivity[i, j] = rdirsupTE(
                    freq_i, x1, x2, x3, self.chord, Uinf, self.Uc, self.cinf)
                calL_sq = abs(self.Directivity[i, j])**2  # |mathcal{L}|², scalar [-]
                # Amiet/Roger&Moreau prefactor (ω b x3 / (4π c∞ S0²))²  [m⁻²]
                prefactor = (omega_i * self.chord * x3 / (4.0 * math.pi * S0**2))**2
                Spp_lin = (
                    prefactor * 2* self.span * calL_sq * phippi_lin  * l_y_i
                )
                self.Spp[i, j] = 10.0 * np.log10(max(Spp_lin, np.finfo(float).tiny) / p_ref_sq)

        # Replace unphysically low values (numerical noise floor)
        self.Spp[self.Spp < -25] = -25.0

        print(f"\n{'Complete Spp Calculation':.^60}\n")
        return self.Spp
    
    def compute_Amiet(self):
        """
        Wrapper function to compute the Amiet TE spectra and coherence.
        """
        # Calculate the spectra and coherence to prep for the Amiet model
        try: 
            self.compute_spectra()
            # Calculate the spectra
            Spp = self.calculate_Spp()
            print("Amiet TE spectra calculated successfully.")
        except Exception as e:
            print(f"Error in Amiet TE spectra calculation: {e}")
            raise
        print(f"\n{'Complete Amiets Trailing Edge Model':=^100}\n")  
        return Spp
        
    @staticmethod
    def read_spectra(filename, has_header:bool=True):
        """
        Reads a two-column file (CSV or .dat) with unknown delimiter and returns two numpy
        arrays (col0, col1).  Rows where the first column (frequency / Strouhal number) is
        zero or negative are dropped, because they cause division-by-zero in the Corcos
        coherence model and log(0) issues elsewhere after Strouhal → Hz conversion.

        Parameters:
            filename   (str):  path to file
            has_header (bool): True if the first line is a header row

        Returns:
            col0 (ndarray): first column values  > 0  (frequency or Strouhal number)
            col1 (ndarray): second column values at those rows (spectral content)
        """
        header_arg = 0 if has_header else None
        df = pd.read_csv(
            filename,
            sep=r'[\s,]+',    # any run of whitespace or commas as delimiter
            engine='python',
            header=header_arg
        )
        if df.shape[1] < 2:
            raise ValueError("File must contain at least two columns.")

        arr = df.iloc[:, :2].to_numpy(dtype=float)

        # Drop rows where the frequency/Strouhal column is zero or negative
        valid = arr[:, 0] > 0.0
        n_dropped = int((~valid).sum())
        if n_dropped > 0:
            print(f'    read_spectra: dropped {n_dropped} row(s) with frequency/Strouhal ≤ 0 '
                  f'from {filename}.')
        arr = arr[valid]

        if len(arr) == 0:
            raise ValueError(f"No valid (positive frequency) rows found in {filename}.")

        return arr[:, 0], arr[:, 1]