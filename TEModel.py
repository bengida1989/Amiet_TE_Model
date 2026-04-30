"""
Wrapper function for Amiet's TE model.
This function completes the TE model by parsing comand line arguments and completes the input data required
This function also performs i/o operations and stores the output in the specified directory
"""
import argparse
import os
import pandas as pd
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from Amiet_TE_Model import AmietTE
from Amiet_TE_Model.utils_checkdata import check_data


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments using argparse.
    
    Returns:
        Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Extract cut-planes from simulation solution files."
    )
    # The required simulation parameters
    parser.add_argument("--output_dir", '-o', type=str, default="./output", help="Output directory")
    parser.add_argument("--output_case", '-oc', type=str, default="Aimiet_TE", help="Output case name (used for the output file naming)")
    parser.add_argument("--input_dir", '-i', type=str, default="./input", help="Input directory")
    parser.add_argument("--input_style", '-s', type=str, default="csv", help="Input data file style", choices=["csv", "custom"])
    parser.add_argument("--input_data", '-d', type=str, default="TA10_BLparams_zones.csv", help="Input data file containing the boundary layer parameters")
    parser.add_argument("--input_data_row", '-dr', type=int, default=10, help="The specific row (probe location) we would like to use in the csv file corresponding to the boundary layer parameters")
    
    # The required observer parameters
    parser.add_argument("--observer_origin", '-ob_o', type=list, default=[0,0,0], help="Observer type", choices=["TE", "LE"])
    parser.add_argument("--observer_number", '-ob_n', type=int, default=12, help="Number of observers")
    parser.add_argument("--observer_radius", '-ob_r', type=float, default=2, help="Observer radius from the origin [m]")
    parser.add_argument("--selected_freqs", '-sf', type=list, default=[500,1000,2000], help="Selected frequencies for the directivity plot")
    
    # The WPS and Coherence options
    parser.add_argument("--Struhal", '-st', type=bool, default=True, help="Whether to use Strouhal scaling for the frequencies (f*delta/Ue) instead of physical frequencies (Hz)")
    parser.add_argument("--WPS_model", '-wps', type=str.lower, default='rozenberg', help="Applied model for the suction-side WPS", choices=["goody", "rozenberg", "lee", "experiment", "simulation"])
    parser.add_argument("--WPS_model_PS", '-wps_ps', type=str.lower, default=None, help="Applied model for the pressure-side WPS. If not provided the pressure-side contribution is assumed to be zero.", choices=["goody", "rozenberg", "lee", "experiment", "simulation"])
    parser.add_argument("--Coherence_model", '-coh', type=str.lower, default='corcos', help="Applied model for the Coherence", choices=["corcos", "experiment", "simulation"])
    parser.add_argument("--WPS_path", '-wps_path', type=str, default=None, help="Path to the suction-side WPS data file, relative to the input directory")
    parser.add_argument("--WPS_path_PS", '-wps_path_ps', type=str, default=None, help="Path to the pressure-side WPS data file, relative to the input directory")
    parser.add_argument("--Coherence_path", '-coh_path', type=str, default=None, help="Path to the Coherence data file, relative to the input directory")

    # The required boundary layer parameters
    parser.add_argument("--Uref", type=float, default=30.0, help="Reference velocity (m/s)")
    parser.add_argument("--Ue", type=float, default=32.0, help="Edge velocity (m/s)")
    parser.add_argument("--delta", type=float, default=0.01, help="Boundary layer thickness (m)")
    parser.add_argument("--delta_star", type=float, default=0.006, help="Displacement thickness (m)")
    parser.add_argument("--theta", type=float, default=0.005, help="Momentum thickness (m)")
    parser.add_argument("--tau_w", type=float, default=0.25, help="Wall shear stress (Pa)")
    parser.add_argument("--beta_c", type=float, default=14.2, help="Clauser pressure gradient parameter")
    parser.add_argument("--PI", type=float, default=0.95, help="Pressure gradient parameter")
    parser.add_argument("--Rt", type=float, default=18.5, help="External/internal time scale ratio")
    parser.add_argument("--dpdx", type=float, default=-1345.0, help="Pressure gradient (Pa/m)")
    parser.add_argument("--chord", type=float, default=0.3048, help="Chord length (m)")
    parser.add_argument("--span", type=float, default=0.5715, help="Span (m)")
    parser.add_argument("--cinf", type=float, default=343.0, help="Speed of sound (m/s)")
    parser.add_argument("--Uc", type=float, default=24, help="Convection velocity (m/s)")
    
    return parser.parse_args()

class ComputeAmiet_TE():
    def __init__(self, args):
        self.start = time.time()
        # I/O handling
        self.input_dir = args.input_dir
        self.output_dir = args.output_dir
        if os.path.exists(self.output_dir) is False:
            os.makedirs(self.output_dir, exist_ok=True)
        self.spectra_dir = os.path.join(self.output_dir, args.output_case+"_Spectra")
        self.directivity_dir = os.path.join(self.output_dir, args.output_case+"_Directivity")
        if os.path.exists(self.spectra_dir) is False:           # create the spectra directory
            os.makedirs(self.spectra_dir, exist_ok=True)
        if os.path.exists(self.directivity_dir) is False:       # create the directivity directory
            os.makedirs(self.directivity_dir, exist_ok=True)
        args.input_data = os.path.join(self.input_dir, args.input_data) 
        # Observer parameters
        self.observer_number = args.observer_number
        self.observer_radius = args.observer_radius
        self.observer_origin = args.observer_origin
        self.observer_probes = self.create_observers()
        

        if args.input_style == "csv":
            tmp = pd.read_csv(args.input_data ,  header='infer', sep=' ')
            data = tmp.iloc[args.input_data_row]
        elif args.input_style == "custom":
            data = self.map_data(args)
        
        # Check the input directory and file validity
        if not os.path.exists(self.input_dir) :
            raise FileNotFoundError(f"Input directory {self.input_dir} does not exist.")
        # Check if the input data file exists
        if not os.path.isfile(args.input_data) and args.input_style == "csv":
            raise FileNotFoundError(f"Input file {args.input_data} does not exist.")
        try: 
            check_data(data)
        except ValueError as e:
            print(f"Error in input data: {e}")
            raise
        self.data = data
        
        
        # check the validity of the input data for experimental or simulation data
        args.WPS_path = os.path.join(self.input_dir, args.WPS_path) if args.WPS_path is not None else None
        args.WPS_path_PS = os.path.join(self.input_dir, args.WPS_path_PS) if args.WPS_path_PS is not None else None
        args.Coherence_path = os.path.join(self.input_dir, args.Coherence_path) if args.Coherence_path is not None else None
        if args.WPS_model == "experiment" or args.WPS_model == "simulation":
            if args.WPS_path is None:
                raise ValueError("Suction-side WPS path must be provided for experimental or simulation data.")
            if not os.path.isfile(args.WPS_path):
                raise FileNotFoundError(f"Suction-side WPS file {args.WPS_path} does not exist.")
        if args.WPS_model_PS is not None and (args.WPS_model_PS == "experiment" or args.WPS_model_PS == "simulation"):
            if args.WPS_path_PS is None:
                raise ValueError("Pressure-side WPS path must be provided for experimental or simulation data.")
            if not os.path.isfile(args.WPS_path_PS):
                raise FileNotFoundError(f"Pressure-side WPS file {args.WPS_path_PS} does not exist.")
        if args.Coherence_model == "experiment" or args.Coherence_model == "simulation":
            if args.Coherence_path is None:
                raise ValueError("Coherence path must be provided for experimental or simulation data.")
            if not os.path.isfile(args.Coherence_path):
                raise FileNotFoundError(f"Coherence file {args.Coherence_path} does not exist.")
        
        
        
    def map_data(self, args) -> pd.Series:
        """
        Map only selected physical arguments into a pandas Series.
        """
        # Full argument dict
        arg_dict = vars(args)
        # Keys to include in the Series
        selected_keys = [
            'Uref', 'Ue', 'delta', 'delta_star', 'theta',
            'tau_w', 'beta_c', 'PI', 'Rt',
            'dpdx', 'chord', 'span', 'cinf', 'Uc',
        ]
        # Filter just the desired keys
        filtered = {k: arg_dict[k] for k in selected_keys if k in arg_dict}
        return pd.Series(filtered)

    def create_observers(self):
        """
        Create `n_probes` observers evenly spaced around a circle of radius `observer_radius`
        in the x–z plane, centered at args.observer_origin = [x0, y0, z0].

        Returns:
            List of [x, y, z] coordinates for each observer.
        """
        n_probes = self.observer_number
        radius   = self.observer_radius
        x0, y0, z0 = self.observer_origin    # unpack full origin coordinate
        if math.isclose(y0, 0.0):
            y0 = 1e-1
        # angles around the circle (radians), endpoint=False to avoid duplicate at 2π
        angles = np.linspace(0, 2*np.pi, n_probes, endpoint=False)

        # circle in x–z plane
        x = x0 + radius * np.cos(angles)
        y = np.full(n_probes, y0)            # constant spanwise coordinate
        z = z0 + radius * np.sin(angles)

        probe_coords = np.column_stack((x, y, z)).tolist()

        # Optionally store on the instance
        self.observers = probe_coords

        return probe_coords
    
    def compute_spectra(self, args):
        """ This function computes the spectra using Amiet's TE model using the classes and functions defined in the Amiet_TE_Model module.
        Args:
            args (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Initializing the amiet model
        amiet = AmietTE(observers=self.observers, data=self.data, freq=None, 
            phipp_method=args.WPS_model,
            phipp_method_PS=args.WPS_model_PS,
            coh_method=args.Coherence_model, 
            spectra=args.WPS_path,
            spectra_PS=args.WPS_path_PS,
            coherence=args.Coherence_path,
            cinf=args.cinf, 
            span=args.span, 
            chord=args.chord, 
            Uc=args.Uc,
            struhal=args.Struhal
            )
        
        # Compute the spectra
        Spp = amiet.compute_Amiet()   # shape: (len(frequencies), n_probes)
        Phipp = amiet.Phipp           # total WPS in dB (SS + PS)
        Phipp_SS = amiet.Phipp_SS     # suction-side WPS in dB
        Phipp_PS = amiet.Phipp_PS     # pressure-side WPS in dB
        
        print("----> Performing I/O Operations.")
        self.plot_spectra(amiet.freq, Spp, Phipp, Phipp_SS, Phipp_PS, args)
        self.plot_directivity(amiet.freq, Spp, args.selected_freqs)
        self.output_file(amiet.freq, Phipp, Phipp_SS, Phipp_PS, Spp)
        print(f"outputting plot and spectra for {len(amiet.freq)} frequencies and {len(self.observers)} observers.")
        print("spectra files are saved in: ", self.spectra_dir)
        print("directivity files are saved in: ", self.directivity_dir)
        end = time.time()
        print("----> I/O operations completed.\n")
        print(f"----> Total compute time: {end - self.start:.2f} seconds.\n\n\n")
    
    def plot_spectra(self, freq, Spp, Phipp, Phipp_SS, Phipp_PS, args):
        """
        Plot the far-field spectra for every observer and the wall-pressure spectra
        (suction side, pressure side, and total).
        """
        self.set_plot_params()
        for i, observer in enumerate(self.observers):
            fig, ax = plt.subplots()
            ax.semilogx(freq, Spp[:, i], 'r-', linewidth=2.5)
            ax.set_title('Amiet\'s Model')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel(r'$10 \log_{10} \left(S_{pp}/p_{\mathrm{ref}}^2\right)$ (dB/Hz)')
            ax.grid(True)
            ax.set_xlim(100, 40000)
            plt.tight_layout()
            figname = os.path.join(self.spectra_dir, f'Amiet_TE_Spectra_Probe_{i:03d}')
            plt.savefig(figname+'.png', format='png', dpi=300)
            plt.close(fig)

        # Wall-pressure spectra: suction side, pressure side (if available), and total
        fig, ax = plt.subplots()
        ax.semilogx(freq, Phipp_SS, 'b--', linewidth=2.0, label='Suction side (SS)')
        has_PS = np.any(np.isfinite(Phipp_PS))
        if has_PS:
            ax.semilogx(freq, Phipp_PS, 'g--', linewidth=2.0, label='Pressure side (PS)')
        ax.semilogx(freq, Phipp, 'r-', linewidth=2.5, label='Total (SS + PS)')
        ax.set_title('Wall Pressure Spectra')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel(r'$10 \log_{10} \left(\phi_{pp}/p_{\mathrm{ref}}^2\right)$ (dB/Hz)')
        ax.grid(True)
        ax.set_xlim(100, 40000)
        ax.legend()
        plt.tight_layout()
        figname = os.path.join(self.spectra_dir, f'WPS_Spectra_' + args.WPS_model)
        plt.savefig(figname+'.png', format='png', dpi=300)
        plt.close(fig)
    
    def plot_directivity(self, freq, Spp, selected_freqs):
        # Create one polar plot
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        angles = np.linspace(0, 2*np.pi, self.observer_number, endpoint=False)
        for f in selected_freqs:
            # find the index of the frequency closest to f
            idx = np.argmin(np.abs(freq - f))
            # normalize to peak and convert to dB or keep linear depending on data
            P = Spp[idx, :]
            # Plot on the same axes
            ax.plot(angles, P, marker='o', linestyle='-', label=f"{freq[idx]:.0f} Hz")

        # Configure the polar plot
        ax.set_title("Directivity Patterns")
        ax.set_theta_zero_location('W')   # zero at the right
        ax.set_theta_direction(-1)        # clockwise increasing
        ax.set_rlabel_position(135)       # move radial labels away from plot
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))  # adjust as needed
        max_val = np.max([Spp[np.argmin(np.abs(freq - f)), :].max() for f in selected_freqs])       # normalize to peak
        ax.set_ylim(0, 1.2 * max_val)
        figname = os.path.join(self.directivity_dir, f'Diectivity_Patterns_'+'_'.join([str(f) for f in selected_freqs]))
        plt.savefig(figname+'.png', format='png', dpi=300)
    
    def output_file(self, freq, Phipp, Phipp_SS, Phipp_PS, Spp):
        """Output the results to individual .dat files for each probe and WPS components.
        
        Args:
            freq (numpy.ndarray): Frequencies, shape (n_freq,).
            Phipp (numpy.ndarray): Total (SS+PS) wall-pressure spectrum in dB, shape (n_freq,).
            Phipp_SS (numpy.ndarray): Suction-side WPS in dB, shape (n_freq,).
            Phipp_PS (numpy.ndarray): Pressure-side WPS in dB, shape (n_freq,).
            Spp (numpy.ndarray): Far-field spectra at each observer, shape (n_freq, n_probes).
        """
        n_probes = Spp.shape[1]

        for i in range(n_probes):
            values = np.column_stack((freq, Spp[:, i]))
            filename = os.path.join(self.spectra_dir, f'Amiet_TE_Spectra_Probe_{i:03d}.dat')
            np.savetxt(filename, values,
                       header='frequency [Hz] PSD [dB/Hz]', comments='', fmt='%.10f %.10f')

        # Total WPS (SS + PS)
        values = np.column_stack((freq, Phipp))
        np.savetxt(os.path.join(self.spectra_dir, 'WPS_Spectra_Total.dat'), values,
                   header='frequency [Hz] PSD_total [dB/Hz]', comments='', fmt='%.10f %.10f')

        # Suction-side WPS
        values_SS = np.column_stack((freq, Phipp_SS))
        np.savetxt(os.path.join(self.spectra_dir, 'WPS_Spectra_SS.dat'), values_SS,
                   header='frequency [Hz] PSD_SS [dB/Hz]', comments='', fmt='%.10f %.10f')

        # Pressure-side WPS (written even when all -inf, for completeness)
        Phipp_PS_out = np.where(np.isfinite(Phipp_PS), Phipp_PS, -999.0)
        values_PS = np.column_stack((freq, Phipp_PS_out))
        np.savetxt(os.path.join(self.spectra_dir, 'WPS_Spectra_PS.dat'), values_PS,
                   header='frequency [Hz] PSD_PS [dB/Hz] (-999 indicates no pressure-side data)',
                   comments='', fmt='%.10f %.10f')
        
    @staticmethod
    def set_plot_params():
        sizes = {'small': 20, 'medium': 24, 'large': 28}
        plt.rcParams.update({
            'font.size': sizes['medium'],
            'axes.titlesize': sizes['medium'],
            'axes.labelsize': sizes['medium'],
            'xtick.labelsize': sizes['small'] ,
            'ytick.labelsize': sizes['small'] ,
            'legend.fontsize': sizes['small'],
            'figure.titlesize': sizes['large'],
            'mathtext.fontset': 'stix',
            'font.family': 'STIXGeneral',
            'grid.linewidth': 1,
            'grid.linestyle': ':',
            'lines.linewidth': 2,
            'lines.markersize': 6
        })

def main(args):
    amiet = ComputeAmiet_TE(args)
    amiet.compute_spectra(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)