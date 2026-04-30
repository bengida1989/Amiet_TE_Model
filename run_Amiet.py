# run_amiet.py
"""
Can be used to run the Amiet TE model from a dictionary of arguments.
"""
import argparse
from TEModel import main
from TEModel import parse_args
import numpy as np

def run_amiet():
    # Set arguments in a dictionary
    args_dict = {
        "output_dir": "./output",
        "output_case": "Test_TE",
        "input_dir": "./input",
        "input_style": "custom",
        "input_data": "",
        "input_data_row": 38,

        # Observer
        "observer_origin": [0.0,0.0,0.0],
        "observer_number": 128,
        "observer_radius": 2.0,
        "selected_freqs": [500, 1000, 2000, 5000],

        # WPS and coherence
        "Struhal": True,
        "WPS_model": "simulation",
        "WPS_model_PS": "simulation",         
        "Coherence_model": "crocos",
        "WPS_path": 'CF08_WPS_038.csv',
        "WPS_path_PS": "CG08_WPS_038.csv",     
        "Coherence_path": 'correlation_length.dat',

        # Airoil Parameters
        "Uref": 16.0,
        "chord": 0.1256,
        "span": 0.5715,

        # Boundary layer parameters
        "Ue": 32.0,
        "delta": 0.01,
        "delta_star": 0.006,
        "theta": 0.005,
        "tau_w": 0.25,
        "beta_c": 14.2,
        "PI": 0.95,
        "Rt": 18.5,
        "dpdx": -1345.0,
        "cinf": 343.0, 
        "Uc": 24.0,
    }

    # Convert to Namespace
    args = argparse.Namespace(**args_dict)

    # Run the TE model
    main(args)

if __name__ == "__main__":
    run_amiet()