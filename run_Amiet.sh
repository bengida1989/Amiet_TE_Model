#!/bin/bash

# ----------------------------------------
# Shell script to run Amiet TE model via CLI
# ----------------------------------------

# Default arguments, you can modify these as needed
OUTPUT_DIR="./output"
OUTPUT_CASE="Test_TE"
INPUT_DIR="./input"
INPUT_DATA="TA10_BLparams_zones.csv"
INPUT_DATA_ROW=20
INPUT_STYLE="csv"
OBSERVER_ORIGIN="[0,0.1,0]"
OBSERVER_NUMBER=128
OBSERVER_RADIUS=2.0
SELECTED_FREQS="[500,1000,2000,5000]"
WPS_MODEL="simulation"
WPS_MODEL_PS=""                 # Pressure-side WPS model: goody|rozenberg|lee|experiment|simulation (leave empty to assume zero PS contribution)
COH_MODEL="corcos"
WPS_PATH=""                     # Path to suction-side WPS data (required when WPS_MODEL is "simulation" or "experiment")
WPS_PATH_PS=""                  # Path to pressure-side WPS data (required when WPS_MODEL_PS is "simulation" or "experiment")
COH_PATH=""                     # Path to Coherence model data (required when COH_MODEL is "simulation" or "experiment")
UE=30.0
DELTA=0.01
DELTA_STAR=0.006
THETA=0.005
TAUW=0.25
BETA_C=14.2
PI=0.95
RT=18.5
DPDX=-1345.0
CHORD=0.3048
SPAN=0.5715
CINF=343.0
UC=24.0

# Call the Python script with CLI arguments
python run_amiet.py \
  --output_dir "$OUTPUT_DIR" \
  --output_case "$OUTPUT_CASE" \
  --input_dir "$INPUT_DIR" \
  --input_data "$INPUT_DATA" \
  --input_data_row "$INPUT_DATA_ROW" \
  --input_style "$INPUT_STYLE" \
  --observer_origin "$OBSERVER_ORIGIN" \
  --observer_number "$OBSERVER_NUMBER" \
  --observer_radius "$OBSERVER_RADIUS" \
  --selected_freqs "$SELECTED_FREQS" \
  --WPS_model "$WPS_MODEL" \
  --WPS_model_PS "$WPS_MODEL_PS" \
  --Coherence_model "$COH_MODEL" \
  --WPS_path "$WPS_PATH" \
  --WPS_path_PS "$WPS_PATH_PS" \
  --Coherence_path "$COH_PATH" \
  --Ue "$UE" \
  --delta "$DELTA" \
  --delta_star "$DELTA_STAR" \
  --theta "$THETA" \
  --tau_w "$TAUW" \
  --beta_c "$BETA_C" \
  --PI "$PI" \
  --Rt "$RT" \
  --dpdx "$DPDX" \
  --chord "$CHORD" \
  --span "$SPAN" \
  --cinf "$CINF" \
  --Uc "$UC"