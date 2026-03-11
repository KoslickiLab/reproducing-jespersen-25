#!/usr/bin/env bash
# Launch JupyterLab for the data preparation notebooks.
# Run this script from the project root directory.
#
# Notebooks (in recommended execution order):
#   notebooks/data_download.ipynb            -- download SDSS spectra and WISE photometry
#   notebooks/sdss_data_preprocessing.ipynb  -- preprocess SDSS spectra -> data/sdss_output/
#   notebooks/wise_data_preprocessing.ipynb  -- crossmatch and preprocess WISE -> data/wise_output/

conda activate astronomy

# Launch JupyterLab rooted at the project directory so that relative paths
# inside the notebooks (e.g. ../data/sdss_output/) resolve correctly.
# Output is logged to logs/jupyter.log and the process runs in the background.
nohup jupyter lab --notebook-dir="$(pwd)" notebooks/ > logs/jupyter.log 2>&1 &

echo "JupyterLab started (PID $!). See logs/jupyter.log for the server URL."
