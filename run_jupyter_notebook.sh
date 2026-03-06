# Make sure the environment is active
conda activate astronomy

# Launch JupyterLab (recommended)
#jupyter lab sdss_spectra_clustering.ipynb
nohup jupyter lab sdss_spectra_clustering.ipynb > jupyter.log 2>&1 &

# — or classic Notebook —
#jupyter notebook sdss_spectra_clustering.ipynb