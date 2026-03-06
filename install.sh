#1. Creat env
conda create -n astronomy python=3.11 -y

conda activate astronomy

# 2. Core scientific stack (conda-forge for best compatibility)
mamba install -c conda-forge numpy scipy matplotlib scikit-learn astropy -y

# 3. Astronomy-specific packages
mamba install -c conda-forge astroquery specutils -y

# 4. Jupyter
mamba install -c conda-forge jupyterlab notebook ipykernel -y

# 5. Register the environment as a Jupyter kernel
python -m ipykernel install --user --name astronomy --display-name "Python 3 (astronomy)"

# 6. (Optional, for the full paper pipeline) spender autoencoder + PyTorch
mamba install -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1 -y
pip install spender