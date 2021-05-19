#!/bin/sh

# unload already installed software
module unload python3
module unload cuda
module unload cudnn

# load modules
module load python3/3.6.13
module load cuda/9.0
module load cudnn/v7.0.5-prod-cuda-9.0

# setup virtual environment
python3 -m venv python_env
source ./python_env/bin/activate

# install needed packages
pip3 install -U -r requirements.txt

# create necessary directories
mkdir -p logs
mkdir -p models/saved_models
