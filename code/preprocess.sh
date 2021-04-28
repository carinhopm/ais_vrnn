#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuk80
### -- set the job Name --
#BSUB -J CargTank_1912
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- specify that the cores should be in the same host --
#BSUB -R "span[hosts=1]"
### -- Select the resources: 2 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 16GB of system-memory per core
#BSUB -R "rusage[mem=16GB]"
### -- specify that we want the job to get killed if it exceeds 8GB per core/slot --
#BSUB -M 16GB
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o logs/CargTank_1912_%J.out
#BSUB -e logs/CargTank_1912_%J.err
# -- end of LSF options --


# print GPU info.
nvidia-smi

# unload already installed software
module unload python3
module unload cuda
module unload cudnn

# load modules
module load python3/3.6.13
module load cuda/10.2
module load cudnn/v7.6.5.32-prod-cuda-10.2

# activate the virtual environment which includes the necessary python packages
source ./python_env/bin/activate

echo "Setup completed. Running script...\n"

#python3 checking.py
#python3 run_model.py
fil-profile run run_model.py



