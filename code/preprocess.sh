#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J CargTank_2002
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=16GB]"
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
#BSUB -o CargTank_2002.out
#BSUB -e CargTank_2002.err
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

python3 run_model.py
#python3 checking.py




