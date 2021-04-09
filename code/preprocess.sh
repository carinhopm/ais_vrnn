#!/bin/sh
# embedded options to bsub - start with #BSUB
# -- Name of the job --
#BSUB -J preprocess_2002_CargTank
# -- specify queue --
#BSUB -q gputitanxpascal
# -- number of processors
###BSUB -n 4
### -- Specify that the cores must be on the same host --
###BSUB -R "span[hosts=1]"
### -- Select the resources: 1 gpu in exclusive process mode --
###BSUB -gpu "num=1:mode=exclusive_process"
### -- specify that we need 2GB of memory per core/slot -- 
#BSUB -R "rusage[mem=2GB]"
### -- Specify that we want the job to get killed if it exceeds 5 GB per core/slot --
#BSUB -M 5GB
# -- set walltime limit: hh:mm -- 
#BSUB -W 24:00
# -- user email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
###BSUB -u s192770@student.dtu.dk
# -- mail notification --
# -- at start --
###BSUB -B
# -- at completion --
###BSUB -N
# --Specify the output and error file. %J is the job-id --
# --  -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -oo logs/%J.out
#BSUB -eo logs/%J.err

# unload already installed software
module unload python3
module unload cuda
module unload cudnn

# load modules
module load python3/3.6.13
module load cuda/9.0
module load cudnn/v7.0.5-prod-cuda-9.0

# activate the virtual environment which includes the necessary python packages
source ./python_env/bin/activate

echo "Setup completed. Running script...\n"

#python run_model.py
python3 checking.py




