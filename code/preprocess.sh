#!/bin/sh
# embedded options to bsub - start with #BSUB
# -- Name of the job --
#BSUB -J preprocess_2002_CargTank
# -- specify queue --
#BSUB -q gputitanxpascal
### -- specify that we need 2GB of memory per core/slot -- 
#BSUB -R "rusage[mem=2GB]"
# -- number of processors
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
###BSUB -gpu "num=1:mode=exclusive_process"
# -- set walltime limit: hh:mm -- 
#BSUB -W 1:00 
# -- user email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s192770@student.dtu.dk
# -- mail notification --
# -- at completion --
#BSUB -N
# --Specify the output and error file. %J is the job-id --
# --  -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -oo logs/%J.out
#BSUB -eo logs/%J.err

module load python3/3.7.7
module load cuda/10.2
module load cudnn/v7.6.5.32-prod-cuda-10.2
pip3 install --user torch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 progressbar2
conda info --envs
echo "Setup completed. Running script...\n"

#python3 run_model.py
python3 checking.py

