#!/bin/sh

#SBATCH --job-name=tml1
#SBATCH -N 1      # nodes requested
#SBATCH -n 4      # tasks requested
#SBATCH -c 1      # cores requested
#SBATCH -o outfile  # send stdout to outfile
#SBATCH -e errfile  # send stderr to errfile
#SBATCH -t 00:10:00  # time requested in hour:minute:second
#SBATCH --gpus=3

python main_a.py
