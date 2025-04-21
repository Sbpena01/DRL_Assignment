#!/bin/bash
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --mem=32g
#SBATCH -J "Reinforcement Deep-Q Learning"
#SBATCH -A rbe549
#SBATCH -p academic
#SBATCH -t 23:59:59
#SBATCH --gres=gpu:1
#SBATCH --error=slurm_outputs/slurm_dqn_%A_%a.err
#SBATCH --output=slurm_outputs/slurm_dqn_%A_%a.out
#SBATCH --mail-user=sbpena@wpi.edu
#SBATCH --mail-type=ALL

module load py-pip/24.0 ffmpeg

source ./venv/bin/activate

pip install -r requirements.txt

python -u main_ddpg.py
