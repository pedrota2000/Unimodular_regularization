#!/bin/bash
#SBATCH --job-name=MH_paper
#SBATCH --ntasks-per-node=5  # Utilize all CPU cores
#SBATCH --gres=gpu:A40
#SBATCH --cpus-per-task=5
#SBATCH --mem=800G
#SBATCH --time=20-00:00:0
#SBATCH -o /home/pedro/NNholo/MH_flat_paper/results/FV2.o
#SBATCH -e /home/pedro/NNholo/MH_flat_paper/results/FV2.e
#SBATCH --partition=unlimited
#SBATCH --mail-type=ALL       # Send email on job start, end, and failure
#SBATCH --mail-user=pedro.tarancon@fqa.ub.edu  # Replace with your email address

source /home/pedro/env/bin/activate
module load cuda python3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Run the first PyTorch training script
python3 NNholo_MH_paper.py --gpu 0 

# Run the second PyTorch training script
#python3 train_2.py --gpu 0 &
#python3 train_3.py --gpu 0 &
#python3 train_4.py --gpu 0 &
#python3 train_5.py --gpu 0 &

