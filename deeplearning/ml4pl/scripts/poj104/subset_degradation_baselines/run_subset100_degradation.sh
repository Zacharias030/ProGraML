#!/bin/bash
#SBATCH --job-name=pBL100
#SBATCH --time=04:00:00
#SBATCH --partition=total
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=zacharias.vf@gmail.com
#SBATCH --exclude=ault07,ault08,ault20

source /users/zfisches/.bash_profile;
cd /users/zfisches/ProGraML;

srun python deeplearning/ml4pl/models/ggnn/run_ggnn_poj104.py --log_dir=deeplearning/ml4pl/poj104/classifyapp_logs/sub10_ts2222_bs64/ --config_json="{'layer_timesteps': [2, 2, 2], 'lr': 0.0005, 'batch_size': 64, 'output_dropout': 0.0, 'edge_weight_dropout': 0.0, 'graph_state_dropout': 0.1, 'train_subset': [0, 100]}"
# HyperOpt-000-ec66b10:
# {'layer_timesteps': [2, 2, 2, 2], 'lr': 0.0005, 'batch_size': 64, 'output_dropout': 0.0, 'edge_weight_dropout': 0.0, 'graph_state_dropout': 0.0, 'train_subset': [0, 100]}
