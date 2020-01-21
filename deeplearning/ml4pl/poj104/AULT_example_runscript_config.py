"""Exemplary config for use with AULT_training_pipeline.py"""


subfolder = 'sub10_ts4x2_bs64'

template = """#!/bin/bash
#SBATCH --job-name=poj{i:03d}
#SBATCH --time={timelimit}
#SBATCH --partition=total
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=zacharias.vf@gmail.com
#SBATCH --exclude=ault07,ault08,ault20

source /users/zfisches/.bash_profile;
cd /users/zfisches/ProGraML;

srun python deeplearning/ml4pl/models/ggnn/run_ggnn_poj104.py \
--log_dir=deeplearning/ml4pl/poj104/classifyapp_logs/{subfolder}/ \
--config_json="{config_str}" \
"""  # NO WHITESPACE!!!

choices_dict = {
    'layer_timesteps': [[2,2,2,2]], #[[2,2,2], [2,2,2,2], [1,1,1,1,1,1]],
    'lr': [0.0005, 0.001],
    'batch_size': [64], #[32, 128],
    'output_dropout': [0.0, 0.2, 0.5],
    'edge_weight_dropout': [0.0, 0.1, 0.2],
    'graph_state_dropout': [0.0, 0.95, 0.9, 0.8],
    'train_subset': [[0,10]],
    }
