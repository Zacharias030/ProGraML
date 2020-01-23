
"""AULT_run_over_layer_timesteps-01-23"""

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


subfolder = 'run_over_layer_timesteps'


# iterates over last item 'the quickest' and first item the slowest!
choices_dict = {
    'edge_weight_dropout': [0.0], #, 0.2],
    'layer_timesteps': [8*[1], 5*[2], 6*[1], 6*[2], 4*[2]],  # , [2,2,2,2], [1,1,1,1,1,1]],
    'output_dropout': [0.0],  #, 0.2, 0.5],
    'graph_state_dropout': [0.2],  #[0.0, 0.05, 0.1, 0.2],    
    'batch_size': [128], #[32, 128],    
    'lr': [0.00025],
    'train_subset': [[0,100]],
    # binary choices:
    #'msg_mean_aggregation': [True, False],
    #'use_edge_bias': [True, False],
    #'use_node_types': [False, True],
    #'position_embeddings': [True, False],
    }


# make this file executable from anywhere
import os, sys
from pathlib import Path
full_path = os.path.realpath(__file__)
print(full_path)
repo_root = full_path.rsplit('ProGraML', maxsplit=1)[0] + 'ProGraML'
print(repo_root)
#insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, repo_root)
repo_root = Path(repo_root)

from deeplearning.ml4pl.poj104.AULT_write_runscripts import write_runscripts

if __name__ == '__main__':
    write_runscripts(subfolder=subfolder, template=template, choices_dict=choices_dict)