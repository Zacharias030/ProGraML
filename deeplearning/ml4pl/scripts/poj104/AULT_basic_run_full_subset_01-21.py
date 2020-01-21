
"""basic_run_full_subset_01-21"""



subfolder = 'basic_run_full_subset_01-21_weird_testset'

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
    'layer_timesteps': [[2,2,2], [2,2,2,2]],  # , [2,2,2,2], [1,1,1,1,1,1]],
    'lr': [0.0005, 0.001],
    'batch_size': [128], #[32, 128],
    #'output_dropout': [0.0]  #, 0.2, 0.5],
    #'edge_weight_dropout': [0.0, 0.1, 0.2],
    'graph_state_dropout': [0.1],  #[0.0, 0.05, 0.1, 0.2],
    'train_subset': [[0,100]],
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