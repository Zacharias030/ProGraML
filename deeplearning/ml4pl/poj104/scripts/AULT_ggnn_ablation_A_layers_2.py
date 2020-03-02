
########### Make your changes below the line ######################

# subfolder under logs/
subfolder = 'classifyapp_logs/ggnn_ablation_A'
# choose 3 letters to identify your experiment
jobname = 'o2A'
model = 'ggnn_poj104' # 'ggnn_SET' or 'transformer_SET' or 'x_pretraining' (dataset without split!)
dataset = 'poj104' # poj104, ncc, devmap_amd, _nvidia, threadcoarsening_Cypress, _Tahiti, _Fermi, _Kepler
kfold = '' # '' = no kfold
transfer = ''
# 4h runtime per submission
resubmit_times_per_job = 5


# iterates over last item 'the quickest' and first item the slowest!
choices_dict = {
    'edge_weight_dropout': [0.0], #, 0.2],
    #'layer_timesteps': [8*[1], 5*[2], 6*[1], 6*[2], 4*[2]],  # , [2,2,2,2], [1,1,1,1,1,1]],
    'gnn_layers': [8, 10, 6, 4, 12],
    'update_weight_sharing': [2],
    'message_weight_sharing': [2],
    'output_dropout': [0.0],  #, 0.2, 0.5],
    'graph_state_dropout': [0.2],  #[0.0, 0.05, 0.1, 0.2],    
    'batch_size': [128], #[32, 128],    
    'lr': [0.00025],
    'train_subset': [[0,100]],
    'num_epochs': [16],
    # binary choices:
    #'msg_mean_aggregation': [True, False],
    #'use_edge_bias': [True, False],
    #'use_node_types': [True] #[False, True],
    #'position_embeddings': [True, False],
    }




################ Make your changes above the line ################

template = """#!/bin/bash
#SBATCH --job-name={jobname}{i:03d}
#SBATCH --time={timelimit}
#SBATCH --partition=total
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=zacharias.vf@gmail.com
#SBATCH --exclude=ault07,ault08,ault20
#SBATCH --dependency=singleton

source /users/zfisches/.bash_profile;
cd /users/zfisches/ProGraML;

srun python deeplearning/ml4pl/models/ggnn/run.py \
--model={model} \
--dataset={dataset} \
{kfold} \
{transfer} \
--log_dir=deeplearning/ml4pl/poj104/logs/{subfolder}/ \
{restore_by_pattern} \
--config_json="{config_str}" \
"""  # NO WHITESPACE!!!


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

from deeplearning.ml4pl.poj104.scripts.AULT_write_runscripts import write_runscripts


if __name__ == '__main__':
    write_runscripts(subfolder=subfolder, template=template, choices_dict=choices_dict, jobname=jobname, steps=resubmit_times_per_job, model=model, dataset=dataset, kfold=kfold, transfer=transfer)