"""transformer threadcoarsening transfer"""

########### Make your changes below the line ######################

# subfolder under classifyapp_logs/
subfolder = 'threadcoarsening_logs/transformer_transfer_Tahiti_2'
# choose 3 letters to identify your experiment
jobname = 'ctT'

model = None #'devmap_poj104' # 'ggnn_SET' or 'transformer_SET' or 'x_pretraining' (dataset without split!)
dataset = 'threadcoarsening_Tahiti' # poj104, ncc, devmap_amd, _nvidia, threadcoarsening_Cypress, _Tahiti, _Fermi, _Kepler
kfold = '--kfold' # '' = no kfold, '--kfold' for kfold.
transfer = '--transfer=transformer_threadcoarsening' # '' for no transfer, '--transfer=transformer_devmap'
restore = "--restore=deeplearning/ml4pl/poj104/logs/ncc_logs/transformer/2020-03-02_04:12:31_000_31632_model_best.pickle"

# 4h runtime per submission
resubmit_times_per_job = 1

# a set of hyperparameters to grid search over
choices_dict = {
    'random_seed': [43],
    'lr': [0.00025, 5e-5],
    'batch_size': [128, 8, 4],
    'output_dropout': [0.0, 0.1, 0.2], #, 0.5],
    'num_epochs': [600],
    'vocab_size': [8569],
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
--dataset={dataset} \
{kfold} \
{transfer} \
--log_dir=deeplearning/ml4pl/poj104/logs/{subfolder}/ \
{restore_by_pattern} \
--config_json="{config_str}" \
{restore} \
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
    write_runscripts(subfolder=subfolder, template=template, choices_dict=choices_dict, jobname=jobname, steps=resubmit_times_per_job, model=model, dataset=dataset, kfold=kfold, transfer=transfer, restore=restore)
