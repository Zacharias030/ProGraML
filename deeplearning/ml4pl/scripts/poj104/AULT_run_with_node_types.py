"""Config for use with AULT_runscript_writer.py"""

########### Make your changes below the line ######################

# subfolder under classifyapp_logs/
subfolder = 'run_with_node_types_and_structure_only'

# choose 3 letters to identify your experiment
jobname = 'nty'

# 4h runtime per submission
resubmit_times_per_job = 4

# a set of hyperparameters to grid search over
choices_dict = {
    'num_epochs': [20],
    'train_subset': [[0,100]],
    'use_node_types': [True, False],
    'inst2vec_embeddings': ['random', 'none'],
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

srun python deeplearning/ml4pl/models/ggnn/run_ggnn_poj104.py \
--log_dir=deeplearning/ml4pl/poj104/classifyapp_logs/{subfolder}/ \
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

from deeplearning.ml4pl.poj104.AULT_write_runscripts import write_runscripts


if __name__ == '__main__':
    write_runscripts(subfolder=subfolder, template=template, choices_dict=choices_dict, jobname=jobname, steps=resubmit_times_per_job)
