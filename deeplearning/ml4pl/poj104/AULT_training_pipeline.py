# Copyright 2019 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import hashlib, sys, os
import itertools
from pathlib import Path

# make this file executable from anywhere
#if __name__ == '__main__':
full_path = os.path.realpath(__file__)
print(full_path)
repo_root = full_path.rsplit('ProGraML', maxsplit=1)[0] + 'ProGraML'
print(repo_root)
#insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, repo_root)
repo_root = Path(repo_root)

RUNSCRIPT_PATH = repo_root / 'deeplearning' / 'ml4pl' / 'scripts' / 'poj104'
RUNSCRIPT_PATH.mkdir(parents=True, exist_ok=True)

template = """#!/bin/bash
#SBATCH --job-name=poj{i:03d}
#SBATCH --time={timelimit}
#SBATCH --partition=total
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=zacharias.vf@gmail.com
#SBATCH --exclude=ault07,ault08

source /users/zfisches/.bash_profile;
cd /users/zfisches/ProGraML;

srun python deeplearning/ml4pl/models/ggnn/run_ggnn_poj104.py \
--log_dir=deeplearning/ml4pl/poj104/classifyapp_logs/{subfolder}/ \
--config_json={config_str} \
"""  # NO WHITESPACE!!!


def stamp(stuff):
    hash_object = hashlib.sha1(str(stuff).encode("utf-8"))
    hex_dig = hash_object.hexdigest()
    return hex_dig[:7]


def config_generator():
    # GGNN DEVMAP HYPER OPT SERIES
    # devices = [0,1,2,3]
    choices_dict = {
    'layer_timesteps': [[2,2,2,2]], #[[2,2,2], [2,2,2,2], [1,1,1,1,1,1]],
    'lr': [0.0005, 0.001],
    'batch_size': [64], #[32, 128],
    'output_dropout': [0.0, 0.2, 0.5],
    'edge_weight_dropout': [0.0, 0.1, 0.2],
    'graph_state_dropout': [0.0, 0.95, 0.9, 0.8],
    'train_subset': [[0,10]],
    }

    value_list = [choices_dict[k] for k in choices_dict]
    configs = list(itertools.product(*value_list))
    for c in configs:
        yield dict(zip(choices_dict.keys(), c))


def write_runscripts(subfolder=''):
    configs = list(config_generator())
    outpath = RUNSCRIPT_PATH / subfolder
    outpath.mkdir(exist_ok = True, parents=True)
    
    readme = open(outpath / "README.txt", "w")

    for i, config in enumerate(configs):
        stmp = stamp(config)

        # log config to readme
        print(f"HyperOpt-{i:03d}-{stmp}: " + str(config), file=readme)
        
        template_format = {
            #"stamp": stmp,
            "i": i,
            "timelimit": "04:00:00",
            "config_str": str(config).replace('"', "'"),
            "subfolder": subfolder,
            }
        
        runscript = template.format(**template_format)
        print(runscript)
        print("\n")

        with open(outpath / f"run_{i:03d}_{stmp}.sh", "w") as f:
            f.write(runscript)
            f.write(f"\n# HyperOpt-{i:03d}-{stmp}:")
            f.write(f"\n# {config}\n")
    
    readme.close()
    print("Success.")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) > 0:
        write_runscripts(args[0])
    else:
        write_runscripts()
