"""
Usage:
    AULT_submit_restored_run.py --restore CHECKPOINT [options]
    
Options:
    -s --sub SUBFOLDER      Subfolder to log_dir, 
                                will be same as checkpoint if not given.
"""



from docopt import docopt


template = """#!/bin/bash
#SBATCH --job-name=resto
#SBATCH --time={timelimit}
#SBATCH --partition=total
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=zacharias.vf@gmail.com
#SBATCH --exclude=ault07,ault08,ault20

source /users/zfisches/.bash_profile;
cd /users/zfisches/ProGraML;

srun python deeplearning/ml4pl/models/ggnn/run_ggnn_poj104.py \
--log_dir=deeplearning/ml4pl/poj104/classifyapp_logs/{subfolder}/ \
--restore={checkpoint} \
"""  # NO WHITESPACE!!!

import hashlib, sys, os
import itertools
from pathlib import Path
from subprocess import Popen, PIPE

# make this file executable from anywhere
full_path = os.path.realpath(__file__)
print(full_path)
repo_root = full_path.rsplit('ProGraML', maxsplit=1)[0] + 'ProGraML'
print(repo_root)
#insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, repo_root)
repo_root = Path(repo_root)


def submit_job(checkpoint, subfolder=None):
    cp = str(Path(checkpoint).absolute()).replace(str(repo_root), "")
    if cp[0] == '/': cp = cp[1:]

    if not subfolder:
        subfolder = Path(cp).parent.name
        subfolder = str(subfolder)

    template_format = {
        "timelimit": "04:00:00",
        "subfolder": subfolder,
        "checkpoint": cp,
        }

    runscript = template.format(**template_format)
    
    p = Popen(['sbatch'], stdin=PIPE, stdout=PIPE, encoding='utf8')
    p.communicate(runscript)
    #.communicate(runscript)
    
    #Popen('sbatch'| sbatch -n 1')
    
    print("Done.")


if __name__ == "__main__":
    args = docopt(__doc__)
    print(args)
    submit_job(checkpoint=args['CHECKPOINT'], subfolder=args.get('--sub'))

# large_run_full_subset/2020-01-22-17:23:09_101754_model_best.pickle