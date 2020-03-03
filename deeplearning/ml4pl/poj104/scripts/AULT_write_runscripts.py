#!/usr/bin/env python
import hashlib, sys, os
import itertools
from pathlib import Path

# make this file executable from anywhere
full_path = os.path.realpath(__file__)
print(full_path)
repo_root = full_path.rsplit('ProGraML', maxsplit=1)[0] + 'ProGraML'
print(repo_root)
#insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, repo_root)
repo_root = Path(repo_root)


#from deeplearning.ml4pl.models.ggnn.configs import ProGraMLBaseConfig as BaseConfig
from deeplearning.ml4pl.models.ggnn.run import MODEL_CLASSES

SCRIPTS_FOLDER = repo_root / 'deeplearning' / 'ml4pl' / 'poj104' / 'scripts'
SCRIPTS_FOLDER.mkdir(parents=True, exist_ok=True)


def stamp(stuff):
    hash_object = hashlib.sha1(str(stuff).encode("utf-8"))
    hex_dig = hash_object.hexdigest()
    return hex_dig[:7]


def config_generator(choices_dict, model):
    # GGNN DEVMAP HYPER OPT SERIES
    # devices = [0,1,2,3]
    value_list = [choices_dict[k] for k in choices_dict]
    configs = list(itertools.product(*value_list))
    for c in configs:
        manual_choices = dict(zip(choices_dict.keys(), c))
        if model is not None:
            Config = MODEL_CLASSES[model][1]
            config_dict = Config.from_dict(manual_choices).to_dict()
        else: # for transfer learning
            config_dict = manual_choices
        yield config_dict


def write_runscripts(subfolder, template, choices_dict, jobname, steps,
                     dataset, kfold, model=None, transfer=None, restore=None):
    configs = list(config_generator(choices_dict, model))
    outpath = SCRIPTS_FOLDER / subfolder
    outpath.mkdir(exist_ok = True, parents=True)

    print("Writing runscripts to:")
    print(str(outpath.absolute()))

    readme = open(outpath / "README.txt", "w")
    print(f"Writing runscripts to {subfolder} with the following choices_dict configuration:\n{choices_dict}\n", file=readme)

    for i, config in enumerate(configs):
        stmp = stamp(config)

        # log config to readme
        print(f"HyperOpt-{i:03d}-{stmp}: " + str(config), file=readme)

        # write the chained resubmit runscripts
        for j in range(steps):
            resto_str = ''
            if j > 0:
                resto_str = f'--restore_by_pattern {jobname}{i:03d}'
            template_format = {
                #"stamp": stmp,
                "i": i,
                "timelimit": "04:00:00",
                "config_str": str(config).replace('"', "'"),
                "subfolder": subfolder,
                "jobname": jobname,
                "restore_by_pattern": resto_str,
                "dataset": dataset,
                "kfold": kfold,
                "transfer": transfer,
                }
            if model is not None:
                template_format.update({'model': model})

            if restore is not None:
                template_format.update({'restore': restore})

            runscript = template.format(**template_format)
            #print(runscript)
            #print("\n")

            with open(outpath / f"run_{jobname}{i:03d}_{stmp}_step{j:02d}.sh", "w") as f:
                f.write(runscript)
                f.write(f"\n\n# HyperOpt-{i:03d}-{stmp}:")
                f.write(f"\n# {config}\n")

    readme.close()
    print("Success.")


if __name__ == "__main__":
    #from deeplearning.ml4pl.poj104.scripts.AULT_example_runscript_config import template, choices_dict, subfolder, resubmit_template, jobname, resubmit_times_per_job
    print('cannot directly run this here. run your experiment file instead!')
#    args = sys.argv[1:]
 #   if len(args) != 1:
  #      print('Usage: python AULT_your_experiment_script.py')
    #else:
   #    write_runscripts(subfolder=subfolder, template=template, choices_dict=choices_dict, jobname=jobname, steps=resubmit_times_per_job)
