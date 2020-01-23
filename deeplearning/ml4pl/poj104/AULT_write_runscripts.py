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


from deeplearning.ml4pl.models.ggnn.ggnn_config_poj104 import GGNNConfig

SCRIPTS_FOLDER = repo_root / 'deeplearning' / 'ml4pl' / 'scripts' / 'poj104'
SCRIPTS_FOLDER.mkdir(parents=True, exist_ok=True)


def stamp(stuff):
    hash_object = hashlib.sha1(str(stuff).encode("utf-8"))
    hex_dig = hash_object.hexdigest()
    return hex_dig[:7]


def config_generator(choices_dict):
    # GGNN DEVMAP HYPER OPT SERIES
    # devices = [0,1,2,3]
    value_list = [choices_dict[k] for k in choices_dict]
    configs = list(itertools.product(*value_list))
    for c in configs:
        manual_choices = dict(zip(choices_dict.keys(), c))
        config_dict = GGNNConfig.from_dict(manual_choices).to_dict()
        yield config_dict


def write_runscripts(subfolder, template, choices_dict, jobname, steps):
    configs = list(config_generator(choices_dict))
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
                }

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
    from deeplearning.ml4pl.poj104.AULT_example_runscript_config import template, choices_dict, subfolder, resubmit_template, jobname, resubmit_times_per_job

    args = sys.argv[1:]
    if len(args) > 0:
        print('Usage: python AULT_training_pipeline.py')
    else:
        write_runscripts(subfolder=subfolder, template=template, choices_dict=choices_dict, jobname=jobname, steps=resubmit_times_per_job)
