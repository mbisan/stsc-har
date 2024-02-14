baseArguments = {
    "num_workers": 8,
    "dataset": "WISDM",
    "subjects_for_test": [
        [35],
        [34],
        [33],
        [32],
        [31],
        [30],
        [29],
        [28],
        [27],
        [26],
        [25],
        [24],
        [23],
        [22],
        [21],
        [20],
        [19],
        [18],
        [17],
        [16],
        [15],
        [14],
        [13],
        [12],
        [11],
        [10],
        [9],
        [8],
        [7],
        [6],
        [5],
        [4],
        [3],
        [2],
        [1],
        [0]
    ],
    "lr": 0.001,
    "n_val_subjects": 4,
    "arch": "utime",
    "latent_features": 12,
    "pooling": [[2, 2, 2]],
    "batch_size": 64,
    "overlap": 0,
    "max_epochs": 10,
    "training_dir": "training_segment",
    "weight_decayL1": 0.0001,
    "weight_decayL2": 0.00001
}

segExperiments = {
    "mode": "seg",
    "window_size": 128,
    "window_stride": 1,
}

RAM = 32
CPUS = 16

experiments = [segExperiments] #, dtwExperiments, dtwcExperiments, tsExperiments, gasfExperiments, gadfExperiments, mtffExperiments]


import os

from utils.argumentsegmentation import get_model_name, get_command

def create_jobs(args):
    modelname = get_model_name(args)
    modelname_clean = modelname.replace("|", "_")
    modelname_clean = modelname_clean.replace(",", "_")

    return modelname, modelname_clean, f'''#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={CPUS}
#SBATCH --mem={RAM}GB
#SBATCH --time=1-00:00:00
#SBATCH --job-name={modelname_clean}
#SBATCH --output=O-%x.%j.out
#SBATCH --error=E-%x.%j.err

cd $HOME/stsc-har

source $HOME/.bashrc
source activate dev2

python trainsegmentation.py {get_command(args)}
'''

class EmptyExperiment:
    pass

def produce_experiments(args):

    print("Experiments for mode", args["mode"])

    multiple_arguments = []
    for key, value in args.items():
        if "__" not in key:
            if isinstance(value, list):
                multiple_arguments.append((key, len(value)))

    total_experiments = 1
    for i in multiple_arguments:
        total_experiments *= i[1]

    experiment_arguments = [EmptyExperiment() for i in range(total_experiments)]

    for exp in experiment_arguments:
        exp.__dict__.update(args)

    k = 1
    for key, value in args.items():
        if "__" in key:
            continue

        if isinstance(value, list):
            n = len(value)
            if n>1:
                print("Argument with multiple values:", key, "with", n, "values")
            for i, experiment_arg in enumerate(experiment_arguments):
                setattr(experiment_arg, key, value[(i//k)%n])
            k *= n
        else:
            for experiment_arg in experiment_arguments:
                setattr(experiment_arg, key, value)
    
    jobs = []

    cache_dir = os.path.join("./", experiment_arguments[0].training_dir, "cache_jobs")
    if not os.path.exists(os.path.dirname(cache_dir)):
        os.mkdir(os.path.dirname(cache_dir))
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    print("Saving experiments to", cache_dir)

    for exp_arg in experiment_arguments:
        modelname, jobname, job = create_jobs(exp_arg)
        jobs.append("sbatch " + jobname + ".job")
        with open(os.path.join(cache_dir, jobname + ".job"), "w") as f:
            f.write(job)

    print("Created", len(jobs), "jobs")

    return jobs, cache_dir

if __name__ == "__main__":
    jobs = []
    for exp in experiments:
        j, cache_dir = produce_experiments({**baseArguments, **exp})
        jobs += j
    
    bash_script = "#!\\bin\\bash\n" + "\n".join(jobs)
    
    with open(os.path.join(cache_dir, "launch.sh"), "w") as f:
        f.write(bash_script)

    print(f"Number of experiments created: {len(jobs)}")
    print("launch.sh file at", cache_dir)
