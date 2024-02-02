import os

from utils.arguments import get_model_name, get_command
from experiment_definition import experiments, baseArguments, RAM, CPUS

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

cd $HOME/s3ts

source $HOME/.bashrc
source activate dev

python training.py {get_command(args)}
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
