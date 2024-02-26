import os

from utils.arguments import get_model_name, get_command
from experiment_definition import experiments, baseArguments

def create_jobs(args_list):
    modelname = get_model_name(args_list[0])
    modelname_clean = modelname.replace("|", "_")
    modelname_clean = modelname_clean.replace(",", "_")

    jobname = f"{args_list[0].mode}_{args_list[0].dataset}"

    job = f'''#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={args_list[0].cpus}
#SBATCH --mem={args_list[0].ram}GB
#SBATCH --time=1-00:00:00
#SBATCH --job-name={modelname_clean}
#SBATCH --output=O-%x.%j.out
#SBATCH --error=E-%x.%j.err

cd $HOME/stsc-har

source $HOME/.bashrc
source activate dev2
'''
    for args in args_list:
        job += f"\npython {args.command} {get_command(args)}"

    return modelname, jobname, job

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

    EXPS_PER_JOB = 4
    for i in range(0, len(experiment_arguments), EXPS_PER_JOB):
        modelname, jobname, job = create_jobs(experiment_arguments[i:min(i+EXPS_PER_JOB, len(experiment_arguments))])
        jobname += f"{i}.job"
        jobs.append("sbatch " + jobname)
        with open(os.path.join(cache_dir, jobname), "w") as f:
            f.write(job)

    print("Created", len(jobs), "jobs")

    return jobs, cache_dir

if __name__ == "__main__":
    for exp in experiments:
        jobs, cache_dir = produce_experiments({**baseArguments, **exp})
    
        bash_script = "#!\\bin\\bash\n" + "\n".join(jobs)
        
        with open(os.path.join(cache_dir, "launch.sh"), "w") as f:
            f.write(bash_script)

        print(f"Number of experiments created: {len(jobs)}")
        print("launch.sh file at", cache_dir)
