import os

from argparse import ArgumentParser

from utils.arguments import get_model_name, get_command
from utils.experiment_definition import experiments, baseArguments

def create_jobs(args_list):
    modelname = get_model_name(args_list[0])
    modelname_clean = modelname.replace("|", "_")
    modelname_clean = modelname_clean.replace(",", "_")

    jobname = f"{args_list[0].mode}_{args_list[0].dataset}"

    command = ""
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
        command += f"\npython {args.command} {get_command(args)}"

    job += command
    return jobname, job, command

class EmptyExperiment:
    pass

def produce_experiments(args, exps_per_job=3):

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
    commands = []

    save_dir = os.path.join("./", experiment_arguments[0].training_dir, "cache_jobs")
    if not os.path.exists(os.path.dirname(save_dir)):
        os.mkdir(os.path.dirname(save_dir))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    print("Saving experiments to", save_dir)

    for i in range(0, len(experiment_arguments), exps_per_job):
        last = min(i+exps_per_job, len(experiment_arguments))
        jobname, job, command = create_jobs(experiment_arguments[i:last])
        jobname += f"{i}.job"
        commands.append(command)
        jobs.append("sbatch " + jobname)
        with open(os.path.join(save_dir, jobname), "w", encoding="utf-8") as file:
            file.write(job)

    print("Created", len(jobs), "jobs")

    return jobs, commands, save_dir

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dir", type=str, default="./")
    parser.add_argument("--exps_per_job", type=int, default=3)
    parser.add_argument("--local", action="store_true")

    args_ = parser.parse_args()

    if not os.path.exists(args_.dir):
        os.mkdir(args_.dir)

    os.chdir(args_.dir)

    single_launch_script = "#!\\bin\\bash\n"

    for exp_args in experiments:
        all_jobs, all_commands, cache_dir = produce_experiments(
            {**baseArguments, **exp_args}, exps_per_job=args_.exps_per_job)

        if args_.local:
            single_launch_script += f"cd {cache_dir}\n" + "\n".join(all_commands) + "\ncd ../..\n"
        else:
            single_launch_script += f"cd {cache_dir}\n" + "\n".join(all_jobs) + "\ncd ../..\n"

        # with open(os.path.join(cache_dir, "launch.sh"), "w", encoding="utf-8") as f:
        #     f.write("#!\\bin\\bash\n" + "\n".join(all_jobs)) # bash script

        print(f"Number of experiments created: {len(all_jobs)}")

    with open(os.path.join("launch.sh"), "w", encoding="utf-8") as f:
        f.write(single_launch_script) # bash script
    print("launch.sh file at", args_.dir)
