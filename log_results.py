'''
Usage:
    python log_results.py --dir DIRECTORY --out_name FILENAME (--csv) (--sum_labels 1,2 4,6)

Searchs for FILENAME in every directory inside DIRECTORY
FILENAME is a text file containing a python dict/json with keys containing "val"

For each directory prints the name of the directory, which is assumed to be the modelname
along with the corresponding "val" metrics inside FILENAME.

If CSV, the output is printed in csv format

If sum_labels and CM available, classes corresponding to the groups are summed and 
displayed too
'''

import os
from argparse import ArgumentParser
import json

import numpy as np
import torch
from nets.metrics import print_cm, metrics_from_cm, group_classes

def log_csv(entries, loaded):
    loaded_sorted_names = list(loaded.keys())
    loaded_sorted_names.sort()

    print("MODELNAME", *entries, sep=", ")

    for model_name in loaded_sorted_names:
        print(model_name, end=",")

        temp = ""
        for entry in entries:
            temp += f"{loaded[model_name][entry]}, "

        print(temp[:-2])

def log_normal(entries, loaded, name_linesize):
    print(f"{'MODELNAME':>{name_linesize}}", end=" |")
    for entry in entries:
        print(f"{entry:>10}", end="")
    print("\n" + "-"*name_linesize)

    loaded_sorted_names = list(loaded.keys())
    loaded_sorted_names.sort()
    for model_name in loaded_sorted_names:
        name_parts = []
        name = model_name
        while len(name)>0:
            if len(name)>name_linesize:
                name_parts.append(name[:name_linesize])
                name = name[name_linesize:]
            else:
                name_parts.append(name)
                break

        for i, part in enumerate(name_parts):
            e = " |" if i==(len(name_parts)-1) else " |\n"
            print(f"{part:>{name_linesize}}", end=e)

        for entry in entries:
            print(f"{loaded[model_name][entry]:>10.5f}", end="")
        print("\n" + "-"*name_linesize)

def load_data_files(dir_name, out_name):
    directories = list(
        filter(lambda x: os.path.isdir(os.path.join(dir_name, x)),
        os.listdir(dir_name))
    )

    loaded = {}

    for model_name in directories:
        data_file = os.path.join(dir_name, model_name, out_name)
        if os.path.exists(data_file):
            with open(data_file, "r", encoding="utf-8") as f:
                if data_file.endswith(".json"):
                    data = json.load(f)
                elif data_file.endswith(".dict"):
                    data = eval(f.read())

            if "cm" in data:
                if isinstance(data["cm"], str):
                    data["cm"] = eval(data["cm"])

            loaded[model_name] = data

    return loaded, directories

def display_cm_and_metrics(cm):
    print_cm(cm, cm.shape[0])
    copy = ""
    for key, value in metrics_from_cm(cm).items():
        value = value[~value.isnan()]
        m = value.mean().item()
        s = value.std().item()
        copy += f"{m:.5f}\t{s:.5f}\t"

        print(f"{key:>10}", ", ".join([f"{v:.2f}" for v in value]), end="")
        print("->", f"{value.mean().item():.5f}({value.std().item():.5f})")

    print("Copy->", copy, sep="")

def main(args):
    loaded, models = load_data_files(args.dir, args.out_name)

    entries = []
    for entry_dict in loaded.values():
        for entry in entry_dict.keys():
            entries.append(entry)
    entries = list(set(entries))

    if "cm" in entries:
        print("Compute aggregated Confusion Matrix")
        cm_summed = np.zeros_like(np.array(loaded[models[0]]["cm"]))

        for loaded_data in loaded.values():
            cm_summed += np.array(loaded_data["cm"])

        cm_summed = torch.from_numpy(cm_summed)
        display_cm_and_metrics(cm_summed)

        if len(args.sum_labels) > 0:
            print("Summing matrices:")
            new_cm = group_classes(cm_summed, args.sum_labels)
            display_cm_and_metrics(new_cm)

    entries = list(filter(lambda x: (("val" in x) or ("test" in x)) and (not "sub" in x), entries))
    entries.sort()

    if args.csv:
        log_csv(entries, loaded)
        return
    else:
        log_normal(entries, loaded, 85)

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--dir", type=str,
        help="Directory where model checkpoints and train outputs are")
    parser.add_argument("--out_name", type=str, default="results.dict",
        help="Name of the output file to look for")
    parser.add_argument("--csv", action="store_true",
        help="Print in csv format")
    parser.add_argument("--sum_labels", nargs="+", type=str, default=[],
        help="Combine labels")

    _args = parser.parse_args()

    _args.sum_labels = [[int(k) for k in i.split(",")] for i in _args.sum_labels]

    main(_args)
