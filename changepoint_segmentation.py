import os
import json

import warnings

from argparse import ArgumentParser

import numpy as np
from sklearn.metrics import confusion_matrix

import torch

from pytorch_lightning import Trainer

from utils.helper_functions import load_dm

from nets.wrapper import ContrastiveWrapper, DFWrapper
from nets.metrics import print_cm, metrics_from_cm, group_classes

# shut up warnings
warnings.simplefilter("ignore", category=UserWarning)

def find_local_minima_indices(values, warning, alpha = 0.2, warning_time = 3, change_time = 2):

    cp = []

    running_mean = values[0]
    in_warning = 0
    in_change_point = 0
    waiting = False

    for i in range(values.shape[0]):
        running_mean = (1-alpha) * running_mean + alpha * values[i]
        if running_mean > warning:
            in_warning += 1

        compare_value = running_mean # / (1-alpha)

        if in_warning > warning_time and compare_value >= 2*warning:
            in_change_point +=1

        if in_change_point > change_time and compare_value >= 2*warning:
            if not waiting:
                cp.append(i)
                waiting = True
            in_change_point = 0
            in_warning = 0

        if running_mean < 2*warning:
            in_change_point = max(in_change_point - 1, 0)

        if running_mean < warning:
            in_warning = max(in_warning - 1, 0)
            waiting = False

    return cp

class Empty:
    '''Empty class'''
    window_size = 1

def evaluate_model(cpmodel_cp, clsmodel_cp, trainer: Trainer):

    with open(
        os.path.join(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(cpmodel_cp))),
            "results.json"), encoding="utf-8") as f:
        json_results = json.load(f)

    args = Empty()

    args.__dict__.update(json_results["args"])
    dm = load_dm(args)

    # pylint: disable=no-value-for-parameter
    cp_model = ContrastiveWrapper.load_from_checkpoint(checkpoint_path=cpmodel_cp)

    cp_model.eval()
    trainer.test(datamodule=dm, model=cp_model, verbose=False)

    classifier = DFWrapper.load_from_checkpoint(checkpoint_path=clsmodel_cp)

    wsize = args.window_size
    ovlp = 0
    diff = (cp_model.rpr[:(-wsize+ovlp), :] * cp_model.rpr[(wsize-ovlp):, :]).sum(-1)

    kernel = 0.90 ** np.arange(40, 0, -1) # similar to uniform mean
    kernel = kernel / np.sum(kernel)
    diff = -diff+1
    diff = np.convolve(diff.numpy(), kernel, mode="full")[:-39]

    change_points = find_local_minima_indices(diff, 0.01, 1, 1, 1)
    print("Number of change points:", len(change_points))
    # for i in range(len(change_points)):
    #     change_points[i] -= 20

    classifier.eval()

    classes = dm.stsds.SCS[dm.stsds.indices[dm.ds_test.indices]]
    sts = dm.stsds.STS[:, dm.stsds.indices[dm.ds_test.indices]]

    cp_ = [0] + change_points + [classes.shape[0]]

    j=0
    while j<len(cp_)-1:
        if cp_[j+1]-cp_[j]<16:
            new_id = int((cp_.pop(j) + cp_.pop(j))/2)
            cp_.insert(j, new_id)
        j+=1

    cp_[0] = 0
    cp_[-1] = classes.shape[0]

    out_repeated = []
    for j in range(len(cp_)-1):
        out = classifier(sts[None, :, cp_[j]:cp_[j+1]])
        out_repeated.append(out.squeeze().argmax(-1).repeat(cp_[j+1]-cp_[j]))

    cm = confusion_matrix(
        classes.numpy(), torch.cat(out_repeated).numpy(),
        labels=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11))

    # for key, value in metrics_from_cm(torch.from_numpy(cm)).items():
    #     print(key, value.mean().item())

    # print_cm(torch.from_numpy(cm), dm.n_classes)
    return cm

def main(args):

    cpmodels = []
    for dir_name in os.listdir(args.cpdir):
        if "cache" in dir_name:
            continue
        with open(os.path.join(args.cpdir, dir_name, "results.json"), "r", encoding="utf-8") as f:
            cpmodels.append(json.load(f))

    clsmodels = []
    for dir_name in os.listdir(args.clsdir):
        if "cache" in dir_name:
            continue
        with open(os.path.join(args.clsdir, dir_name, "results.json"), "r", encoding="utf-8") as f:
            clsmodels.append(json.load(f))

    cpmodels = sorted(cpmodels, key=lambda x: sum(x["args"]["subjects_for_test"]))
    clsmodels = sorted(clsmodels, key=lambda x: sum(x["args"]["subjects_for_test"]))

    tr = Trainer(accelerator="cpu")
    cm = [evaluate_model(
        os.path.join(os.path.dirname(args.cpdir), cpmodels[i]["path"]),
        os.path.join(os.path.dirname(args.clsdir), clsmodels[i]["path"]),
        tr
    ) for i in range(len(cpmodels))]
    cm = sum(cm)

    for key, value in metrics_from_cm(torch.from_numpy(cm)).items():
        print(key, value.mean().item())

    print_cm(torch.from_numpy(cm), cm.shape[0])

    cm2 = group_classes(torch.from_numpy(cm), [[6, 7, 8, 9, 10, 11]])
    for key, value in metrics_from_cm(cm2).items():
        print(key, value.mean().item())

    print_cm(cm2, cm2.shape[0])

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cpdir", type=str,
        help="Directory where change point models are")
    parser.add_argument("--clsdir", type=str,
        help="Directory where classifiers are")

    _args = parser.parse_args()

    main(_args)
