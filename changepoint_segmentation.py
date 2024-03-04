import os
import json

from sklearn.metrics import confusion_matrix

import torch

from pytorch_lightning import Trainer

from utils.helper_functions import load_dm
from utils.arguments import get_parser

from nets.wrapper import ContrastiveWrapper, DFWrapper
from nets.metrics import print_cm, metrics_from_cm


CPMODEL_CP = "_tests/clr_UCI-HAR_21-20-19-18-17_4_40_1_bs128_lr0.001_l10_l20_cnn_gap_ts20_cnn_ts_decNone_1/version_1/checkpoints/epoch=18-step=7201-val_re=0.0000.ckpt"
CLASS_CP = "_window_classifier/ts_UCI-HAR_21-20-19-18-17_4_50_1_bs128_lr0.001_l10_l20_cnn_gap_ts32_mlp32_1_v1_0.1/version_0/checkpoints/epoch=13-step=5222-val_re=0.7381.ckpt"

parser = get_parser()
args = parser.parse_args('''
--dataset UCI-HAR --batch_size 128 --window_size 40 --normalize --subjects_for_test 21 20 19 18 17 
--max_epochs 10 --lr 0.01 --training_dir training_clr --n_val_subjects 4 --reduce_imbalance
--encoder_architecture cnn_gap_ts --encoder_features 20 --cf 0.001 --label_mode 48
--mode clr3 --voting 0'''.split())

with open(
    os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(CPMODEL_CP))),
        "results.json"), encoding="utf-8") as f:
    json_results = json.load(f)

class Empty:
    '''Empty class'''
    window_size = 1

args = Empty()

args.__dict__.update(json_results["args"])
dm = load_dm(args)

# pylint: disable=no-value-for-parameter
cpModel = ContrastiveWrapper.load_from_checkpoint(checkpoint_path=CPMODEL_CP)

cpModel.eval()
tr = Trainer()
tr.test(datamodule=dm, model=cpModel)

classifier = DFWrapper.load_from_checkpoint(checkpoint_path=CLASS_CP)

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

        if in_warning > warning_time and running_mean >= 2*warning:
            in_change_point +=1

        if in_change_point > change_time and running_mean >= 2*warning:
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

WSIZE = args.window_size
OVLP = 0
diff = (cpModel.rpr[:(-WSIZE+OVLP), :] * cpModel.rpr[(WSIZE-OVLP):, :]).sum(-1)

change_points = find_local_minima_indices(-diff + 1, 0.01, 0.2, 3, 1)
print("Number of change points:", len(change_points))

classifier.eval()

classes = dm.stsds.SCS[dm.stsds.indices[dm.ds_test.indices]]
sts = dm.stsds.STS[:, dm.stsds.indices[dm.ds_test.indices]]

cp_ = [0] + change_points + [classes.shape[0]]

out_repeated = []
for j in range(len(change_points)+1):
    out = classifier(sts[None, :, cp_[j]:cp_[j+1]])
    out_repeated.append(out.squeeze().argmax(-1).repeat(cp_[j+1]-cp_[j]))

cm = confusion_matrix(classes.numpy(), torch.cat(out_repeated).numpy())

for key, value in metrics_from_cm(torch.from_numpy(cm)).items():
    print(key, value.mean().item())

print_cm(torch.from_numpy(cm), dm.n_classes)
