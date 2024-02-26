# %%
import torch
from torch import nn

import numpy as np
from torch.nn import functional as F

import numpy as np
import torch

import matplotlib.pyplot as plt

from utils.helper_functions import load_dm
from utils.arguments import get_parser

from nets.wrapper import *

from pytorch_lightning import Trainer

import json

from sklearn.metrics import confusion_matrix
from nets.metrics import print_cm, metrics_from_cm

# %%
parser = get_parser()
args = parser.parse_args('''
--dataset UCI-HAR --batch_size 128 --window_size 25 --normalize --subjects_for_test 0 1 2 3 4 
--max_epochs 10 --lr 0.01 --training_dir training_clr --n_val_subjects 4 --reduce_imbalance
--encoder_architecture cnn_gap_ts --encoder_features 20 --cf 0.001 --label_mode 48 --mode clr3'''.split())

dm = load_dm(args)

# %%
cpModel = ContrastiveWrapper.load_from_checkpoint(
    checkpoint_path="_tests/clr3_UCI-HAR_0-1-2-3-4_4_50_1_bs64_lr0.001_l10.0001_l21e-05_cnn_gap_ts24_mlpNone_1/version_3/checkpoints/epoch=12-step=9516-val_re=0.0000.ckpt")
values = json.load(open("_tests/clr3_UCI-HAR_0-1-2-3-4_4_50_1_bs64_lr0.001_l10.0001_l21e-05_cnn_gap_ts24_mlpNone_1/results.json", "r"))
threshold = values["test_th"]

cpModel.eval()
print(threshold)

# %%
classifier = DFWrapper.load_from_checkpoint(
    checkpoint_path="_tests/img_UCI-HAR_0-1-2-3-4_4_25_2_bs64_lr0.001_l10.0001_l21e-05_cnn_gap_img24_mlp32_1_m25_p25_r0.1_med1_500/version_0/checkpoints/epoch=17-step=13158-val_re=0.8536.ckpt"
)

classifier.eval()
print()

# %%
cpModel.window_size = 25

# %%
tr = Trainer()
data = tr.test(datamodule=dm, model=cpModel)

# %%
threshold = data[0]["test_th"]

# %%
MAX_SEGMENT_SIZE = 128
MIN_SEGMENT_SIZE = 12
CHANGE_POINT_WINDOW = 25

obs_to_classify = 0
change = False
time_last_change = 0
dissimilarity_index = dm.ds_test.indices[:-CHANGE_POINT_WINDOW]
times = []
classific = []
lbl = []

with torch.no_grad():
    for i, id in enumerate(dissimilarity_index):

        if (cpModel.dissimilarities[i] >= threshold and time_last_change >= MIN_SEGMENT_SIZE) or time_last_change > MAX_SEGMENT_SIZE:
            id_sts = dm.stsds.indices[id]
            class_prev = classifier(dm.stsds.STS[None, :, (id_sts - time_last_change):id_sts])
            times.append(time_last_change)
            classific.append(class_prev)
            time_last_change = 0

        time_last_change += 1

# %%
results = np.concatenate(classific)
pred = np.argmax(results, axis=-1)
y_pred = np.empty(sum(times), dtype=np.int64)

id = 0
for i, time in enumerate(times):
    y_pred[id:(id+time)] = pred[i]
    id += time

print(sum(times)/len(times))

# %%
y_true = dm.stsds.SCS[dm.stsds.indices[dm.ds_test.indices[:y_pred.shape[0]]]]

# %%
cm = confusion_matrix(y_true, y_pred)

# %%
metrics = metrics_from_cm(torch.from_numpy(cm))
metrics

# %%
for key, value in metrics.items():
    print(key, value.mean().item())

# %%
print_cm(torch.from_numpy(cm), dm.n_classes)


