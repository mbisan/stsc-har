import numpy as np

HARTH_LABELS = {
    1: "walking",
    2: "running",
    3: "shuffling",
    4: "stairs_up",
    5: "stairs_down",
    6: "standing",
    7: "sitting",
    8: "lying",
    13: "cycking_sit",
    14: "cycling_stand",
    130: "cycling_sit_idle",
    140: "cycling_stand_idle"
}

'''
With groups 2, 5, 11 (shuffling, standing and sycling_stand_idle)
and 6, 10 (sitting, cycling_sit_idle)

We get the following correspondence:

    0: "walking",
    1: "running",
    2: "stairs_up",
    3: "stairs_down",
    4: "lying",
    5: "cycking_sit",
    6: "cycling_stand",
    7: "sit" (sitting, cycling_sit_idle)
    8: "stand" (shuffling, standing and sycling_stand_idle)
'''

harth_label_mapping = np.zeros(141, dtype=np.int64)
harth_label_mapping[1:9] = np.arange(8)
harth_label_mapping[13] = 8
harth_label_mapping[14] = 9
harth_label_mapping[130] = 10
harth_label_mapping[140] = 11

UCI_HAR_LABELS = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",          
    7: "STAND_TO_SIT",
    8: "SIT_TO_STAND",
    9: "SIT_TO_LIE",
    10: "LIE_TO_SIT",
    11: "STAND_TO_LIE",
    12: "LIE_TO_STAND",
}

ucihar_label_mapping = np.zeros(13, dtype=np.int64)
ucihar_label_mapping[0] = 100
ucihar_label_mapping[1:] = np.arange(12)