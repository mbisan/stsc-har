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

MHEALTH_LABELS = {
    1: "Standing still",
    2: "Sitting and relaxing",
    3: "Lying",
    4: "Walking",
    5: "Climb stairs",
    6: "Waist bends forward",
    7: "Frontal elevation of arms",
    8: "Knees bending (crouch)",
    9: "Cycling",
    10: "Jogging",
    11: "Running",
    12: "Jump front and back",
}

mhealth_label_mapping = np.zeros(13, dtype=np.int64)
mhealth_label_mapping[1:] = np.arange(12)
mhealth_label_mapping[0] = 100


PAMAP2_LABELS = {
    1: "lying",  #0
    2: "sitting", #1
    3: "standing", #2
    4: "walking", #3
    5: "running", #4
    6: "cycling", #5
    7: "Nordic walking", #6
    9: "watching TV", #7
    10: "computer work", #8
    11: "car driving", #9
    12: "ascending stairs", #10
    13: "descending stairs", #11
    16: "vacuum cleaning", #12
    17: "ironing", #13
    18: "folding laundry", #14
    19: "house cleaning", #15
    20: "playing soccer", #16
    24: "rope jumping", #17
    0: "other" #100 (transient activities)
}

pamap2_label_mapping = np.array(
    [100, 0, 1, 2, 3, 4, 5, 6, -1, 7, 8, 9, 10,
    11, -1, -1, 12, 13, 14, 15, 16, -1, -1, -1, 17], dtype=np.int64)

pamap2_basic_activity_recognition = np.array(
    [100, 0, 1, 1, 2, 3, 4, 100, # up to nordic walking, label "other": 5
    -1, 100, 100, 100, 100, 100, -1, -1,
    100, 100, 100, 100, 100, -1, -1, -1, 100], dtype=np.int64)

pamap2_background_activity_recognition = np.array(
    [100, 0, 1, 1, 2, 3, 4, 5, # up to nordic walking, label "other": 5
    -1, 100, 100, 100, 5, 5, -1, -1,
    5, 5, 100, 100, 100, -1, -1, -1, 5], dtype=np.int64)

pamap2_all_activity_recognition = np.array(
    [100, 0, 1, 2, 3, 4, 5, 6, # up to nordic walking, label "other": 5
    -1, 100, 100, 100, 7, 8, -1, -1, 9,
    10, 100, 100, 100, -1, -1, -1, 11], dtype=np.int64)

if __name__ == "__main__":
    for i, val in PAMAP2_LABELS.items():
        print(val, pamap2_basic_activity_recognition[i])
