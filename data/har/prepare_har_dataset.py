import pandas
import numpy as np
import os
import datetime

import re
import wget

import zipfile
import tarfile

DATASETS = {
    "WARD": "https://people.eecs.berkeley.edu/~yang/software/WAR/WARD1.zip",
    "HASC": "http://bit.ly/i0ivEz",
    "UCI-HAR": "https://archive.ics.uci.edu/static/public/341/smartphone+based+recognition+of+human+activities+and+postural+transitions.zip",
    "WISDM": "https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz", # tar.gz
    "USC-HAD": "https://sipi.usc.edu/had/USC-HAD.zip",
    "OPPORTUNITY": "https://archive.ics.uci.edu/static/public/226/opportunity+activity+recognition.zip",
    "UMAFall": "https://figshare.com/ndownloader/articles/4214283/versions/7",
    "UDC-HAR": "https://lbd.udc.es/research/real-life-HAR-dataset/data_raw.zip",
    "HARTH": "http://www.archive.ics.uci.edu/static/public/779/harth.zip",
    "UniMiB-SHAR": "https://www.dropbox.com/s/x2fpfqj0bpf8ep6/UniMiB-SHAR.zip?dl=1",
    "REALDISP": "https://archive.ics.uci.edu/static/public/305/realdisp+activity+recognition+dataset.zip",
    "DAPHNET-FOG": "https://archive.ics.uci.edu/static/public/245/daphnet+freezing+of+gait.zip",
    "MHEALTH": "https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip",
    "TempestaTMD": "https://tempesta.cs.unibo.it/projects/us-tm2017/static/dataset/raw_data/raw_data.tar.gz", # tar
    "MHEALTH": "https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip",
    "PAMAP2": "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"
}

def download(dataset_name = "all", dataset_dir=None):

    if dataset_name == "all":
        for name in DATASETS.keys():
            download(dataset_name=name, dataset_dir=dataset_dir)
        return None
    
    assert dataset_name in DATASETS.keys()

    if not os.path.exists(f"{dataset_dir}/{dataset_name}"):
        os.mkdir(f"{dataset_dir}/{dataset_name}")

    #check if directory is empty
    if not os.listdir(f"{dataset_dir}/{dataset_name}/"):
        # download zipped dataset
        print(f"Downloading dataset {dataset_name}")
        file = wget.download(DATASETS[dataset_name], out=f"{dataset_dir}/{dataset_name}/")
        print(f"{dataset_name} downloaded to {file}")

def unpack(dataset_name = "all", dataset_dir=None):

    if dataset_name == "all":
        for name in DATASETS.keys():
            unpack(dataset_name=name, dataset_dir=dataset_dir)
        return None
    
    if not os.path.exists(f"{dataset_dir}/{dataset_name}") or not os.listdir(f"{dataset_dir}/{dataset_name}/"):
        return print(f"Dataset is not downloaded to {dataset_dir}/{dataset_name}/")
    
    assert dataset_name in DATASETS.keys()
    assert os.path.exists(f"{dataset_dir}/{dataset_name}")

    files = os.listdir(f"{dataset_dir}/{dataset_name}")
    assert len(files) > 0

    if len(files) > 1:
        return print(f"Dataset {dataset_name} already unpacked")

    if dataset_name in ["WISDM", "TempestaTMD"]:
        unpack_tar(dataset_dir, dataset_name)
    else:
        print(os.path.join(f"{dataset_dir}/{dataset_name}", files[-1]))
        # use zipfile to decompress
        with zipfile.ZipFile(os.path.join(f"{dataset_dir}/{dataset_name}", files[-1]), "r") as zip_ref:
            zip_ref.extractall(f"{dataset_dir}/{dataset_name}")

        # unpack zip files inside zip file
        files_new = os.listdir(f"{dataset_dir}/{dataset_name}")
        for new_file in files_new:
            if new_file[-3:] == "zip" and new_file not in files:
                with zipfile.ZipFile(os.path.join(f"{dataset_dir}/{dataset_name}", new_file), "r") as zip_ref:
                    zip_ref.extractall(f"{dataset_dir}/{dataset_name}")

def unpack_tar(dataset_dir, dataset_name):
    ds_dir = os.path.join(dataset_dir, dataset_name)
    assert os.path.exists(ds_dir)

    files = os.listdir(ds_dir)
    assert len(files) == 1

    f = tarfile.open(os.path.join(ds_dir, files[0]))

    f.extractall(ds_dir)

############################################################################################

def prepare_harth(dataset_dir):

    ds = []

    counts = {}
    event_length = {}

    for dir, _, files in os.walk(os.path.join(dataset_dir, "harth")):
        files.sort()

        for i, file in enumerate(files):
            print(file)
            ds = pandas.read_csv(os.path.join(dir, file))

            ds["tstmp"] = pandas.to_datetime(ds["timestamp"])
            ds["dt"] = (ds["tstmp"] - ds["tstmp"][0]) / datetime.timedelta(milliseconds=1)

            # check if the sampling rate is correct, at 50hz
            # all .csv files have some timestamps with jumps of less or more than 15ms
            # for jumps with less than 10ms, we remove observations, for jumps with more than 25ms, we consider a new STS

            remove = []
            j = 0
            for i in range(len(ds) - 1):
                diff = ds["dt"][i + 1] - ds["dt"][j]
                if diff < 15:
                    remove.append(i + 1)
                else:
                    j = i + 1

            print(f"Removed {len(remove)} observations.")
            ds.drop(remove, inplace=True)
            ds = ds.reset_index()

            splits = []
            last = 0
            for i in range(len(ds) - 1):
                if (ds["dt"][i + 1] - ds["dt"][i]) > 500: # I get the same number of splits for 25ms, 50ms or 100 ms
                    splits.append(ds.loc[last:i])
                    last = i + 1
            splits.append(ds.loc[last:len(ds)])

            if not os.path.exists(os.path.join(dataset_dir, f"{file.replace('.csv', '')}")):
                os.mkdir(os.path.join(dataset_dir, f"{file.replace('.csv', '')}"))

            for i, sp in enumerate(splits):
                labels = sp[["label"]].to_numpy()

                with open(os.path.join(dataset_dir, f"{file.replace('.csv', '')}/acc{i}.npy"), "wb") as f:
                    np.save(f, sp[["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]].to_numpy())
                
                with open(os.path.join(dataset_dir, f"{file.replace('.csv', '')}/label{i}.npy"), "wb") as f:
                    np.save(f, labels)
                
                # update class counts
                lb, c = np.unique(labels, return_counts=True)
                for i, l in enumerate(lb):
                    counts[l] = counts.get(l, 0) + c[i]

                # update event counts and event length
                current_event = 0
                for i in range(1, labels.size - 1):
                    if labels[i] != labels[current_event]:
                        event_length[labels[current_event].item()] = \
                            event_length.get(labels[current_event].item(), []) + [i - current_event]
                        current_event = i
                
                # last event
                event_length[labels[current_event].item()] = \
                            event_length.get(labels[current_event].item(), []) + [labels.size - current_event]
    
    # print statistics
    total = sum(counts.values())
    print(f"Total number of observations: {total}")

    for c in counts.keys():
            print(f"{len(event_length[c])} events in class {c},")
            print(f"\twith size (min) {min(event_length[c])}, (max) {max(event_length[c])}, (mean) {np.mean(event_length[c])}")
            print(f"\t{counts[c]} observations ({(counts[c]/total):.2f})")


def prepare_uci_har(dataset_dir):
    # load data
    files = os.listdir(os.path.join(dataset_dir, "RawData"))
    files.sort()

    data = {}

    total_points = 0
    for file in files:
        if file == "labels.txt":
            data[file] = pandas.read_csv(os.path.join(dataset_dir, "RawData", file), sep=" ", header=None).to_numpy().astype(np.int64)
            continue

        data[file] = pandas.read_csv(os.path.join(dataset_dir, "RawData", file), sep=" ", header=None).to_numpy()

        obs = data[file].shape[0]
        total_points += obs

    print(total_points)

    # split into subjects
    if not os.path.exists(os.path.join(dataset_dir, "processed")):
        os.mkdir(os.path.join(dataset_dir, "processed"))

    user_splits = {}

    for j, file in enumerate([f for f in files if "acc" in f]):
        n_obs = data[file].shape[0]

        re_result = re.match(r"acc_exp(\d+)_user(\d+).txt", file)

        experiment_number = re_result.group(1)
        user_id = re_result.group(2)

        exp_n_int = int(experiment_number)
        user_id_int = int(user_id)

        if not os.path.exists(os.path.join(dataset_dir, "processed", f"subject{user_id}")):
            os.mkdir(os.path.join(dataset_dir, "processed", f"subject{user_id}"))
        
        split_n = user_splits.get(user_id, 0)
        sensor_dir = os.path.join(dataset_dir, "processed", f"subject{user_id}", f"sensor{split_n}.npy")
        label_dir = os.path.join(dataset_dir, "processed", f"subject{user_id}", f"label{split_n}.npy")

        experiment_labels = np.zeros(n_obs)

        for row in data["labels.txt"]:
            exp_id, u_id, label, start, end = row
            if exp_id == exp_n_int and u_id == user_id_int:
                experiment_labels[start:(end+1)] = label

        with open(sensor_dir, "wb") as f:
            np.save(f, np.hstack((data[file], data[file.replace("acc", "gyro")])))
                    
        with open(label_dir, "wb") as f:
            np.save(f, experiment_labels)

        user_splits[user_id] = user_splits.get(user_id, 0) + 1


def prepare_wisdm(dataset_dir):
    data_dir = os.path.join(dataset_dir, "WISDM_ar_v1.1")
    assert os.path.exists(data_dir)

    # clean original csv
    with open(os.path.join(data_dir, "WISDM_ar_v1.1_raw.txt"), "r") as f:
        data = f.read()
    lines = list(map(lambda x: x.strip().strip(","), data.split(";")))

    data_new = "\n".join(lines)
    with open(os.path.join(data_dir, "clean.csv"), "w") as f:
        f.write(data_new)
    
    act_label = {'Downstairs':0, 'Jogging':1, 'Sitting':2, 'Standing':3, 'Upstairs':4, 'Walking':5}
    df = pandas.read_csv(os.path.join(data_dir, "clean.csv"), 
        header=None, names=["USER", "ACTIVITY", "TIMESTAMP", "acc_x", "acc_y", "acc_z"])
    
    df = df.dropna()

    for user_id in df["USER"].unique(): # order of appearance in the file, so it does not change
        user = df[df["USER"] == user_id]
        usersorted = user.sort_values("TIMESTAMP")

        acc_data = usersorted[["acc_x", "acc_y", "acc_z"]].to_numpy()
        timestamps = usersorted["TIMESTAMP"].to_numpy() / 50000000
        diff_ts = np.diff(timestamps)

        assert np.sum((diff_ts<1.05)*(diff_ts>0.95))

        labels = list(map(lambda x: act_label[x], usersorted["ACTIVITY"]))

        with open(os.path.join(dataset_dir, f"subject_{user_id}_sensor.npy"), "wb") as f:
            np.save(f, acc_data)
        with open(os.path.join(dataset_dir, f"subject_{user_id}_class.npy"), "wb") as f:
            np.save(f, np.array(labels, dtype=int))

def prepare_mhealth(dataset_dir):
    subject_files = list(filter(lambda x: "subject" in x, os.listdir(os.path.join(dataset_dir, "MHEALTHDATASET"))))
    subject_files.sort()

    mhealth_names = [
        "C-acc_x",  # Column 1: acceleration from the chest sensor (X axis)
        "C-acc_y",  # Column 2: acceleration from the chest sensor (Y axis)
        "C-acc_z",  # Column 3: acceleration from the chest sensor (Z axis)
        "ECG_lead1",  # Column 4: electrocardiogram signal (lead 1)
        "ECG_lead2",  # Column 5: electrocardiogram signal (lead 2)
        "LA-acc_x",  # Column 6: acceleration from the left-ankle sensor (X axis)
        "LA-acc_y",  # Column 7: acceleration from the left-ankle sensor (Y axis)
        "LA-acc_z",  # Column 8: acceleration from the left-ankle sensor (Z axis)
        "LA-gyro_x",  # Column 9: gyro from the left-ankle sensor (X axis)
        "LA-gyro_y",  # Column 10: gyro from the left-ankle sensor (Y axis)
        "LA-gyro_z",  # Column 11: gyro from the left-ankle sensor (Z axis)
        "LA-mag_x",  # Column 12: magnetometer from the left-ankle sensor (X axis)
        "LA-mag_y",  # Column 13: magnetometer from the left-ankle sensor (Y axis)
        "LA-mag_z",  # Column 14: magnetometer from the left-ankle sensor (Z axis)
        "RA-acc_x",  # Column 15: acceleration from the right-lower-arm sensor (X axis)
        "RA-acc_y",  # Column 16: acceleration from the right-lower-arm sensor (Y axis)
        "RA-acc_z",  # Column 17: acceleration from the right-lower-arm sensor (Z axis)
        "RA-gyro_x",  # Column 18: gyro from the right-lower-arm sensor (X axis)
        "RA-gyro_y",  # Column 19: gyro from the right-lower-arm sensor (Y axis)
        "RA-gyro_z",  # Column 20: gyro from the right-lower-arm sensor (Z axis)
        "RA-mag_x",  # Column 21: magnetometer from the right-lower-arm sensor (X axis)
        "RA-mag_y",  # Column 22: magnetometer from the right-lower-arm sensor (Y axis)
        "RA-mag_z",  # Column 23: magnetometer from the right-lower-arm sensor (Z axis)
        "Label"  # Column 24: Label (0 for the null class)
    ]

    sensor_groups = {
        "chest_acc": ["C-acc_x", "C-acc_y", "C-acc_z"],
        "ECG": ["ECG_lead1", "ECG_lead2"],
        "left_ankle": ["LA-acc_x", "LA-acc_y", "LA-acc_z", "LA-gyro_x", "LA-gyro_y", "LA-gyro_z", "LA-mag_x", "LA-mag_y", "LA-mag_z"],
        "right_lower_arm": ["RA-acc_x", "RA-acc_y", "RA-acc_z", "RA-gyro_x", "RA-gyro_y", "RA-gyro_z", "RA-mag_x", "RA-mag_y", "RA-mag_z"],
        "label": ["Label"]
    }

    for subject in subject_files:
        df = pandas.read_csv(os.path.join(dataset_dir, "MHEALTHDATASET", subject), 
            sep="\t", header=None, 
            names=mhealth_names)
        
        subject_folder = os.path.join(dataset_dir, subject.replace(".log", ""))

        if not os.path.exists(subject_folder):
            os.mkdir(subject_folder)

        for name, value in sensor_groups.items():

            with open(os.path.join(subject_folder, f"{name}.npy"), "wb") as f:
                np.save(f, df[value].to_numpy())

def prepare_pamap2(dataset_dir):
    subject_files_0 = list(filter(lambda x: "subject" in x, os.listdir(os.path.join(dataset_dir, "PAMAP2_Dataset", "Protocol"))))
    subject_files_1 = list(filter(lambda x: "subject" in x, os.listdir(os.path.join(dataset_dir, "PAMAP2_Dataset", "Optional"))))

    column_names = [
        "timestamp",  # Column 1: timestamp (s)
        "activityID",  # Column 2: activityID
        "heart_rate",  # Column 3: heart rate (bpm)
        # Hand IMU
        "hand_temp",  # Column 4: temperature (°C) for hand IMU
        "hand_acc_x", "hand_acc_y", "hand_acc_z",  # Column 5-7: 3D-acceleration data for hand IMU (ms-2)
        "hand_acc_x_6g", "hand_acc_y_6g", "hand_acc_z_6g",  # Column 8-10: 3D-acceleration data (±6g) for hand IMU (ms-2)
        "hand_gyro_x", "hand_gyro_y", "hand_gyro_z",  # Column 11-13: 3D-gyroscope data for hand IMU (rad/s)
        "hand_mag_x", "hand_mag_y", "hand_mag_z",  # Column 14-16: 3D-magnetometer data for hand IMU (μT)
        "hand_orientation_1", "hand_orientation_2", "hand_orientation_3", "hand_orientation_4",  # Column 17-20: orientation data for hand IMU
        # Chest IMU
        "chest_temp",  # Column 21: temperature (°C) for chest IMU
        "chest_acc_x", "chest_acc_y", "chest_acc_z",  # Column 22-24: 3D-acceleration data for chest IMU (ms-2)
        "chest_acc_x_6g", "chest_acc_y_6g", "chest_acc_z_6g",  # Column 25-27: 3D-acceleration data (±6g) for chest IMU (ms-2)
        "chest_gyro_x", "chest_gyro_y", "chest_gyro_z",  # Column 28-30: 3D-gyroscope data for chest IMU (rad/s)
        "chest_mag_x", "chest_mag_y", "chest_mag_z",  # Column 31-33: 3D-magnetometer data for chest IMU (μT)
        "chest_orientation_1", "chest_orientation_2", "chest_orientation_3", "chest_orientation_4",  # Column 34-37: orientation data for chest IMU
        # Ankle IMU
        "ankle_temp",  # Column 38: temperature (°C) for ankle IMU
        "ankle_acc_x", "ankle_acc_y", "ankle_acc_z",  # Column 39-41: 3D-acceleration data for ankle IMU (ms-2)
        "ankle_acc_x_6g", "ankle_acc_y_6g", "ankle_acc_z_6g",  # Column 42-44: 3D-acceleration data (±6g) for ankle IMU (ms-2)
        "ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z",  # Column 45-47: 3D-gyroscope data for ankle IMU (rad/s)
        "ankle_mag_x", "ankle_mag_y", "ankle_mag_z",  # Column 48-50: 3D-magnetometer data for ankle IMU (μT)
        "ankle_orientation_1", "ankle_orientation_2", "ankle_orientation_3", "ankle_orientation_4",  # Column 51-54: orientation data for ankle IMU
    ]

    sensor_groups = {
        "time": "timestamp",  # Column 1: timestamp (s)
        "label": "activityID",  # Column 2: activityID
        "HR": "heart_rate",
        "hand_temp": "hand_temp",
        "hand_acc_16g": ["hand_acc_x", "hand_acc_y", "hand_acc_z"],
        "hand_acc_6g": ["hand_acc_x_6g", "hand_acc_y_6g", "hand_acc_z_6g"],
        "hand_gyro": ["hand_gyro_x", "hand_gyro_y", "hand_gyro_z"],
        "hand_mag": ["hand_mag_x", "hand_mag_y", "hand_mag_z"],
        "hand_orientation": ["hand_orientation_1", "hand_orientation_2", "hand_orientation_3", "hand_orientation_4"],
        "chest_temp": "chest_temp",
        "chest_acc_16g": ["chest_acc_x", "chest_acc_y", "chest_acc_z"],
        "chest_acc_6g": ["chest_acc_x_6g", "chest_acc_y_6g", "chest_acc_z_6g"],
        "chest_gyro": ["chest_gyro_x", "chest_gyro_y", "chest_gyro_z"],
        "chest_mag": ["chest_mag_x", "chest_mag_y", "chest_mag_z"],
        "chest_orientation": ["chest_orientation_1", "chest_orientation_2", "chest_orientation_3", "chest_orientation_4"],
        "ankle_temp": "ankle_temp",
        "ankle_acc_16g": ["ankle_acc_x", "ankle_acc_y", "ankle_acc_z"],
        "ankle_acc_6g": ["ankle_acc_x_6g", "ankle_acc_y_6g", "ankle_acc_z_6g"],
        "ankle_gyro": ["ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z"],
        "ankle_mag": ["ankle_mag_x", "ankle_mag_y", "ankle_mag_z"],
        "ankle_orientation": ["ankle_orientation_1", "ankle_orientation_2", "ankle_orientation_3", "ankle_orientation_4"]
    }

    for i, subjects in enumerate([subject_files_0, subject_files_1]):
        for subject in subjects:
            df = pandas.read_csv(os.path.join(dataset_dir, "PAMAP2_Dataset", "Protocol" if i==0 else "Optional", subject), 
                sep=" ", header=None, 
                names=column_names)
            
            subject_folder = os.path.join(dataset_dir, subject.replace(".dat", ""))

            if not os.path.exists(subject_folder):
                os.mkdir(subject_folder)

            for name, value in sensor_groups.items():

                with open(os.path.join(subject_folder, f"{name}{i}.npy"), "wb") as f:
                    np.save(f, df[value].to_numpy())

if __name__ == "__main__":
    if not os.path.exists("./datasets"):
        os.mkdir("./datasets")

    download("UCI-HAR", "./datasets")
    download("HARTH", "./datasets")
    download("WISDM", "./datasets")
    download("MHEALTH", "./datasets")
    download("PAMAP2", "./datasets")

    unpack("PAMAP2", "./datasets")
    unpack("MHEALTH", "./datasets")
    unpack("UCI-HAR", "./datasets")
    unpack("HARTH", "./datasets")
    unpack("WISDM", "./datasets")
    
    prepare_mhealth("./datasets/MHEALTH")
    prepare_pamap2("./datasets/PAMAP2")
    prepare_uci_har("./datasets/UCI-HAR")
    prepare_harth("./datasets/HARTH")
    prepare_wisdm("./datasets/WISDM")

    pass
