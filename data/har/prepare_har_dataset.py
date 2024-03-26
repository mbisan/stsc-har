import os
import datetime
from io import StringIO

import re
import pandas
import numpy as np

from numba import jit

@jit(nopython=True)
def remove_obs(times: np.ndarray, min_dt: float) -> np.ndarray:
    # remove observations that leave less than min_dt time between them
    out_indices = np.zeros_like(times, dtype=np.bool_)
    curr_time = times[0]
    for i in range(1, out_indices.shape[0]):
        if times[i] - curr_time < min_dt:
            out_indices[i] = True
        else:
            curr_time = times[i]
    return out_indices

def prepare_harth(dataset_dir):
    # pylint: disable=too-many-locals

    total_obs = 0
    total_splits = 0

    for dir_name, _, files in os.walk(os.path.join(dataset_dir, "harth")):
        files.sort()

        for i, file in enumerate(files):
            print(file, end=" ")
            ds = pandas.read_csv(os.path.join(dir_name, file))

            ds["tstmp"] = pandas.to_datetime(ds["timestamp"])
            ds["dt"] = (ds["tstmp"] - ds["tstmp"][0]) / datetime.timedelta(milliseconds=1)

            remove = remove_obs(ds["dt"].to_numpy(), 15)
            print(f"Removed {np.sum(remove)} observations.")
            ds.drop(remove.nonzero()[0], inplace=True)
            ds = ds.reset_index()

            # changes of more than 1s without observations
            splits = [0] + (np.diff(ds["dt"].to_numpy()) > 2000).nonzero()[0].tolist() + [len(ds)]

            total_splits += len(splits) - 1
            total_obs += len(ds)

            sensor_data = ds[[
                "back_x", "back_y", "back_z",
                "thigh_x", "thigh_y", "thigh_z"]].to_numpy()
            label_data = ds[["label"]].to_numpy()

            if not os.path.exists(os.path.join(dataset_dir, f"subject{file.replace('.csv', '')}")):
                os.mkdir(os.path.join(dataset_dir, f"subject{file.replace('.csv', '')}"))

            for i in range(1, len(splits)):

                user_dir = os.path.join(dataset_dir, f"subject{file.replace('.csv', '')}")
                with open(os.path.join(user_dir, f"sensor{i:03d}.npy"), "wb") as f:
                    np.save(f, sensor_data[splits[i-1]:splits[i], :])

                with open(os.path.join(user_dir, f"label{i:03d}.npy"), "wb") as f:
                    np.save(f, label_data[splits[i-1]:splits[i]])

    print("Obs:", total_obs, "Splits:", total_splits)


def prepare_uci_har(dataset_dir):
    # load data
    files = os.listdir(os.path.join(dataset_dir, "RawData"))
    files.sort()

    total_obs = 0
    total_splits = 0

    label_data = pandas.read_csv(
        os.path.join(dataset_dir, "RawData", "labels.txt"),
        sep=" ", header=None).to_numpy().astype(np.int64)

    for file in [f for f in files if "acc" in f]:
        acc_data = pandas.read_csv(
            os.path.join(dataset_dir, "RawData", file), sep=" ", header=None).to_numpy()
        gyro_data = pandas.read_csv(
            os.path.join(dataset_dir, "RawData", file.replace("acc", "gyro")),
                sep=" ", header=None).to_numpy()

        assert acc_data.shape[0] == gyro_data.shape[0]

        sensor_data = np.hstack([acc_data, gyro_data])
        n_obs = sensor_data.shape[0]
        total_obs += n_obs
        total_splits += 1

        re_result = re.match(r"acc_exp(\d+)_user(\d+).txt", file)
        experiment_number = int(re_result.group(1))
        user_id = int(re_result.group(2))

        if not os.path.exists(os.path.join(dataset_dir, f"subject{user_id:02d}")):
            os.mkdir(os.path.join(dataset_dir, f"subject{user_id:02d}"))

        sensor_dir = os.path.join(
            dataset_dir, f"subject{user_id:02d}", f"sensor{experiment_number:02d}.npy")
        label_dir = os.path.join(
            dataset_dir, f"subject{user_id:02d}", f"label{experiment_number:02d}.npy")

        experiment_labels = np.zeros(n_obs)

        for row in label_data:
            exp_id, u_id, label, start, end = row
            if exp_id == experiment_number and u_id == user_id:
                experiment_labels[start:(end+1)] = label

        with open(sensor_dir, "wb") as f:
            np.save(f, sensor_data)

        with open(label_dir, "wb") as f:
            np.save(f, experiment_labels)

    print("Obs:", total_obs, "Splits:", total_splits)


def prepare_wisdm(dataset_dir):
    data_dir = os.path.join(dataset_dir, "WISDM_ar_v1.1")
    assert os.path.exists(data_dir)

    total_obs = 0
    total_splits = 0

    # clean original csv
    with open(os.path.join(data_dir, "WISDM_ar_v1.1_raw.txt"), "r", encoding="utf-8") as f:
        data = f.read()
    lines = list(map(lambda x: x.strip().strip(","), data.split(";")))
    lines = list(filter(lambda x: not ",0,0,0," in x, lines)) # remove missing values

    data_new = "\n".join(lines)

    act_label = {'Downstairs':0, 'Jogging':1, 'Sitting':2, 'Standing':3, 'Upstairs':4, 'Walking':5}
    df = pandas.read_csv(StringIO(data_new),
        header=None, names=["USER", "ACTIVITY", "TIMESTAMP", "acc_x", "acc_y", "acc_z"])

    df = df.dropna()
    df["dt"] = df["TIMESTAMP"] // 1000000

    for user_id in df["USER"].unique(): # order of appearance in the file, so it does not change
        if not os.path.exists(os.path.join(dataset_dir, f"subject{user_id:02d}")):
            os.mkdir(os.path.join(dataset_dir, f"subject{user_id:02d}"))

        user = df[df["USER"] == user_id]
        user = user.sort_values("TIMESTAMP")
        user = user.reset_index()
        user["dt_0"] = user["dt"] - user["dt"][0]

        remove = remove_obs(user["dt_0"].to_numpy(), 40) # at least 40ms between obs
        print(f"Removed {np.sum(remove)} observations.")
        user.drop(remove.nonzero()[0], inplace=True)

        acc_data = user[["acc_x", "acc_y", "acc_z"]].to_numpy()
        labels = np.array(list(map(lambda x: act_label[x], user["ACTIVITY"])))

        splits = [0] + (np.diff(user["dt"].to_numpy()) > 2000).nonzero()[0].tolist() + [len(user)]

        total_obs += acc_data.shape[0]
        total_splits += len(splits) - 1

        for i in range(1, len(splits)):

            user_dir = os.path.join(dataset_dir, f"subject{user_id:02d}")
            with open(os.path.join(user_dir, f"sensor{i:03d}.npy"), "wb") as f:
                np.save(f, acc_data[splits[i-1]:splits[i], :])

            with open(os.path.join(user_dir, f"label{i:03d}.npy"), "wb") as f:
                np.save(f, labels[splits[i-1]:splits[i]])

    print("Obs:", total_obs, "Splits:", total_splits)


if __name__ == "__main__":
    if not os.path.exists("./datasets"):
        os.mkdir("./datasets")

    prepare_uci_har("./datasets/UCI-HAPT")
    prepare_harth("./datasets/HARTH")
    prepare_wisdm("./datasets/WISDM")
