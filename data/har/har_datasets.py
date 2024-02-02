import os
import numpy as np

from data.base import STSDataset

# Load datasets predefined

class UCI_HARDataset(STSDataset):

    def __init__(self,
            dataset_dir: str = None,
            wsize: int = 10,
            wstride: int = 1,
            normalize: bool = True,
            label_mapping: np.ndarray = None
            ) -> None:
        super().__init__(wsize=wsize, wstride=wstride)

        '''
            UCI-HAR dataset handler

            Inputs:
                dataset_dir: Directory of the prepare_har_dataset.py
                    processed dataset.
                wsize: window size
                wstride: window stride
        '''

        # load dataset
        files = os.listdir(os.path.join(dataset_dir, "processed"))
        files.sort()
        
        splits = [0]
        self.subject_indices = [0]

        STS = []
        SCS = []

        for f in files:
            # get separated STS
            segments = filter(
                lambda x: "sensor" in x,
                os.listdir(os.path.join(dataset_dir, "processed", f)))

            for s in segments:

                sensor_data = np.load(os.path.join(dataset_dir, "processed", f, s))
                STS.append(sensor_data)
                label_data = np.load(os.path.join(dataset_dir, "processed", f, s.replace("sensor", "label")))
                SCS.append(label_data)

                splits.append(splits[-1] + sensor_data.shape[0])

            self.subject_indices.append(splits[-1])

        self.splits = np.array(splits)

        self.STS = np.concatenate(STS).T
        self.SCS = np.squeeze(np.concatenate(SCS).astype(np.int32))

        self.label_mapping = label_mapping
        if not self.label_mapping is None:
            self.SCS = self.label_mapping[self.SCS]

        self.indices = np.arange(self.SCS.shape[0])
        for i in range(wsize * wstride):
            self.indices[self.splits[:-1] + i] = 0
        self.indices[self.SCS == 100] = 0 # remove observations with no label
        self.indices = self.indices[np.nonzero(self.indices)]

        if normalize:
            self.normalizeSTS("normal")

class HARTHDataset(STSDataset):

    def __init__(self,
            dataset_dir: str = None,
            wsize: int = 10,
            wstride: int = 1,
            normalize: bool = True,
            label_mapping: np.ndarray = None
            ) -> None:
        super().__init__(wsize=wsize, wstride=wstride)

        '''
            HARTH dataset handler

            Inputs:
                dataset_dir: Directory of the prepare_har_dataset.py
                    processed dataset.
                wsize: window size
                wstride: window stride
        '''

        # load dataset
        files = list(filter(
            lambda x: ".csv" in x,
            os.listdir(os.path.join(dataset_dir, "harth")))
        )
        files.sort()
        
        splits = [0]

        self.subject_indices = [0]

        STS = []
        SCS = []
        for f in files:
            # get separated STS
            segments = filter(
                lambda x: "acc" in x,
                os.listdir(os.path.join(dataset_dir, f[:-4])))

            for s in segments:

                sensor_data = np.load(os.path.join(dataset_dir, f[:-4], s))
                STS.append(sensor_data)
                label_data = np.load(os.path.join(dataset_dir, f[:-4], s.replace("acc", "label")))
                SCS.append(label_data)

                splits.append(splits[-1] + sensor_data.shape[0])

            self.subject_indices.append(splits[-1])

        self.splits = np.array(splits)

        self.STS = np.concatenate(STS).T
        self.SCS = np.squeeze(np.concatenate(SCS).astype(np.int32))

        self.label_mapping = label_mapping
        if not self.label_mapping is None:
            self.SCS = self.label_mapping[self.SCS]

        self.indices = np.arange(self.SCS.shape[0])
        for i in range(wsize * wstride):
            self.indices[self.splits[:-1] + i] = 0
        self.indices = self.indices[np.nonzero(self.indices)]

        if normalize:
            self.normalizeSTS("normal")

class WISDMDataset(STSDataset):

    def __init__(self,
            dataset_dir: str = None,
            wsize: int = 10,
            wstride: int = 1,
            normalize: bool = True,
            label_mapping: np.ndarray = None
            ) -> None:
        super().__init__(wsize=wsize, wstride=wstride)

        '''
            WISDM dataset handler

            Inputs:
                dataset_dir: Directory of the prepare_har_dataset.py
                    processed dataset.
                wsize: window size
                wstride: window stride
        '''

        # load dataset
        files = list(filter(
            lambda x: "sensor.npy" in x,
            os.listdir(os.path.join(dataset_dir)))
        )
        files.sort()
        
        splits = [0]

        STS = []
        SCS = []
        for f in files:
            sensor_data = np.load(os.path.join(dataset_dir, f))
            STS.append(sensor_data)
            SCS.append(np.load(os.path.join(dataset_dir, f.replace("sensor", "class"))))

            splits.append(splits[-1] + sensor_data.shape[0])

        self.splits = np.array(splits)

        self.STS = np.concatenate(STS).T
        self.SCS = np.concatenate(SCS).astype(np.int32)

        self.label_mapping = label_mapping
        if not self.label_mapping is None:
            self.SCS = self.label_mapping[self.SCS]

        self.indices = np.arange(self.SCS.shape[0])
        for i in range(wsize * wstride):
            self.indices[self.splits[:-1] + i] = 0
        self.indices = self.indices[np.nonzero(self.indices)]

        if normalize:
            self.normalizeSTS("normal")
