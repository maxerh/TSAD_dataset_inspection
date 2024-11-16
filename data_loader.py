import os
import numpy as np
import pandas as pd
import pickle

main_data_path = "/media/4TB_mntpnt/datasets"

class DataLoader:
    def __init__(self):
        super().__init__()
        self.data = None
        self.name = None

    def get_data_dim(self):
        if self.name == 'SMAP':
            return 25
        elif self.name == 'MSL':
            return 55
        elif self.name == 'SMD':
            return 38
        elif str(self.name).startswith("machine"):
            return 38
        elif self.name == 'PSM':
            return 25
        elif self.name == 'SWaT':
            return 50
        elif self.name == 'WADI':
            return 127
        else:
            raise ValueError('unknown dataset ' + str(self.name))

    def read_pkl(self, file: str):
        with open(file, 'rb') as f:
            data = pickle.load(f)
        return data

    def read_csv(self, file: str):
        with open(file, 'rb') as f:
            df = pd.read_csv(f)
        data = df.to_numpy()
        return data

    def load_dataset(self, dsname: str, entity=None):
        self.name = dsname
        if not entity:
            entity = self.name.lower()
        datapath = os.path.join(main_data_path, self.name, entity)
        if self.name in ["SMD", "MSL", "SMAP", "SWaT", "WADI"]:
            self.data = self.read_pkl(f"{datapath}_train.pkl")
        elif self.name == "PSM":
            self.data = self.read_csv(f"{datapath}_train.csv")
            self.data = np.nan_to_num(self.data)
        else:
            raise ValueError('unknown dataset ' + str(self.name))