import numpy as np
import os
import nsf.nsf_utils as nsf_utils

from matplotlib import pyplot as plt
from torch.utils.data import Dataset


def load_power():
    def load_data():
        file = os.path.join(utils.get_data_root(), "power", "data.npy")
        return np.load(file)

    def load_data_split_with_noise():
        rng = np.random.RandomState(42)

        data = load_data()
        rng.shuffle(data)
        N = data.shape[0]

        data = np.delete(data, 3, axis=1)
        data = np.delete(data, 1, axis=1)
        ############################
        # Add noise
        ############################
        # global_intensity_noise = 0.1*rng.rand(N, 1)
        voltage_noise = 0.01 * rng.rand(N, 1)
        # grp_noise = 0.001*rng.rand(N, 1)
        gap_noise = 0.001 * rng.rand(N, 1)
        sm_noise = rng.rand(N, 3)
        time_noise = np.zeros((N, 1))
        # noise = np.hstack((gap_noise, grp_noise, voltage_noise, global_intensity_noise, sm_noise, time_noise))
        # noise = np.hstack((gap_noise, grp_noise, voltage_noise, sm_noise, time_noise))
        noise = np.hstack((gap_noise, voltage_noise, sm_noise, time_noise))
        data += noise

        N_test = int(0.1 * data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1 * data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]

        return data_train, data_validate, data_test

    def load_data_normalised():
        data_train, data_validate, data_test = load_data_split_with_noise()
        data = np.vstack((data_train, data_validate))
        mu = data.mean(axis=0)
        s = data.std(axis=0)
        data_train = (data_train - mu) / s
        data_validate = (data_validate - mu) / s
        data_test = (data_test - mu) / s

        return data_train, data_validate, data_test

    return load_data_normalised()


def save_splits():
    train, val, test = load_power()
    splits = (("train", train), ("val", val), ("test", test))
    for split in splits:
        name, data = split
        file = os.path.join(utils.get_data_root(), "power", "{}.npy".format(name))
        np.save(file, data)


def print_shape_info():
    train, val, test = load_power()
    print(train.shape, val.shape, test.shape)


class PowerDataset(Dataset):
    def __init__(self, split="train", frac=None):
        path = os.path.join(utils.get_data_root(), "power", "{}.npy".format(split))
        self.data = np.load(path).astype(np.float32)
        self.n, self.dim = self.data.shape
        if frac is not None:
            self.n = int(frac * self.n)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.n


def main():
    dataset = PowerDataset(split="train")
    print(type(dataset.data))
    print(dataset.data.shape)
    print(dataset.data.min(), dataset.data.max())
    plt.hist(dataset.data.reshape(-1), bins=250)
    plt.show()


if __name__ == "__main__":
    main()
