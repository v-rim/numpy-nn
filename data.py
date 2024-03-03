import numpy as np


class CSVDataLoader:
    def __init__(self, name, set_ratios=[1], batch_size=1, shuffle=False):
        arr = np.genfromtxt(f"{name}", delimiter=",", skip_header=1)
        if shuffle:
            rng = np.random.default_rng()
            rng.shuffle(arr)

        data = [arr]
        if batch_size > 0:
            data = np.array_split(data[0], len(data[0]) // batch_size)
        data = [np.array(batch) for batch in data]

        set_ratios = [0] + set_ratios
        set_indices = np.cumsum(np.array(set_ratios) / sum(set_ratios)) * len(data)
        self.datasets = [
            data[int(set_indices[i]) : int(set_indices[i + 1])]
            for i in range(len(set_indices) - 1)
        ]

    def get_set(self, set_id=0):
        # Returns a list of numpy arrays representing batches
        return self.datasets[set_id]

    def get_set_array(self, set_id=0):
        # Returns the set as a large concatenated array
        return np.concatenate(self.datasets[set_id], axis=0)
    
def save_array(array, name, header):
    with open(f"{name}.csv", "w") as f:
        f.write(f"{header}\n")
        for row in array:
            for i, val in enumerate(row):
                f.write(f"{val}")

                if i != len(row) - 1:
                    f.write(", ")
                else:
                    f.write("\n")

