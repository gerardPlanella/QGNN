import os
import torch
from torch_geometric.data import Dataset

class IsingModelDataset(Dataset):
    def __init__(self, file_path, transform=None, pre_transform=None):
        super(IsingModelDataset, self).__init__(os.path.dirname(file_path), transform, pre_transform)

        self.file_path = file_path
        self.data_array = torch.load(self.file_path)
        print("Loaded %d data points from %s" % (len(self.data_array), self.file_path))

    def len(self):
        return len(self.data_array)

    def get(self, idx):
        # Return the preprocessed data point at the given index
        return self.data_array[idx]

    @classmethod
    def load(cls, file_path):
        """
        Load a dataset from a file.

        Args:
            file_path (str): Path to the file containing the dataset.

        Returns:
            dataset: An instance of the IsingModelDataset class containing the loaded data.
        """
        return IsingModelDataset(file_path)


if __name__ == "__main__":
    # Example usage:
    data_file = os.path.dirname(os.path.abspath(__file__)) + "/../../data/nk_1800_10.pt"
    # Load the dataset
    dataset = IsingModelDataset.load(data_file)
    # Access data points from the loaded dataset
    rand_idx = int(torch.randint(len(dataset), (1,)))
    data_point = dataset[rand_idx]
    print(data_point)
    print(data_point.y_node_rdms[0])
    print(data_point.y_edge_rdms[0])
