import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset


class TSDataset(Dataset):
    def __init__(self, data_length):
        self.data_length = data_length
        self.data = self.create_data()

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        return self.data[idx]
    
    def create_data(self):
        # list of tensors, where every tensor is of shape (1000,)
        # every such tensor is a sin wave with random wave length and drift and a little bit noise
        # and random frequency and random amplitude
        data = []
        for i in range(self.data_length):
            x = torch.arange(0, 1000)
            drift = torch.randn(1) * 0.1
            noise = torch.randn(1000) * 0.001
            frequency = torch.randn(1) * 0.1
            amplitude = torch.randn(1) * 0.1
            sin_wave = amplitude * torch.sin(x * frequency + drift) + noise
            data.append(sin_wave)
        return data   

def create_dataloader(data_length, batch_size=32, shuffle=True):
    dataset = TSDataset(data_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset, dataloader


if __name__ == "__main__":
    # Example usage
    data_length = 100  # 100 samples
    dataset, dataloader = create_dataloader(data_length)

    for batch in dataloader:
        # plot the batch
        for i in range(len(batch)):
            plt.plot(batch[i].numpy())
        plt.title("Example Time Series")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.show()
        break 

