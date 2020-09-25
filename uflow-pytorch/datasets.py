from torch.utils.data import Dataset as PytorchDataset

class Dataset(PytorchDataset):
    torch.nn.functional.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=False)

    def __len__(self):
        if self.return_indices:
            return self.to_frame - self.from_frame - 1
        else:
            return self.to_frame - self.from_frame - 1

    def __getitem__(self, idx):
        pass
