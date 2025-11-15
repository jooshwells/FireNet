import torch
from torch.utils.data import Dataset
from PIL import Image

# Custom dataset for all data
class FirearmsDataset(Dataset):
    def __init__(self, fo_view, transform=None):
        self.transform = transform

        # Put filepaths/labels into lists since FiftyOne indexes by ID
        self.filepaths = [sample.filepath for sample in fo_view]
        self.labels = [1 if sample.is_firearm else 0 for sample in fo_view]

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        image = Image.open(self.filepaths[idx]).convert('RGB')

        if (self.transform):
            image = self.transform(image)

        return image, self.labels[idx]