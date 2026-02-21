import os
from torch.utils.data import Dataset
import torchvision
class FilteredImageFolder(Dataset):
    def __init__(self, imagefolder: torchvision.datasets.ImageFolder, subset_filenames: list[str]):
        self.imagefolder = imagefolder
        # Map: filename -> index in ImageFolder
        filename_to_index = {os.path.basename(path): i for i, (path, _) in enumerate(imagefolder.samples)} # saving all the filenames of my images in a dict {filename: label} ; imagefolder.samples is a list of tuples (fullpath, label) for all the images in the dataset
        # Preserve subset_filenames order
        self.indices = [filename_to_index[fname] for fname in subset_filenames if fname in filename_to_index] # stores the indices in the indicated order passed they will be used below to get_item
    # EOF
    def __len__(self):
        return len(self.indices)
    # EOF
    def __getitem__(self, idx):
        return self.imagefolder[self.indices[idx]] # it indexes the indices e.g. if self.indices = [40, 22, 3, ...], self.imagefolder[self.indices[1]] will yield self.imagefolder[22], i.e. the wanted (and ordered) image 
    # EOF
# EOC
