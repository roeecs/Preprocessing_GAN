import torch.utils.data as data
import h5py


class H5Dataset(data.Dataset):
    """
        this class helps using the unlabled dataset
    """
    def __init__(self, file_path, imgs_dataset_name, transform=lambda x: x):
        super(H5Dataset, self).__init__()
        self.imgs_dataset_name = imgs_dataset_name
        self.h5_file = h5py.File(file_path)
        self.data = self.h5_file.get(self.imgs_dataset_name)
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.data[index, :, :, :].transpose(1, 2, 0)) if type(index) == int else [
            self.transform(i) for i in self.data[index, :, :, :].transpose(1, 2, 0)]

    def __len__(self):
        return self.data.shape[0]

    def get_imgs_dataset(self):
        return self.h5_file[self.imgs_dataset_name]