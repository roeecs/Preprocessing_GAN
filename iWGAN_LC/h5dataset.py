import torch.utils.data as data
import h5py


class H5Dataset(data.Dataset):

    def __init__(self, file_path, imgs_dataset_name='imgs', labels_dataset_name='labels', transform=None):
        super(H5Dataset, self).__init__()
        h5_file = h5py.File(file_path)
        self.data = h5_file.get(imgs_dataset_name)
        self.transform = transform
        self.labels = h5_file.get(labels_dataset_name)

    def __getitem__(self, index):
        if self.transform:
            images = self.transform(self.data[index, :, :, :].transpose(1, 2, 0)) if type(index) == int else [
                self.transform(i) for i in self.data[index, :, :, :].transpose(1, 2, 0)]
            labels = self.labels[index] if type(index) == int else [i for i in self.labels[index]]
            return images, labels

    def __len__(self):
        return self.labels.shape[0]
