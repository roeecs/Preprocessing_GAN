import torch
import h5py

# device
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")


def get_features_generator(dataset_loader, vgg):
    """
    :param dataset_loader: the dataset loader we use
    :param vgg: the vgg model we use
    :return: extracted features of 1k images
    """
    f = []
    uter = iter(dataset_loader)
    for i, batch in enumerate(uter):
        with torch.no_grad():
            t = vgg.features(batch.to(device))
            t = vgg.avgpool(t)
            t = t.view(t.size(0), -1)
            f.append(t.cpu())
            if i % 1000 == 999:
                yield f
                f = []
    yield f


def write_dataset_to_hdf5(dataset, file_path, dataset_name, biggan_bool = False):
    """

    :param dataset: the dataset to be written
    :param file_path: path to file to write to
    :param dataset_name
    :param biggan_bool: indicates if the file is gonna be use for biggan or for iWGAN architectures
    :return: None
    """
    with h5py.File(file_path, "a") as f:
        data_loaded = f.get(dataset_name)
        if data_loaded is not None:
            del f[dataset_name]
        if biggan_bool:
            f.create_dataset(dataset_name, data=dataset, dtype='<i8')
        else:
            f.create_dataset(dataset_name, data=dataset)
