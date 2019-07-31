from cluster_funcs import get_features_generator, write_dataset_to_hdf5
import sys
import torch
import torchvision.models as models
from sklearn.decomposition import PCA, IncrementalPCA
import torchvision.transforms as transforms
from h5dataset import H5Dataset
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

# CONSTS
VGG_MIN_IMG_SIZE = 224
BATCH_SIZE = 1
CHUNK_SIZE = 1000

# device
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")


def cluster(data_file_path, imgs_dataset_name, output_path, biggan_bool):
    # get the vgg model
    vgg = models.vgg11(pretrained=True).to(device)
    vgg.train(False)
    vgg.eval()

    # get the incremental pca model
    ipca = IncrementalPCA(n_components=10, batch_size=16)

    # define the data transformation
    data_transform = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize(VGG_MIN_IMG_SIZE),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                         ])

    # geting dataset from hdf5 file
    images = H5Dataset(data_file_path, imgs_dataset_name, transform=data_transform)
    dataset_loader = torch.utils.data.DataLoader(images, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)
    data_size = len(images)

    # training the iPCA model
    a_gen = get_features_generator(dataset_loader, vgg)
    print("Extracting features")
    for _ in tqdm(range(data_size // CHUNK_SIZE + 1)):
        a = next(a_gen)
        with torch.no_grad():
            labels = torch.Tensor(CHUNK_SIZE, 25088)
            torch.cat(a, out=labels)
        labels = labels.numpy()[:, :]
        ipca.partial_fit(labels)

    # iPCA transforms
    y = []
    a_gen = get_features_generator(dataset_loader, vgg)
    print("Reducing dimensions")
    for _ in tqdm(range(data_size // CHUNK_SIZE + 1)):
        a = next(a_gen)
        with torch.no_grad():
            labels = torch.Tensor(CHUNK_SIZE, 25088)
            torch.cat(a, out=labels)
        labels = labels.numpy()[:, :]
        y.append(ipca.transform(labels))

    # concatenate features to one array
    features_after_pca = y[0]
    for i in range(1, data_size // CHUNK_SIZE + 1):
        features_after_pca = np.concatenate((features_after_pca, y[i]))

    # k means
    print("Running K-Means")
    kmeans = KMeans(n_clusters=64, random_state=0).fit(features_after_pca)
    labels = kmeans.labels_

    # write data and labels to new file
    write_dataset_to_hdf5(images.get_imgs_dataset(), output_path, 'imgs')
    write_dataset_to_hdf5(labels, output_path, 'labels', biggan_bool)


def main(argv):
    if len(argv) < 4:
        print("not enough arguments!")
        return None
    biggan = False
    if len(argv) == 5:
        if argv[4] == "biggan":
            biggan = True
    cluster(argv[1], argv[2], argv[3], biggan)


if __name__ == '__main__':
    main(sys.argv)
