import torch


DATA_PATH = '/content/drive/My Drive/SadnaGAN/copyDebug.hdf5'
VAL_PATH = '/content/drive/My Drive/SadnaGAN/copyDebug.hdf5'
RESTORE_MODE = False    # if True, it will load saved model from OUT_PATH and continue to train
START_ITER = 0  # starting iteration
OUTPUT_PATH = '/content/drive/My Drive/SadnaGAN/results64/'  # output path where result (.e.g drawing images, cost, chart) will be stored
DIM = 64    # Model dimensionality
CRITIC_ITERS = 5    # How many iterations to train the critic for
GENER_ITERS = 1
N_GPUS = 1  # Number of GPUs
BATCH_SIZE = 64     # Batch size. Must be a multiple of N_GPUS
END_ITER = 100000    # How many iterations to train for
LAMBDA = 10     # Gradient penalty lambda hyperparameter
OUTPUT_DIM = 64*64*3    # Number of pixels in each image
LR = 1e-4   # learning rate
NUM_CLASSES = 64  # num of image's classes
D_DIM = 128
G_DIM = 128
RANDOM_NOISE_SIZE = 128
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")