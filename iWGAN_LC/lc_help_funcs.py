import torch
from constants import *

#   Constants and placeholders
label_placeholder = None
y = None
to_one_hot_holder = None


def concat_input_labels(input, labels, n_classes):
    """

    :param input: the input vector
    :param labels: the labels
    :param n_classes: number of classes
    :return: a joint vector that holds the input and labels
    """
    labels = labels.view(-1, n_classes, 1, 1)
    input_shape = input.shape
    labels_shape = labels.shape
    ones = torch.ones([input_shape[0], labels_shape[1], input_shape[2], input_shape[3]])
    ones = ones.to(device)
    labels = labels*ones
    input = torch.cat((input, labels), 1)
    return input


def create_random_labels(batch_size, num_classes):
    """

    :param batch_size: the size of the batch
    :param num_classes: the number of classes
    :return: a vector with random batch_size labels
    """
    global y, label_placeholder
    if y is None:
        y = torch.LongTensor(batch_size, 1).to(device)
    y = y.random_() % num_classes
    if label_placeholder is None:
        label_placeholder = torch.FloatTensor(batch_size, num_classes).to(device)
    label_placeholder.zero_()
    label_placeholder.scatter_(1, y, 1)
    return label_placeholder


def to_one_hot(labels, num_classes):
    """
    :param labels: labels' vector
    :param num_classes: number of classes
    :return: a one hot vector that represent the labels
    """
    global to_one_hot_holder
    labels = labels.long()
    labels = labels.view(len(labels),1)
    if to_one_hot_holder is None:
        to_one_hot_holder = torch.FloatTensor(len(labels), num_classes).to(device) # we assume len(lables) == batchsize always
    to_one_hot_holder.zero_()
    to_one_hot_holder.scatter_(1, labels, 1)
    return to_one_hot_holder


def gen_rand_noise():
    """
    :return: a generated random noise at size RANDOM_NOISE_SIZE * BATCH_SIZE
    """
    noise = torch.randn(BATCH_SIZE, RANDOM_NOISE_SIZE)
    noise = noise.to(device)

    return noise