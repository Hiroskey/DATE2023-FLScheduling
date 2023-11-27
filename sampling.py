#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
import math
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    print(num_items)
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def mnist_prob_sched(dataset, num_users, MAXIMUM_COMPUTATIONAL_CAPAILITIES, COMPUTATIONAL_FLUCTUATIONS, LAMBDA, DEADLINE):
    """
    Probabilistic data size scheduling for each client from MNIST dataset
    :param dataset:
    :param num_users:
    Added below
    :param maximum_computational_capability for each client:
    :param computational fluctuation for each client:
    :param lambda for each client (interruptions)
    :param deadline:
    :return:
    """
    def expected_value(d, prob):
        return d*prob

    def probability(T_dead, d, mu, T_cm, a, lambda_val):
        # The result of integra
        term1 = 1 - math.exp(-lambda_val * T_dead)
        term2_numerator = d  * lambda_val * math.exp(-mu * (T_dead - T_cm) / d + mu * a) * (1 - math.exp((-lambda_val + mu/d) * T_dead))
        term2_denominator = lambda_val * d - mu
        term2 = term2_numerator / term2_denominator
        result = term1 - term2
        return result

    prob = [0] * num_users
    exp = [0] * num_users
    data = [0] * num_users
    T_cm = 0
    max_num_items = int(len(dataset)/num_users)
    for c in range(num_users):
        max_exp = 0
        for data_size in range(1, max_num_items):
            prob[c] = probability(DEADLINE, data_size, COMPUTATIONAL_FLUCTUATIONS[c], T_cm, MAXIMUM_COMPUTATIONAL_CAPAILITIES[c], LAMBDA[c])
            exp[c] = expected_value(data_size, prob[c])
            if 0 < prob[c] and prob[c] < 1:
                if max_exp < exp[c]:
                    max_exp < exp[c]
                    data[c] = data_size
    print('number of data samples:', data)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, data[i], replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users, data


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
