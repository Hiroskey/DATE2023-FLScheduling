#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, mnist_prob_sched
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
from utils.scheduling import Scheduler

import signal
import sys
import atexit

from utils.custom_utils import can_complete_task, select_user, apply_preemption
import random

# Global variables
# SEED_VALUE = 2023
# MIN_RATIO = 0.6
MAX_RATIO = 1.2
#DEADLINE = 600 # Deadline Constraint, 10 minutes in seconds

# Calculate dataset size for each user by using total dataset size and number of users
UNIFORM_DATASET_SIZE = None # This will be calculated in the main loop; it defines the size of the dataset for each user in the case of uniform distribution
MAX_COMPUTATION = 0.08
FLUCTUATION_COMPUTATION = 30

# Usecase specific variables
UNIFORM_DATASIZE_DISTRIBUTION = True # If it is True, data is distributed uniformly among users

# Interruption parameters
LAMBDA_I = 0.004 # Avarage rate at which interruptions occur per unit time (per second), 0.004 means 4 interruption every 1000 seconds
MU_K = 300 # Execution rate (interruption duration) in seconds

class Node:
    def __init__(self, node_idx, user_idx, available_time):
        self.node_idx = node_idx
        self.user_idx = user_idx
        self.available_time = available_time
        self.tested_users = []
    def __repr__(self):
        return f"Node(node_idx={self.node_idx}, user_idx={self.user_idx}, available_time={self.available_time})"

##################################################
######## SETUP ###################################
if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    '''
    Hiroki aded begin:
    '''
    # Input parameters for the clients (consider computationally heterogenious clients)
    DEADLINE = 600 # Deadline Constraint, 10 minutes in seconds
    MAXIMUM_COMPUTATIONAL_CAPAILITIES = [random.uniform(0.01, MAX_RATIO) for _ in range(args.num_users)]
    COMPUTATIONAL_FLUCTUATIONS = [random.uniform(1, MAX_COMPUTATION) for _ in range(args.num_users)]
    DATASIZE = [0 for _ in range(args.num_users)] # Initialized
    LAMBDA = [int(random.uniform(6, 60)) for _ in range(args.num_users)] # each duration takes LAMBDA
    '''
    Hiroki added: end
    '''

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            '''
            Hiroki added: begin
            '''
            # dict_users = mnist_iid(dataset_train, args.num_users)
            dict_users, DATASIZE = mnist_prob_sched(dataset_train, args.num_users, MAXIMUM_COMPUTATIONAL_CAPAILITIES, COMPUTATIONAL_FLUCTUATIONS, LAMBDA, DEADLINE)
            '''
            Hiroki added: end
            '''
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # Finding the size of the dataset for each user by using the total dataset size and number of users
    # For comparison ?
    '''
    Hiroki modified: begin
    '''
    # if UNIFORM_DATASIZE_DISTRIBUTION:
    #     UNIFORM_DATASET_SIZE = len(dataset_train) / args.num_users
    #     
    # else:
    #     raise ValueError("Error in dataset size distribution") # This is not implemented yet
    '''
    Hiroki modified: end
    '''

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    acc = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    # scheduler
    k = max(int(args.frac * args.num_users), 1)
    selectedUsers = np.zeros(args.num_users)
    scheduler = Scheduler(args.num_users, dict_users, dataset_train, args.snr, args.sched, args.comp,args.model,args.dataset)

    # This provides to generate a summary of the data when the program is interrupted
    def signal_handler(signum, frame):
        scheduler.logger.generate_summary()
        sys.exit(0)

    # Register the signal handler for interruption (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)

    # Register the method to be called upon normal termination
    atexit.register(scheduler.logger.generate_summary)

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]    

    '''
    Hiroki modified: begin
    '''
    # Computational capacities for each user 
    # maximum_computation_capabilities = [MAX_COMPUTATION for _ in range(args.num_users)]
    # fluction_computation_capabilities = [FLUCTUATION_COMPUTATION for _ in range(args.num_users)]
    '''
    Hiroki modified: end
    '''
    
    # Node generation
    nodes = [Node(idx+1, None, DEADLINE) for idx in range(k)] # k: defined in the scheduling section

    ##################################################
    ##### MAIN LOOP ##################################
    for iter in range(args.epochs):
        print("Concept: " + args.concept)
        print("Round: " + str(iter))
        
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        
        node_to_update = []
        #idx_to_remove = []

        # User selection
        idxs_users, compress_ratio = scheduler.newUsers(k, iter) # Iter is the current round number, used for logging
        selected_idxs = set(idxs_users) 
        selectedUsers[idxs_users] += 1
        available_users = set(range(args.num_users)) - selected_idxs
        
        if args.sched != "BN2":
            print("idxs_users: " + str(idxs_users))
            print("compress_ratio: " + str(compress_ratio))

        # Assign users to nodes
        for node, user_idx in zip(nodes, idxs_users):
            node.user_idx = user_idx
         
        for node in nodes:
            idx = node.user_idx

            # Check if the user can complete task
            # can_complete, node.available_time, latency_info = can_complete_task(fluction_computation_capabilities[idx], 
            #                             maximum_computation_capabilities[idx], 
            #                             node.available_time, UNIFORM_DATASET_SIZE, LAMBDA_I, 
            #                             MU_K)
            can_complete, node.available_time, latency_info = can_complete_task(COMPUTATIONAL_FLUCTUATIONS[idx], 
                                        MAXIMUM_COMPUTATIONAL_CAPAILITIES[idx], 
                                        node.available_time, DATASIZE[idx], LAMBDA[idx], 
                                        MU_K)
            
            if not can_complete:
                node_to_update.append(node)
                #node.user_idx = None
                print(f"[NODE {node.node_idx}] User {idx} FAILED and the remaining time is {node.available_time} seconds.")
        
        # Handle nodes whose users couldn't complete the task
        for node in node_to_update:
            last_unsuccesful_user = node.user_idx
            node.tested_users.append(last_unsuccesful_user)
            node.user_idx = None
            if args.concept == "dynamic":
                while available_users:
                    # if last_unsuccesful_user is in the available users, then remove it
                    if last_unsuccesful_user in available_users:
                        available_users.remove(last_unsuccesful_user)
                    # new_idx = select_user(available_users,
                    #                       node.available_time,
                    #                       fluction_computation_capabilities,
                    #                       maximum_computation_capabilities)
                    new_idx = select_user(available_users,
                                          node.available_time,
                                          COMPUTATIONAL_FLUCTUATIONS,
                                          MAXIMUM_COMPUTATIONAL_CAPAILITIES)
                    

                    if node.available_time <= 0:
                        print(f"[NODE {node.node_idx}] No available time left for User {new_idx} replacement.")
                        break
                    else:
                        # can_complete, node.available_time, latency_info = can_complete_task(fluction_computation_capabilities[new_idx], 
                        #                                             maximum_computation_capabilities[new_idx], 
                        #                                             node.available_time, UNIFORM_DATASET_SIZE, LAMBDA_I, 
                        #                                             MU_K)
                        can_complete, node.available_time, latency_info = can_complete_task(COMPUTATIONAL_FLUCTUATIONS[new_idx], 
                                                                    MAXIMUM_COMPUTATIONAL_CAPAILITIES[new_idx], 
                                                                    node.available_time, DATASIZE[idx], LAMBDA[idx], 
                                                                    MU_K)
                    # !!! Think an idea: can_complete is True but available_time is more than 0

                    # !!! Think an idea: Should I release the users for the next node, if they are not selected?

                    if can_complete:
                        #selected_idxs.add(last_tested_user)
                        node.user_idx = new_idx
                        print(f"User {node.user_idx} was randomly chosen to replace user {idx}.")
                        break
                    else:
                        available_users.remove(new_idx)
                        node.tested_users.append(new_idx)

        for node in nodes:
            # !!! Think an idea: If no node is available, then we need to pass local updates to the next round

            if not node.user_idx is None:
                idx = node.user_idx
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                if args.all_clients:
                    w_locals[idx] = copy.deepcopy(w)
                else:
                    w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))

        # update global weights
        # added w_global to parameters
        if w_locals:
            w_glob = FedAvg(w_locals, w_glob, compress_ratio, k, args.comp, args.sched, args.blocks)

        if w_glob:
            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)

        if net_glob:
            # Added accurency convergence
            acc_avg, _ = test_img(net_glob, dataset_test, args)
            acc.append(acc_avg)

            if loss_locals:
                # print loss
                loss_avg = sum(loss_locals) / len(loss_locals)
                print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
                loss_train.append(loss_avg)

                # plot loss curve
                plt.figure()
                plt.plot(range(len(loss_train)), loss_train)
                plt.ylabel('train_loss')
                plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}_snr{}_comp{}_scheduling{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.snr, args.comp, args.sched))

                # save result
                np.save('./save/loss_fed_{}_{}_{}_C{}_iid{}_snr{}_comp{}_scheduling{}.npy'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.snr, args.comp, args.sched), loss_train)
                np.save('./save/acc_fed_{}_{}_{}_C{}_iid{}_snr{}_comp{}_scheduling{}.npy'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.snr, args.comp, args.sched), acc)
                if args.sched != 'BN2':
                    np.save('./save/ni_fed_{}_{}_{}_C{}_iid{}_snr{}_comp{}_scheduling{}.npy'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.snr, args.comp, args.sched), selectedUsers)

                # testing
                net_glob.eval()
                acc_train, loss_train = test_img(net_glob, dataset_train, args)
                acc_test, loss_test = test_img(net_glob, dataset_test, args)
                print("Training accuracy: {:.2f}".format(acc_train))
                print("Testing accuracy: {:.2f}".format(acc_test))