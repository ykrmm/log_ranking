from __future__ import division
import numpy as np
import random
import sys
import operator
import copy
from collections import defaultdict
import os, re
import argparse
from os.path import join
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import torch_geometric.data as data
from torch_geometric.utils import to_undirected,coalesce
import torch


# STATIC 
# LOAD THE NETWORK
def load_static_network(network,datapath,mode='interaction',min_interaction=5,test_ratio=0.1,val_ratio=0.1,pred_ratio=0.2):
    '''
    This function loads the input network.
    The network should be in the following format:
    One line per interaction/edge.
    Each line should be: user, item, timestamp, state label, array of features.
    Timestamp should be in cardinal format (not in datetime).
    State label should be 1 whenever the user state changes, 0 otherwise. If there are no state labels, use 0 for all interactions.
    Feature list can be as long as desired. It should be atleast 1 dimensional. If there are no features, use 0 for all interactions. 
    '''

    datapath = join(datapath,network+'.csv')

    user_sequence = []
    item_sequence = []
    feature_sequence = []
    user_dict = {}
    print("\n\n**** Loading %s network from file: %s ****" % (network, datapath))
    f = open(datapath,"r")
    f.readline()
    for cnt, l in enumerate(f):
        # FORMAT: user, item, timestamp, state label, feature list 
        ls = l.strip().split(",")
        user_sequence.append(ls[0])
        item_sequence.append(ls[1])
        feature_sequence.append(list(map(float,ls[4:])))
    f.close()

    user_sequence = np.array(user_sequence) 
    item_sequence = np.array(item_sequence)

    print("Formating item sequence")
    nodeid = 0
    item2id = {}
    for cnt, item in enumerate(item_sequence):
        if item not in item2id:
            item2id[item] = nodeid
            nodeid += 1
    num_items = len(item2id)
    item_sequence_id = [item2id[item] for item in item_sequence]

    print("Formating user sequence")
    nodeid = 0
    user2id = {}
    for cnt, user in enumerate(user_sequence):
        if user not in user2id:
            user2id[user] = nodeid
            nodeid += 1
    num_users = len(user2id)
    user_sequence_id = [user2id[user] for user in user_sequence]
    print('Initially',len(user2id),'Users and ',len(item2id),'items')
    # edge_index pytorch geometric 
    print('Construct user dict')
    edge_index = torch.LongTensor([user_sequence_id,item_sequence_id])
    edge_index[1] += len(user2id)
    edge_features = torch.Tensor(feature_sequence) if len(feature_sequence) > 0 else None
    # remove multiple entries 
    edge_index,edge_features = coalesce(edge_index=edge_index,edge_attr=edge_features)
    user_dict = construct_static_user_dict(edge_index,edge_features)

    # Remove users with low interactions: 
    print('Removing users with low interactions')
    user_dict = remove_user_with_few_interactions(user_dict,threshold = min_interaction)
    n_users,n_items, list_users, list_items = count_users_items(user_dict)
    print(n_users,'Users after preprocessing and',n_items,'after preprocessing')
    if mode == 'interaction':
        user_dict_train, user_dict_val, user_dict_test = split_static_interaction(user_dict,val_ratio=val_ratio,test_ratio=test_ratio)
        print("*** Network loading completed ***\n\n")
        return user_dict_train,user_dict_val,user_dict_test,user2id,item2id,list_users,list_items
    elif mode == 'user':
        user_dict_train,user_dict_struct_val,user_dict_pred_val,user_dict_struct_test,user_dict_pred_test = \
            split_static_user(user_dict,val_user_ratio=val_ratio,test_user_ratio=test_ratio,pred_ratio=pred_ratio)
        print("*** Network loading completed ***\n\n")
        return user_dict_train,user_dict_struct_val,user_dict_pred_val,user_dict_struct_test,user_dict_pred_test,user2id,item2id,list_users,list_items

    # To undirected 
    

def remove_user_with_few_interactions(user_dict,threshold=5):
   
    user_to_delete = []
    for u,l in user_dict.items():
        if len(l) < threshold : 
            user_to_delete.append(u)
    for u in user_to_delete:
        del user_dict[u]
    return user_dict

def count_users_items(user_dict):
    n_users = len(user_dict.keys())

    l_items = []
    for l in user_dict.values():
        for i in l: 
            l_items.append(i[0])

    list_items = list(np.unique(l_items))
    list_users = list(user_dict.keys())
    n_items = len(list_items)
    return n_users,n_items,list_users,list_items

def construct_static_user_dict(edge_index,edge_features):
    user_dict = {}

    for i,(user,item) in enumerate(zip(edge_index[0],edge_index[1])):
        try:
            user_dict[int(user)].append((int(item),list(edge_features[i])))
        except:
            user_dict[int(user)] = [(int(item),list(edge_features[i]))]
    return user_dict


def user_dict_to_edge_index(user_dict):
    """
    convert user_dict to edge_index, edge_features for pytorch geometric use
    """
    u_l = []
    i_l = []
    edge_features = []
    for user,interactions in user_dict.items():
        for item in interactions: 
            u_l.append(user)
            i_l.append(item[0])
            edge_features.append(item[1])
    
    u_l = torch.LongTensor(u_l)
    i_l = torch.LongTensor(i_l)
    edge_index = torch.vstack((u_l,i_l))
    edge_features = torch.Tensor(edge_features)
    edge_index,edge_features = to_undirected(edge_index=edge_index,edge_attr=edge_features) 

    return edge_index,edge_features


def split_static_interaction(user_dict,val_ratio=0.1,test_ratio=0.1):
    """
        user_dict : 
        mode 'interaction' or 'user' return a transductive or an inductive split 
        return : user_dict_train, user_dict_val, user_dict_test
    """
    user_dict_train = {}
    user_dict_val = {}
    user_dict_test = {}

    
    for user,interactions in user_dict.items():
        i_train,i_test = train_test_split(interactions,test_size=test_ratio,random_state=57)
        i_train,i_val = train_test_split(i_train,test_size=val_ratio,random_state=57)

        user_dict_train[user] = i_train
        user_dict_val[user] = i_val
        user_dict_test[user] = i_test

    return user_dict_train,user_dict_val,user_dict_test

    

def split_static_user(user_dict,val_user_ratio=0.2,test_user_ratio=0.2,pred_ratio=0.2):
    """
        
    """
    user_dict_train = {}
    user_dict_val = {}
    user_dict_test = {}

    user_dict_struct_val = {}
    user_dict_struct_test = {}
    user_dict_pred_val = {}
    user_dict_pred_test = {}

    user_list = list(user_dict.keys())
    user_train,user_test = train_test_split(user_list,test_size=test_user_ratio,random_state=57) 
    user_train,user_val = train_test_split(user_train,test_size=val_user_ratio,random_state=57)

    for u in user_train:
        user_dict_train[u] = user_dict[u]
    for u in user_val:
        user_dict_val[u] = user_dict[u]
    for u in user_test:
        user_dict_test[u] = user_dict[u]


    for user,interactions in user_dict_val.items():
        i_struct,i_pred = train_test_split(interactions,test_size=pred_ratio,random_state=57)
        user_dict_struct_val[user] = i_struct
        user_dict_pred_val[user] = i_pred

    for user,interactions in user_dict_test.items():
        i_struct,i_pred = train_test_split(interactions,test_size=pred_ratio,random_state=57)
        user_dict_struct_test[user] = i_struct
        user_dict_pred_test[user] = i_pred
        
    
    return user_dict_train,user_dict_struct_val,user_dict_pred_val,user_dict_struct_test,user_dict_pred_test


# DYNAMIC 


def load_dynamic_network(network,datapath, time_scaling=True):
    '''
    This function loads the input network.
    The network should be in the following format:
    One line per interaction/edge.
    Each line should be: user, item, timestamp, state label, array of features.
    Timestamp should be in cardinal format (not in datetime).
    State label should be 1 whenever the user state changes, 0 otherwise. If there are no state labels, use 0 for all interactions.
    Feature list can be as long as desired. It should be atleast 1 dimensional. If there are no features, use 0 for all interactions. 
    '''

    datapath = join(datapath,network+'.csv')

    user_sequence = []
    item_sequence = []
    label_sequence = []
    feature_sequence = []
    timestamp_sequence = []
    start_timestamp = None
    y_true_labels = []
    user_dict = {}
    print("\n\n**** Loading %s network from file: %s ****" % (network, datapath))
    f = open(datapath,"r")
    f.readline()
    for cnt, l in enumerate(f):
        # FORMAT: user, item, timestamp, state label, feature list 
        ls = l.strip().split(",")
        user_sequence.append(ls[0])
        item_sequence.append(ls[1])
        if start_timestamp is None:
            start_timestamp = float(ls[2])
        timestamp_sequence.append(float(ls[2]) - start_timestamp) 
        y_true_labels.append(int(ls[3])) # label = 1 at state change, 0 otherwise
        feature_sequence.append(list(map(float,ls[4:])))
    f.close()

    user_sequence = np.array(user_sequence) 
    item_sequence = np.array(item_sequence)
    timestamp_sequence = np.array(timestamp_sequence)

    print("Formating item sequence")
    nodeid = 0
    item2id = {}
    item_timedifference_sequence = []
    item_current_timestamp = defaultdict(float)
    for cnt, item in enumerate(item_sequence):
        if item not in item2id:
            item2id[item] = nodeid
            nodeid += 1
        timestamp = timestamp_sequence[cnt]
        item_timedifference_sequence.append(timestamp - item_current_timestamp[item])
        item_current_timestamp[item] = timestamp
    num_items = len(item2id)
    item_sequence_id = [item2id[item] for item in item_sequence]

    print("Formating user sequence")
    nodeid = 0
    user2id = {}
    user_timedifference_sequence = []
    user_current_timestamp = defaultdict(float)
    user_previous_itemid_sequence = []
    user_latest_itemid = defaultdict(lambda: num_items)
    for cnt, user in enumerate(user_sequence):
        if user not in user2id:
            user2id[user] = nodeid
            nodeid += 1
        timestamp = timestamp_sequence[cnt]
        user_timedifference_sequence.append(timestamp - user_current_timestamp[user])
        user_current_timestamp[user] = timestamp
        user_previous_itemid_sequence.append(user_latest_itemid[user])
        user_latest_itemid[user] = item2id[item_sequence[cnt]]
    num_users = len(user2id)
    user_sequence_id = [user2id[user] for user in user_sequence]

    if time_scaling:
        print("Scaling timestamps")
        user_timedifference_sequence = scale(np.array(user_timedifference_sequence) + 1)
        item_timedifference_sequence = scale(np.array(item_timedifference_sequence) + 1)

    # edge_index pytorch geometric 

    edge_index = torch.LongTensor([user_sequence_id,item_sequence_id])
    edge_index[1] += len(user2id)
    edge_features = torch.Tensor(feature_sequence) if len(feature_sequence) > 0 else None
    # remove multiple entries 
    edge_index,edge_features = coalesce(edge_index=edge_index,edge_attr=edge_features)
    user_dict = construct_static_user_dict(edge_index,edge_features)

    # To undirected 
    edge_index,edge_features = to_undirected(edge_index=edge_index,edge_attr=edge_features)

    print("*** Network loading completed ***\n\n")
    return (user2id, user_sequence_id, user_timedifference_sequence, user_previous_itemid_sequence,user_dict, \
        item2id, item_sequence_id, item_timedifference_sequence, \
        timestamp_sequence, \
        feature_sequence, \
        y_true_labels,edge_index,edge_features)


def construct_temporal_userdict(user_sequence,item_sequence,time_sequence):
    user_dict = {}