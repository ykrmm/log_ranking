
import argparse
import torch
import torch_geometric as pyg
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch import embedding
from utils.load_data import load_static_network,user_dict_to_edge_index
from utils.sample import construct_batch
from modules.embeddings import initalize_embeddings
from time import time
# INITIALIZE PARAMETERS
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='Name of the network/dataset')
parser.add_argument('--split', default='inductive',type=str, help='Name of the network/dataset')
parser.add_argument('--model', default="jodie", help='Model name to save output in file')
parser.add_argument('--gpu', default=0, type=int, help='ID of the gpu to run on. If set to -1 (default), the GPU with most free memory will be chosen.')



# TRAINING 
parser.add_argument('--train_proportion', default=0.8, type=float, help='Fraction of interactions (from the beginning) that are used for training.The next 10% are used for validation and the next 10% for testing')
#parser.add_argument('--state_change', default=True, type=bool, help='True if training with state change of users along with interaction prediction. False otherwise. By default, set to True.')
parser.add_argument('--training', default="transductive",type=str, help='Inductive or transductive training')
parser.add_argument('--epochs', default=50, type=int, help='Number of epochs to train the model')
parser.add_argument('--batch_size', default=50, type=int, help='Batch_size')
parser.add_argument('--K', default=1, type=int, help='Number of negative items for a positive interaction.')
# EMBEDDINGS PARAMETERS 
parser.add_argument('--embedding_dim', default=172, type=int, help='Number of dimensions of the dynamic embedding')
parser.add_argument('--init_emb', default='xu', type=str, help='Initialization of the nodes embeddings')
args = parser.parse_args()





# LOAD DATA 

datapath = 'data'

if args.training == 'transductive':
    user_dict_train, user_dict_val, user_dict_test, user2id, item2id,list_uers,list_items\
         = load_static_network(args.dataset,datapath,mode='interaction')

elif args.training == 'inductive':
    user_dict_train,user_dict_struct_train,user_dict_pred_train,user_dict_struct_val,user_dict_pred_val,user_dict_struct_test,user_dict_pred_test,user2id,item2id,list_users,list_items\
         = load_static_network(args.dataset,datapath,mode='user')

n_users = len(user2id)
n_items = len(item2id)

edge_index_train, edge_features_train = user_dict_to_edge_index(user_dict_train)

device = torch.device("cuda:"+str(args.gpu))


# INITALIZE EMBEDDINGS 

embedding_dict = initalize_embeddings(n_users=n_users,n_items=n_items,init=args.init_emb,emb_size=args.embedding_dim)
all_emb = torch.cat((embedding_dict['user_emb'],embedding_dict['item_emb']))

import ipdb; ipdb.set_trace()
# INITIALIZE MODELS 



# CREATE DATALOADER 

graph_train = Data(x=all_emb,edge_index=edge_index_train,edge_attr=edge_features_train)



# TRANSFORMS DATA 


# TRAINING 

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

for ep in args.epoch:
    
    # shuffle edge_index 
    idx = torch.randperm(edge_index_train.shape[1])
    edge_index_train = edge_index_train[:,idx]
    edge_features_train = edge_features_train[:,idx]

    # training log
    s = 0 
    loop_length = len(edge_index_train[0])
    train_s_t = time()

    while s + args.batch_size <= loop_length:
        
        # Construct batch 
        b = torch.arange(s,s+args.batch_size)
        batch_interactions = edge_index_train[:,b]
        batch_features = edge_features_train[:,b]
        batch = construct_batch(user_dict_train,batch_interactions,batch_features,list_items,args.K,device)
        s += args.batch_size 

        # Inference and backward
        batch_loss = model(batch)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    train_e_t = time()

# EVAL 

