
import argparse
from load_data import load_network
# INITIALIZE PARAMETERS
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='Name of the network/dataset')
parser.add_argument('--split', default='inductive',type=str, help='Name of the network/dataset')
parser.add_argument('--model', default="jodie", help='Model name to save output in file')
parser.add_argument('--gpu', default=-1, type=int, help='ID of the gpu to run on. If set to -1 (default), the GPU with most free memory will be chosen.')
parser.add_argument('--epochs', default=50, type=int, help='Number of epochs to train the model')
parser.add_argument('--embedding_dim', default=128, type=int, help='Number of dimensions of the dynamic embedding')
parser.add_argument('--train_proportion', default=0.8, type=float, help='Fraction of interactions (from the beginning) that are used for training.The next 10% are used for validation and the next 10% for testing')
parser.add_argument('--state_change', default=True, type=bool, help='True if training with state change of users along with interaction prediction. False otherwise. By default, set to True.')
args = parser.parse_args()





# LOAD DATA 

datapath = 'data'

(user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
 item2id, item_sequence_id, item_timediffs_sequence, 
 timestamp_sequence, feature_sequence, y_true,edge_index) = load_network(args.dataset,datapath)





