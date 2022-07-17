import torch 
from torch import nn



def initalize_embeddings(n_users,n_items,emb_size=172,init='xu'):
    """
        n_users: # of users in the graphs
        n_intems: # of items in the graphs
        init: 'xn' : xavier normal, 'xu': xavier uniform, 'u' : uniform, 'n': standard normal distribution

    """
    if init == 'xn':
        initializer = nn.init.xavier_normal_
    elif init == 'xu':
        initializer = nn.init.xavier_uniform_
    elif init == 'u':
        initializer = nn.init.uniform_
    elif init == 'n':
        initializer = nn.init.normal_
    else:
        raise NameError(init,"must be between 'xn' 'xu' , 'u' or 'n' ")

    embedding_dict = {
            'user_emb': initializer(torch.empty(n_users,emb_size)),
            'item_emb': initializer(torch.empty(n_items,emb_size))
        }

    return embedding_dict