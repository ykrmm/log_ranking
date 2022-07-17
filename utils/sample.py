import random
import torch

def construct_batch(user_dict,batch_interactions,batch_features,l_items,n_negs,device):
    batch = {}
    batch['users'] = batch_interactions[0]
    batch['pos_items'] = batch_interactions[1]  
    batch['edge_features'] = batch_features

    def sampling(users,user_dict,l_items,n):
        neg_items = []

        for user in users:
            user = int(user)
            negitems = []
            for i in range(n):
                while True:
                    neg_item = random.choice(l_items)
                    if neg_item not in user_dict[user]:
                        break

                negitems.append(neg_item)
            neg_items.append(neg_item)
        return neg_items

    batch['neg_items'] = torch.LongTensor(sampling(batch['users'],user_dict,l_items,n=n_negs)).to(device).squeeze()

    return batch