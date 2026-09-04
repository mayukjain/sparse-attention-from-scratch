# masks.py
import torch

def causal_mask(block_size):
    return torch.tril(torch.ones(block_size, block_size))

def sliding_window_mask(block_size,window_size):
    masks=torch.zeros(block_size, block_size)
    for i in range(block_size):
        for j in range(max(0,i+1-window_size),i+1):
            masks[i,j]=1
    return masks

    

def bigbird_mask(block_size,window_size,num_random,num_global):
    masks=torch.zeros(block_size, block_size)
    for i in range(block_size):
        for j in range(max(0,i+1-window_size),i+1):
                masks[i,j]=1
    for i in range(1,block_size):
        cond=min(i,num_random)
        pos=torch.randperm(i)[:cond]
        masks[i,pos]=1
    masks[:num_global, :] = 1
    masks[:, :num_global] = 1
    causal = torch.tril(torch.ones(block_size, block_size))
    masks = masks * causal  ##so that global doesnt see futur tokens and decoder only
    return masks