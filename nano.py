import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)
from hyperparams import *
from data import get_batch,decode
from model import GPT

@torch.no_grad()
def estm_loss(model):
    out={}
    model.eval()
    for spilt in ["train","val"]:
        losses=torch.zeros(eval_itr)
        for k in range(eval_itr):
            X,Y=get_batch(spilt)
            logits,loss=model(X,Y)
            losses[k]=loss.item()
        out[spilt]=losses.mean()
    model.train()
    return out



m=GPT()
optimiser=torch.optim.AdamW(m.parameters(),lr=learn_rate)
m.to(device)
for step in range(max_iter):
    if step%eval_interval==0 or step==max_iter-1:
        losses=estm_loss(m)
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb,yb=get_batch('train')
    logits,loss=m(xb,yb)
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    optimiser.step()

context=torch.zeros((1,1),dtype=torch.long,device=device)
print(decode(m.genrate(context,max_new_tokens=500)[0].tolist()))