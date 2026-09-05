import torch
torch.manual_seed(1337)
from hyperparams import *
from data import get_batch,decode
from model import GPT
import os

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

def train():
    m = GPT()
    optimiser = torch.optim.AdamW(m.parameters(), lr=learn_rate)
    m.to(device)
    for step in range(max_iter):
        if step % eval_interval == 0 or step == max_iter - 1:
            losses = estm_loss(m)
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        xb, yb = get_batch('train')
        logits, loss = m(xb, yb)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(m.state_dict(), f'checkpoints/model_{mask_type}.pt')
    print(f"Saved checkpoint to checkpoints/model_{mask_type}.pt")
    return m


def generate(m=None, max_new_tokens=500):
    if m is None:
        m = GPT()
        m.to(device)
        m.load_state_dict(torch.load(f'checkpoints/model_{mask_type}.pt'))
    m.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    with torch.no_grad():
        generated = m.genrate(context, max_new_tokens=max_new_tokens)
    print(f"--- {mask_type} generation ---")
    print(decode(generated[0].tolist()))


if __name__ == "__main__":
    m = train()
    generate(m)

