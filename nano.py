import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

batch_size=64   ##how many indpendet smaples to run in parallel
block_size=256   #max context len of predic
max_iter=5000
eval_interval=500
learn_rate=3e-4
device="cuda"
eval_itr=200
n_embd=384 
n_layer=6
n_head=6
dropout=0.2

with open("/home/mayukjain/Projects/ml/dl/nanogpt/input.txt", "r",encoding="utf-8") as f:
    text = f.read()

chars=sorted(list(set(text)))
vocab_size=len(chars)
stoi={ch:i for i,ch in enumerate(chars)}
itos={i:ch for i,ch in enumerate(stoi)}
encode= lambda s: [stoi[c] for c in s]
decode= lambda s: "".join([itos[c] for c in s])

data=torch.tensor(encode(text))
n=int(0.9*len(data))
train_data=data[:n]
val_data=data[n:]

def get_batch(split):
    data=train_data if split=="train" else val_data
    ix=torch.randint(len(data)-block_size,(batch_size,))
    x=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+block_size+1] for i in ix])
    x=x.to(device)
    y=y.to(device)
    return x,y

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

class bigram(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,n_embd)
        self.postion_embedding_table=nn.Embedding(block_size,n_embd)
        self.lm_head=nn.Linear(n_embd,vocab_size)
        self.blocks=nn.Sequential(*[block(n_embd,n_head=n_head) for _ in range(n_layer)])
        self.ln=nn.LayerNorm(n_embd)
    def forward(self,idx,target=None):
        B,T=idx.shape
        token_embd=self.token_embedding_table(idx)  ##(B,T,C)
        pos_emb=self.postion_embedding_table(torch.arange(T,device=device))  ##(T,C)
        x=token_embd+pos_emb  ##(B,T,C)
        x=self.blocks(x)
        x=self.ln(x)
        logits=self.lm_head(x)
        
        if target is None:
            loss=None
        else:
            b,t,c=logits.shape
            logits=logits.view(b*t,c)
            target=target.view(b*t)
            loss=F.cross_entropy(logits,target)
        return logits,loss
    def genrate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            idx_comp=idx[:,-block_size:]   ##as embedding table can only take max of last block_size elem at a time
            logits,loss=self(idx_comp)
            logits=logits[:,-1,:]
            probs=F.softmax(logits,dim=-1)
            idx_next=torch.multinomial(probs,num_samples=1)  ##(b,1)
            idx=torch.cat((idx,idx_next),dim=1)  ##(b,t+1)
        return idx

class head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key=nn.Linear(n_embd,head_size,bias=False)
        self.query=nn.Linear(n_embd,head_size,bias=False)
        self.value=nn.Linear(n_embd,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        B,T,C=x.shape
        q=self.query(x)
        k=self.key(x)
        wei=q @ k.transpose(-2,-1)*C**-0.5
        wei=wei.masked_fill(self.tril[:T,:T]==0,float("-inf"))
        wei=torch.softmax(wei,dim=-1)
        wei=self.dropout(wei)
        v=self.value(x)
        out=wei @ v
        return out

class multi_head_attention(nn.Module):
    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads=nn.ModuleList(head(head_size) for _ in range(num_heads))
        self.proj=nn.Linear(n_embd,n_embd)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        out= torch.cat([h(x) for h in self.heads],dim=-1)   #this project layer added after residual connection
        return self.dropout(self.proj(out))

class feedfoward(nn.Module):
    def __init__(self,n_embd):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_embd,4*n_embd),  ##4x imprves accuracy
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd),  ##projection layer added after resiudal connection  
            nn.Dropout(dropout),
                    )
    def forward(self,x):
        return self.net(x)

class block(nn.Module):
    def __init__(self,n_embd,n_head):
        super().__init__()
        head_size=n_embd//n_head
        self.sa=multi_head_attention(n_head,head_size)
        self.ffwd=feedfoward(n_embd)
        self.ln1=nn.LayerNorm(n_embd)   ##similar to batch norm but for NLP, it normalizes acroess the same batch to the embedings
        self.ln2=nn.LayerNorm(n_embd)
    def forward(self,x):
        x=x+self.sa(self.ln1(x))  ##x is a residual line for input direct to post attn and post compuation
        x=x+self.ffwd(self.ln2(x))
        return x

m=bigram()
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