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
window_size=128
mask_type = 'bigbird' 
num_global=2
num_random=3


## testing ke liya to chnage
'''
batch_size = 16
block_size = 32
max_iter = 200
eval_interval = 100
learn_rate = 3e-4
device = "cuda"
eval_itr = 20
n_embd = 64
n_layer = 2
n_head = 2
dropout = 0.0

window_size = 8
mask_type = 'bigbird'
num_global = 2
num_random = 3'''