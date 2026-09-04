from model import head as Head
import hyperparams
from hyperparams import n_embd, n_head
from masks import sliding_window_mask, bigbird_mask
import torch
import time
import matplotlib.pyplot as plt

seq_lens = [512, 1024, 2048, 4096, 8192]
variants = ['dense', 'sliding', 'bigbird']

head_size = n_embd // n_head
window_size = hyperparams.window_size
num_global = hyperparams.num_global
num_random = hyperparams.num_random

def make_mask(mask_type, seq_len):
    if mask_type == 'dense':
        mask = torch.tril(torch.ones(seq_len, seq_len))
    elif mask_type == 'sliding':
        mask = sliding_window_mask(seq_len, window_size)
    elif mask_type == 'bigbird':
        mask = bigbird_mask(seq_len, window_size, num_random, num_global)
    return mask.cuda()

def tester(head_instance, x):
    for _ in range(3):
        _ = head_instance(x)
    torch.cuda.synchronize()
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        _ = head_instance(x)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    avg_time_ms = ((end - start) / 10) * 1000
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return avg_time_ms, peak_mem_mb

results = []
for variant in variants:
    for seq_len in seq_lens:
        try:
            
            head_instance = Head(head_size=head_size).cuda()
            
           
            new_mask = make_mask(variant, seq_len)
            head_instance.mask = new_mask  
            
            x = torch.randn(1, seq_len, n_embd, device='cuda')
            time_ms, mem_mb = tester(head_instance, x)
            results.append({'variant': variant, 'seq_len': seq_len, 'time_ms': time_ms, 'mem_mb': mem_mb})
            print(f"{variant:10} seq_len={seq_len:5} time={time_ms:.2f}ms mem={mem_mb:.1f}MB")
        except torch.cuda.OutOfMemoryError:
            print(f"{variant:10} seq_len={seq_len:5} OOM — skipping")
            torch.cuda.empty_cache()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
line_styles = {'dense': '-', 'sliding': '--', 'bigbird': ':'}
for variant in variants:
    data = [r for r in results if r['variant'] == variant]
    if not data:
        continue
    xs = [r['seq_len'] for r in data]
    ts = [r['time_ms'] for r in data]
    ms = [r['mem_mb'] for r in data]
    ax1.plot(xs, ts, marker='o', label=variant, linestyle=line_styles[variant], alpha=0.7, linewidth=2)
    ax2.plot(xs, ms, marker='o', label=variant, linestyle=line_styles[variant], alpha=0.7, linewidth=2)

for ax, ylabel in [(ax1, 'Time (ms)'), (ax2, 'Peak Memory (MB)')]:
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('benchmark_plot.png', dpi=150)
plt.show()