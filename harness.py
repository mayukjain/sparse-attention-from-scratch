import torch
from masks import *
from hyperparams import *



def test_sliding_causality():
    for bs in [8, 16, 32]:
        for w in [2, 4, 8]:
            mask=sliding_window_mask(bs,w)
            upper_triangle_sum = mask.triu(diagonal=1).sum().item()
            assert upper_triangle_sum == 0, f"Failed for block_size={bs}, window_size={w}"


def test_sliding_band_structure():
    mask = sliding_window_mask(8, window_size=3)

    assert mask[0, 0] == 1
    assert mask[2, 0] == 1
    assert mask[3, 1] == 1
    assert mask[7, 5] == 1
    
   
    assert mask[3, 0] == 0
    assert mask[4, 1] == 0 
    assert mask[7, 4] == 0
    assert mask[0, 1] == 0
    assert mask[3, 4] == 0
    
    print("PASS: test_sliding_band_structure")


def test_sliding_full_window_equals_dense():
    window_size =torch.randint(1, 8, (1,)).item()
    block_size=window_size
    mask = sliding_window_mask(block_size, window_size)
    dense_mask = causal_mask(block_size)
    assert torch.equal(mask, dense_mask)
    print("PASS: test_sliding_full_window_equals_dense")
    


def test_sliding_diagonal_always_included():
    window_size=torch.randint(1, 8, (1,)).item()
    block_size=torch.randint(window_size, 16, (1,)).item()
    mask = sliding_window_mask(block_size, window_size)
    diagonal_sum = torch.diag(mask).sum().item()
    assert diagonal_sum == block_size, f"Diagonal sum {diagonal_sum} != block_size {block_size} for block_size={block_size}, window_size={window_size}"
    print("PASS: test_sliding_diagonal_always_included")
    


def test_bigbird_causality():
    window_size=torch.randint(1, 8, (1,)).item()
    block_size=torch.randint(window_size, 16, (1,)).item()
    mask=bigbird_mask(block_size,window_size,num_global=num_global,num_random=num_random)
    diag_sum=mask.triu(diagonal=1).sum().item()
    assert diag_sum==0 ,f"Failed for block_size={block_size}, window_size={window_size}"
    


def test_bigbird_has_global_tokens():
    window_size=torch.randint(1, 8, (1,)).item()
    block_size=torch.randint(window_size, 16, (1,)).item()
    mask=bigbird_mask(block_size,window_size,num_global=num_global,num_random=num_random)
    global_cols = mask[num_global:, :num_global]
    assert global_cols.all(), \
        f"Rows past global don't fully attend to global columns:\n{global_cols}"



def test_bigbird_diagonal_always_included():
    for bs in [16, 32]:
        for w in [4, 8]:
            mask = bigbird_mask(bs, w, num_random=3, num_global=2)
            diag_sum = torch.diag(mask).sum().item()
            assert diag_sum == bs,f"bs={bs}, w={w}: diagonal sum {diag_sum}, expected {bs}"


def test_bigbird_includes_sliding_window():
    bs = torch.randint(1,8,(1,)).item()
    w = torch.randint(1,8,(1,)).item()
    
    sliding = sliding_window_mask(bs, w)
    bigbird = bigbird_mask(bs, w, num_random=3, num_global=2)
    
    
    diff = bigbird - sliding
    assert (diff >= 0).all(), f"BigBird missing some sliding-window positions:\n{diff}"
    

if __name__ == "__main__":

    test_sliding_causality()
    print("PASS: sliding causality")
    test_sliding_band_structure()
    print("PASS: sliding band structure")
    test_sliding_full_window_equals_dense()
    print("PASS: sliding full window equals dense")
    test_sliding_diagonal_always_included()
    print("PASS: sliding diagonal always included")

    test_bigbird_causality()
    print("PASS: bigbird causality")
    test_bigbird_has_global_tokens()
    print("PASS: bigbird global tokens")
    test_bigbird_diagonal_always_included()
    print("PASS: bigbird diagonal always included")
    test_bigbird_includes_sliding_window()
    print("PASS: bigbird includes sliding")
    
    print("PASS: bigbird at least as dense as sliding")
    
    print("\nAll tests passed.")