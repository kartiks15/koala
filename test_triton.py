import torch 
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)
    
@triton.jit
def multiply_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=1.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=1.0)

    output = x * y

    tl.store(output_ptr + offsets, output, mask=mask)

def add(x, y):
    output = torch.empty_like(x)
    
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024) 
    return output

def multiply(x, y):
    output = torch.empty_like(x)
    
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    
    multiply_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024) 
    return output
if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    size = 3
    x = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
    y = torch.tensor([4.0, 5.0, 6.0], device=DEVICE)
    output_torch = x @ y
    # Only run Triton multiply if a CUDA device is present and Triton driver is active
    try:
        output_triton = multiply(x, y)
    except Exception as e:
        output_triton = None
        print('Triton multiply failed to run:', e)

    print(output_torch)
    print(output_triton)
    if output_triton is not None:
        print(f'The maximum difference between torch and triton is '
              f'{torch.max(torch.abs(output_torch - output_triton))}')