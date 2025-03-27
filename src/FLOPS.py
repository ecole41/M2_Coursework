def RoPE(D, N, H):
    flops = 3 * H * (D/H) * N * (2 * (D/H) - 1)
    return flops

def self_attention(D, N, H):
    qvk_flops = 3 * (D/H) * N * (2 * D - 1)
    attention_flops = N**2 * (2 * (D/H) - 1) + 10 + N**2
    softmax_flops = N * (22 * N - 1)
    self_attention_flops = (D/H) * N * (2 * N - 1)
    linear_transform_flops = D * N * (2 * D - 1)
    total_flops = H * (qvk_flops + attention_flops + softmax_flops + self_attention_flops) + linear_transform_flops
    return total_flops

def residual_connection(D, N):
    flops = 5 * N * D + 10 * N
    return flops

def feed_forward(D, N, D_FF):
    flops = N * (6 * D * D_FF + 13 * D_FF)
    return flops

def LoRA(D, N, r):
    flops = 2 * ((D/H) * D * (2 * r - 1) + (D/H) * D)
    return flops

def total_flops_forward(num_layers, D, N, H, D_FF, lora=False, r=4):
    """
    This function computes the total number of FLOPS for a forward pass through 
    the full transformer layers
    num_layers: number of transformer layers
    D: dimension of the input
    N: number of tokens
    H: number of heads
    D_FF: dimension of the feed forward network
    lora: whether to include LoRA in the computation
    r: LoRA rank
    """
    flops = num_layers * (RoPE(D, N, H) + self_attention(D, N, H) + 2 * residual_connection(D, N) + feed_forward(D, N, D_FF))
    if lora == True:
        flops += LoRA(D, N, r) * num_layers
    return flops

def total_flops_training(num_layers, D, N, H, D_FF, lora, r, total_steps):
    """
    This function computes the total number of FLOPS for the entire training run
    num_layers: number of transformer layers
    D: dimension of the input
    N: number of tokens
    H: number of heads
    D_FF: dimension of the feed forward network
    lora: whether to include LoRA in the computation
    r: LoRA rank
    total_steps: total number of training steps
    """
    forward_flops = num_layers * (
        RoPE(D, N, H) + 
        self_attention(D, N, H) + 
        2 * residual_connection(D, N) + 
        feed_forward(D, N, D_FF)
    )
    
    # Add LoRA FLOPS if applicable
    if lora:
        forward_flops += LoRA(D, N, r) * num_layers

    # Compute Backpropagation FLOPS (2x Forward)
    backward_flops = 2 * forward_flops

    # Total FLOPS per training step (Forward + Backward)
    total_flops_per_step = forward_flops + backward_flops

    # Compute total FLOPS for the entire training run
    total_training_flops = total_flops_per_step * total_steps

    # Print step breakdown
    print(f"Forward FLOPS per step: {forward_flops}")
    print(f"Backward FLOPS per step: {backward_flops}")
    print(f"Total FLOPS per step: {total_flops_per_step}")
    print(f"Total training FLOPS: {total_training_flops}")

    return total_training_flops