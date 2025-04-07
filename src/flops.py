
"""
FLOPS Estimation Framework for Qwen2.5-Instruct Model

This script provides functions to calculate the FLOPS (floating point operations)
for different components of the Qwen2.5-Instruct model.

The overall structure includes:
    1. Basic modules (embedding, multi-head attention, feed-forward network, normalization)
    2. A single Transformer block that combines these modules.
    3. The overall Qwen model (embedding + multiple Transformer blocks + output layer).
    4. Total FLOPS for an experiment (training or inference), considering batch size, sequence length,
       number of steps, and that backward pass FLOPS = 2 × forward pass FLOPS.
"""

def flops_embedding(batch_size: int, seq_len: int, hidden_dim: int) -> int:
    """
    Calculate FLOPS for the embedding layer.
    
    For simplicity, assume:
    - Each token is embedded into a vector of size hidden_dim.
      - Adding positional embeddings: one addition per element.
    
    Args:
        batch_size: Number of samples in the batch.
        seq_len: Length of each sequence.
        hidden_dim: Dimension of the embeddings.
    
    Returns:
        Estimated FLOPS for the embedding operation.
    """
    # Each element: one addition (for positional embedding)
    return batch_size * seq_len * hidden_dim

def flops_attention(batch_size: int, seq_len: int, hidden_dim: int, num_heads: int, r: int = 0) -> int:
    """
    Estimate FLOPS for a multi-head attention layer.
    
    This should include:
      - Q, K, V projections (matrix multiplications).
      - Scaled dot-product attention computation (including Q*K^T, softmax, and multiplication with V).
      - Output projection.
    
    Args:
        batch_size: Number of samples in the batch.
        seq_len: Sequence length.
        hidden_dim: Hidden dimension of the model.
        num_heads: Number of attention heads.
    
    Returns:
        Estimated FLOPS for the attention layer.
    """
    # Q, K, V projections: three matrix multiplications.
    # For each: (seq_len x hidden_dim) x (hidden_dim x hidden_dim)
    flops_qkv = 3 * batch_size * seq_len * hidden_dim * (2 * hidden_dim - 1)

    # Consider LoRA
    # If LoRA is applied to Q and V projections, add extra FLOPs
    flops_lora = 0
    if r > 0:
        # Each LoRA: A (d x r) + B (r x d) matrix multiply per token => FLOPs = 2 * d * r per projection
        # Applied to Q and V => 2 * (2 * d * r) * tokens
        flops_lora = 2 * 2 * batch_size * seq_len * hidden_dim * r  # 2 for Q and V

    flops_qkv += flops_lora  # include LoRA

    
    # Dimension per head
    d_k = hidden_dim / num_heads  
    
    # Attention score computation: for each head, compute Q*K^T.
    # For one head: (seq_len x d_k) x (d_k x seq_len) -> (seq_len x seq_len)
    # FLOPS per head: seq_len * seq_len * (2*d_k - 1)
    flops_attn_score = batch_size * num_heads * seq_len * seq_len * (2 * d_k - 1)
    
    # Softmax over the attention scores:
    flops_softmax = batch_size * num_heads * seq_len * seq_len * 12
    
    # Multiplication with V: for each head, multiply attention matrix (seq_len x seq_len)
    # with V (seq_len x d_k) to get (seq_len x d_k)
    # FLOPS per head: seq_len * d_k * (2*seq_len - 1)
    flops_attn_v = batch_size * num_heads * seq_len * d_k * (2 * seq_len - 1)
    
    # Output projection: maps the concatenated heads (shape: batch_size x seq_len x hidden_dim)
    # through a linear layer: (seq_len x hidden_dim) x (hidden_dim x hidden_dim)
    flops_out_proj = batch_size * seq_len * hidden_dim * (2 * hidden_dim - 1)
    
    total_flops = int(flops_qkv + flops_attn_score + flops_softmax + flops_attn_v + flops_out_proj)
    return total_flops
   

def flops_feedforward(batch_size: int, seq_len: int, hidden_dim: int, ffn_ratio: float = 4.0) -> int:
    """
    Estimate FLOPS for the feed-forward network (MLP) in a Transformer.
    
    Typically, the FFN consists of two linear layers and an activation function.
    Assume:
      - First linear layer: from hidden_dim to intermediate_dim (where intermediate_dim = ffn_ratio * hidden_dim).
      - Activation : SwigLu.
      - Second linear layer: from intermediate_dim back to hidden_dim.
    
    Args:
        batch_size: Number of samples in the batch.
        seq_len: Sequence length.
        hidden_dim: Hidden dimension of the model.
        ffn_ratio: Multiplier to determine the intermediate dimension.
    
    Returns:
        Estimated FLOPS for the FFN.
    """
    intermediate_dim = int(hidden_dim * ffn_ratio)
    
    # For each token: (2*hidden_dim - 1) * intermediate_dim
    flops_linear1 = batch_size * seq_len * ((2 * hidden_dim - 1) * intermediate_dim)
    # Activation FLOPS:
    flops_activation = batch_size * seq_len * intermediate_dim * 14
    # Second linear layer FLOPS:
    flops_linear2 = batch_size * seq_len * ((2 * intermediate_dim - 1) * hidden_dim)
    return flops_linear1 + flops_activation + flops_linear2

def flops_norm(batch_size: int, seq_len: int, hidden_dim: int) -> int:
    """
    Estimate FLOPS for a normalization layer (RMSNorm 
    
  
    
    Args:
        batch_size: Number of samples in the batch.
        seq_len: Sequence length.
        hidden_dim: Hidden dimension.
    
    Returns:
        Estimated FLOPS for the normalization layer.
    """
    # Per-token FLOPS for RMSNorm (no bias):
    flops_per_token = (4 * hidden_dim) + 11

    # Total tokens = batch_size * seq_len
    total_flops = (batch_size * seq_len) * flops_per_token
    return total_flops

def flops_transformer_block(batch_size: int, seq_len: int, hidden_dim: int, num_heads: int, ffn_ratio: float = 4.0, r: int = 0) -> int:
    """
    Estimate FLOPS for a single Transformer block, which includes:
      - Multi-head attention.
      - Feed-forward network.
      - Two normalization layers.
    
    Args:
        batch_size: Number of samples in the batch.
        seq_len: Sequence length.
        hidden_dim: Hidden dimension of the model.
        num_heads: Number of attention heads.
        ffn_ratio: FFN ratio for intermediate dimension.
    
    Returns:
        Estimated FLOPS for one Transformer block.
    """
    flops_attn = flops_attention(batch_size, seq_len, hidden_dim, num_heads,r)
    flops_ffn = flops_feedforward(batch_size, seq_len, hidden_dim, ffn_ratio)
    # Assume two normalization layers per block
    flops_norm_total = 2 * flops_norm(batch_size, seq_len, hidden_dim)
    flops_residual = 2 * batch_size * seq_len * hidden_dim
    return flops_attn + flops_ffn + flops_norm_total + flops_residual


def flops_cross_entropy_loss(batch_size: int, seq_len: int) -> int:
    
    vocab_size = 151936
    return int(batch_size * seq_len * 11 * vocab_size)


def flops_qwen_model(batch_size: int, seq_len: int, hidden_dim: int, num_layers: int, num_heads: int, ffn_ratio: float, r: int = 0 ) -> int:
    """
    Estimate FLOPS for one forward pass of the entire Qwen2.5-Instruct model.
    This includes:
      - Embedding layer.
      - Multiple Transformer blocks.
      - Output (LM head) layer.
    
    Args:
        batch_size: Number of samples in the batch.
        seq_len: Sequence length.
        hidden_dim: Hidden dimension of the model.
        num_layers: Number of Transformer blocks.
        num_heads: Number of attention heads.
        ffn_ratio: FFN ratio for the feed-forward network.
    
    Returns:
        Estimated FLOPS for one forward pass.
    """
    flops_emb = flops_embedding(batch_size, seq_len, hidden_dim)
    flops_blocks = 0
    for _ in range(num_layers):
        flops_blocks += flops_transformer_block(batch_size, seq_len, hidden_dim, num_heads, ffn_ratio, r)
    # Output layer (LM head): a linear mapping from hidden_dim to vocab_size.
    vocab_size = 151936
    flops_output = batch_size * seq_len * ((2 * hidden_dim - 1) * vocab_size)
    flops_loss = flops_cross_entropy_loss(batch_size, seq_len)
    return flops_emb + flops_blocks + flops_output + flops_loss

def flops_for_experiment(num_steps: int, batch_size: int, seq_len: int, hidden_dim: int, num_layers: int, num_heads: int, ffn_ratio: float, r: int ,training: bool = True) -> int:
    """
    Calculate the total estimated FLOPS for an experiment.
    
    For training:
      Total FLOPS = num_steps * (forward FLOPS + backward FLOPS)
                  = num_steps * (forward FLOPS * 3)  (assuming backward pass costs 2x forward)
    For inference:
      Total FLOPS = num_steps * forward FLOPS
    
    Args:
        num_steps: Number of training or inference steps.
        batch_size: Batch size.
        seq_len: Sequence length.
        hidden_dim: Hidden dimension.
        num_layers: Number of Transformer blocks.
        num_heads: Number of attention heads.
        ffn_ratio: FFN ratio.
        training: Whether to calculate FLOPS for training (includes backward pass).
    
    Returns:
        Total estimated FLOPS for the experiment.
    """
    forward_flops = flops_qwen_model(batch_size, seq_len, hidden_dim, num_layers, num_heads, ffn_ratio,r)
    if training:
        total_per_step = forward_flops * 3  # forward + 2×backward
    else:
        total_per_step = forward_flops
    return num_steps * total_per_step

