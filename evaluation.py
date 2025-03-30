import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def split_context_target(token_ids: torch.Tensor, context_ratio: float = 0.8):
    """
    Splits a 1D tensor of token IDs into context and target parts.
    
    Args:
        token_ids (torch.Tensor): A 1D tensor of token IDs.
        context_ratio (float): Fraction of tokens to use as context.
    
    Returns:
        (context_ids, target_ids) (torch.Tensor, torch.Tensor)
    """
    total_length = len(token_ids)
    context_length = int(total_length * context_ratio)
    context_ids = token_ids[:context_length]
    target_ids = token_ids[context_length:]
    return context_ids, target_ids

def decode_tokens_to_numbers(text: str):
    """
    Decodes a LLMTIME-formatted string into a list of numeric values.
    Example format: "0.25,1.50;0.27,1.47;0.31,1.42"
    
    We split by semicolon to separate timesteps, then by comma for variables,
    and parse each as a float.
    
    Args:
        text (str): The decoded text from the model's output.
    
    Returns:
        List[float]: A flat list of numeric values (prey, predator, prey, predator, ...).
    """
    numbers = []
    timesteps = text.split(";")
    for step in timesteps:
        # Split each timestep by commas
        parts = step.split(",")
        for p in parts:
            try:
                # Convert the string to float if possible
                val = float(p.strip())
                numbers.append(val)
            except ValueError:
                # If conversion fails (e.g., empty string), skip
                continue
    return numbers

###############################################################################
# Main Evaluation Function
###############################################################################

def evaluation(model, tokenizer, tokenized_data, context_ratio: float = 0.7):
    """
    Evaluates the model in a fully autoregressive manner using model.generate.
    
    For each sequence:
      1. Split the sequence into context and target using split_context_target.
      2. Use model.generate (with output_scores=True and return_dict_in_generate=True)
         to generate all target tokens at once.
      3. Extract the per-token logits (scores) for each generated token and compute the 
         cross-entropy loss against the ground truth token.
      4. Compute the average loss over the generated tokens and log the loss curve.
      5. Decode the generated tokens and the ground truth target tokens into numeric values,
         and compute the Mean Squared Error (MSE) for forecast evaluation.
    
    Args:
        model: The Qwen2.5-Instruct model.
        tokenizer: The corresponding tokenizer.
        tokenized_data (List[torch.Tensor]): List of 1D token ID tensors.
        context_ratio (float): Fraction of tokens used as context.
    
    Returns:
        Tuple[float, float]: The average cross-entropy loss and MSE over evaluated sequences.
    """
   
    num_eval = 10  # Evaluate first 10 sequences
    all_seq_losses = []
    all_seq_mses = []
    
    for i in range(num_eval):
        # Retrieve sequence and split into context and target
        seq = tokenized_data[i].to(device)
        context_ids, target_ids = split_context_target(seq, context_ratio)
        target_ids = target_ids[:100]
        input_ids = context_ids.unsqueeze(0)  # Shape: (1, context_length)
        
        # Generate tokens autoregressively using model.generate with scores output
        with torch.no_grad():
            gen_output = model.generate(
                input_ids,
                max_new_tokens=len(target_ids),
                do_sample=False,  # Greedy decoding
                output_scores=True,
                return_dict_in_generate=True
            )
        
        # gen_output.sequences contains context + generated tokens.
        generated_ids = gen_output.sequences[0]
        # gen_output.scores is a tuple of logits for each generated token, each with shape (batch_size, vocab_size)
        scores = gen_output.scores
        
        # Compute per-token loss using the returned scores and corresponding ground truth token
        token_losses = []
        for j, score in enumerate(scores):
            # Ground truth token for step j is target_ids[j]
            gt_token = target_ids[j].unsqueeze(0)  # Shape: (1,)
            loss_j = torch.nn.functional.cross_entropy(score, gt_token)
            token_losses.append(loss_j.item())
        avg_loss_seq = np.mean(token_losses)
        all_seq_losses.append(avg_loss_seq)
        
        # For forecast evaluation, compare generated tokens (excluding context) to ground truth target tokens
        generated_target_ids = generated_ids[len(context_ids):]
        pred_text = tokenizer.decode(generated_target_ids, skip_special_tokens=True)
        true_text = tokenizer.decode(target_ids, skip_special_tokens=True)
        pred_numbers = decode_tokens_to_numbers(pred_text)
        true_numbers = decode_tokens_to_numbers(true_text)
        mse = float("inf")
        if len(pred_numbers) == len(true_numbers) and len(pred_numbers) > 0:
            mse = np.mean((np.array(pred_numbers) - np.array(true_numbers)) ** 2)
        all_seq_mses.append(mse)
        
        # plot and log the loss curve
       
        plt.figure()
        plt.plot(token_losses, label="Token Loss")
        plt.title(f"Loss Curve for Sequence {i}")
        plt.xlabel("Prediction Step")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        wandb.log({f"loss_curve_seq_{i}": wandb.Image(plt)})
        plt.close()
        
        # plot and log forecast comparison (requires numeric reshaping, e.g., into (-1, 2))
        try:
            pred_array = np.array(pred_numbers).reshape(-1, 2)
            true_array = np.array(true_numbers).reshape(-1, 2)
            plt.figure()
            plt.plot(true_array[:, 0], label="True Prey")
            plt.plot(true_array[:, 1], label="True Predator", linestyle="--")
            plt.plot(pred_array[:, 0], label="Predicted Prey")
            plt.plot(pred_array[:, 1], label="Predicted Predator", linestyle="--")
            plt.title(f"Forecast Comparison for Sequence {i}")
            plt.legend()
            wandb.log({f"forecast_seq_{i}": wandb.Image(plt)})
            plt.close()
        except Exception as e:
            print(f"Sequence {i}: Error in plotting forecast: {e}")
        
        wandb.log({
            "loss_per_sequence": avg_loss_seq,
            "mse_per_sequence": mse,
        }, step=i+1)
    
    avg_loss_overall = np.mean(all_seq_losses) if all_seq_losses else float("inf")
    avg_mse_overall = np.mean(all_seq_mses) if all_seq_mses else float("inf")
    wandb.log({
        "avg_loss": avg_loss_overall,
        "avg_mse": avg_mse_overall,
    })
    return avg_loss_overall, avg_mse_overall