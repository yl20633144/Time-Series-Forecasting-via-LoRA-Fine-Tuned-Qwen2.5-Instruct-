# Convert the data to LLMTIME format
def convert_to_llmtime(data_array, alpha, decimal_places):
    """
    Convert a dataset of predator-prey time series into LLMTIME format.

    Parameters:
    data_array: NumPy array 
        Containing time series data of shape (N, T, 2),
                  where N is the number of systems, T is the number of time points,
                  and the last dimension represents (prey, predator).

    alpha: float
        Scaling factor computed from the training set.

    decimal_places: int
        Number of decimal places to round the scaled values.

    Returns:
    - A list of LLMTIME-formatted strings, where each string represents a time series.
    """
    text_list = []
    for i in data_array:
        #Each system has a shape of (100, 2), each row corresponds to a time point and the columns are the prey and predator populations
        time_steps = []
        for j in i:
            #Scaling the data for each (prey,predator) value
            
            prey_formatted = format(j[0] / alpha, f".{decimal_places}f")
            predator_formatted = format(j[1] / alpha, f".{decimal_places}f")
            # Join values of a timestep with commas
            step_str = f"{prey_formatted},{predator_formatted}"
            time_steps.append(step_str)
        # Use semicolon ";" to separate different timesteps in the same trajectory
        llmtime_seq = ";".join(time_steps)
        text_list.append(llmtime_seq)
    return text_list

def load_and_preprocess(file_path, decimal_places, max_target_value):
    """
    Load and preprocess the predator-prey dataset from an HDF5 file.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file containing the dataset.
    decimal_places : int
        Number of decimal places to round the scaled values.
    max_target_value : float
        Maximum target value for scaling.

    Returns
    -------
    train_texts : list of str
        Preprocessed training data sequences in LLMTIME format.
    val_texts : list of str
        Preprocessed validation data sequences in LLMTIME format.
    test_texts : list of str
        Preprocessed test data sequences in LLMTIME format.
    """

    with h5py.File(file_path, "r") as f:
        # Access the full dataset
        trajectories = f["trajectories"][:]
        

        # Example from the sheet of accessing a single trajectory (for the first 50 points):
        # system_id = 0  # First system
        # prey = trajectories[system_id, :50, 0]
        # predator = trajectories[system_id, :50, 1]
        # times = time_points[:50]

    total_number = trajectories.shape[0]  # 1000

    # ---Spilt dataset into train, validation, and test sets---
    # Generate a random permutation of indices for all systems
    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(total_number) 

    # Compute split indices for ratio 7 : 1.5 : 1.5 (sums to 10)
    total_ratio = 10.0
    train_end = int(total_number * (7.0 / total_ratio))       # ~70%
    val_end = train_end + int(total_number * (1.5 / total_ratio))  # ~15%
    # remainder ~15% for test

    # Slice the permuted indices for each split
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    # Gather data for each split
    train_data = trajectories[train_indices]
    val_data = trajectories[val_indices]
    test_data = trajectories[test_indices]
    #---End of data split---

    # Compute alpha based on training set only. 
    train_max = np.max(train_data)
    alpha = train_max / max_target_value 


    train_texts = convert_to_llmtime(train_data, alpha, decimal_places)
    val_texts = convert_to_llmtime(val_data, alpha, decimal_places)
    test_texts = convert_to_llmtime(test_data, alpha, decimal_places)

    return train_texts, val_texts, test_texts


def load_qwen():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    # Freeze all parameters except LM head bias
    for param in model.parameters():
        param.requires_grad = False

    # Add trainable bias to logits
    assert model.lm_head.bias is None
    model.lm_head.bias = torch.nn.Parameter(
        torch.zeros(model.config.vocab_size, device=model.device)
    )
    model.lm_head.bias.requires_grad = True

    return model, tokenizer


class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r: int, alpha: int = None):
        super().__init__()
        assert isinstance(original_linear, nn.Linear)
        self.original_linear = original_linear
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False
        in_dim = original_linear.in_features
        out_dim = original_linear.out_features
        self.r = r
        self.alpha = alpha if alpha else r

        device = original_linear.weight.device
        self.A = nn.Parameter(torch.empty(r, in_dim, device=device))
        self.B = nn.Parameter(torch.zeros(out_dim, r, device=device))
        nn.init.kaiming_normal_(self.A, nonlinearity="linear")

    def forward(self, x):
        base_out = self.original_linear(x)
        lora_out = (x @ self.A.T) @ self.B.T
        return base_out + lora_out * (self.alpha / self.r)


def apply_lora(model, r=4, alpha=None):
    for layer in model.model.layers:
        layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=r, alpha=alpha)
        layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=r, alpha=alpha)
    return model


def load_data(tokenizer, path="lotka_volterra_data.h5", max_ctx_length=512, stride=256):
    train_texts, val_texts, test_texts = load_and_preprocess(
        path, decimal_places=2, max_target_value=9.99
    )

    def process_sequences(texts):
        all_input_ids = []
        for text in texts:
            encoding = tokenizer(text, return_tensors="pt", add_special_tokens=False)
            seq_ids = encoding.input_ids[0]
            for i in range(0, len(seq_ids), stride):
                chunk = seq_ids[i : i + max_ctx_length]
                if len(chunk) < max_ctx_length:
                    chunk = torch.cat(
                        [torch.full((max_ctx_length - len(chunk),), tokenizer.pad_token_id),chunk]
                    )
                all_input_ids.append(chunk)
        return torch.stack(all_input_ids)

    return process_sequences(train_texts), process_sequences(val_texts), process_sequences(test_texts)


def train_lora(model, train_input_ids, learning_rate=1e-5, batch_size=4, max_steps=10000):
    train_dataset = TensorDataset(train_input_ids)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=learning_rate
    )
    accelerator = Accelerator()
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    model.train()
    steps = 0
    losses = []
    pbar = tqdm(total=max_steps, desc="Training")
    while steps < max_steps:
        for (batch,) in train_loader:
            optimizer.zero_grad()
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()

            wandb.log({
                "step": steps,
                "loss": loss.item()
            })

            losses.append(loss.item())
            steps += 1
            pbar.update(1)
            pbar.set_postfix({"loss": loss.item()})
            if steps % 50 == 0:
                tqdm.write(f"Step {steps}: loss = {loss.item():.4f}")
            if steps >= max_steps:
                break
    pbar.close()
        
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    wandb.log({"training_loss_curve": wandb.Image(plt)})
    plt.close()
    return losses


# def evaluation(model, tokenizer, tokenized_data, context_ratio=0.7):
#     """
#     Evaluate the untrained Qwen2.5-Instruct model's forecasting ability by:
#       1) Splitting each tokenized sequence into context and target.
#       2) Generating predictions from the context.
#       3) Computing:
#          - Cross-entropy loss & perplexity over the entire sequence
#          - MSE of the decoded numeric predictions vs. the true target
    
#     Args:
#         model: The untrained Qwen2.5-Instruct model from qwen.py.
#         tokenizer: The Qwen2.5-Instruct tokenizer.
#         tokenized_data (List[torch.Tensor]): List of tokenized sequences (1D tensors).
#         context_ratio (float): Fraction of tokens to use as context.
    
#     Returns:
#         (avg_loss, avg_perplexity, avg_mse): Tuple of floats representing
#         the mean cross-entropy loss, perplexity, and mean squared error (forecast).
#     """
#     losses = []
#     mses = []

#     # Evaluate on a subset (e.g. 10 sequences) for brevity
#     num_eval = 10

    
#     for i in range(num_eval):
#         # 1) Retrieve the i-th tokenized sequence
#         seq = tokenized_data[i].to(device)

#         # 2) Split into context vs target
#         context_ids, target_ids = split_context_target(seq, context_ratio)

#         # 3) Generate predictions from the context
#         input_ids = context_ids.unsqueeze(0)  # add batch dimension
#         max_gen_length = len(target_ids)   # we aim to generate as many tokens as the target
#         with torch.no_grad():
#             generated = model.generate(
#                 input_ids,
#                 max_new_tokens=max_gen_length, 
#                 do_sample=False  # Greedy generation
#             )
        
#         # 4) Compute cross-entropy loss over the entire sequence (context + target)
#         #    The 'labels' argument means the model will compute language modeling loss
#         #    comparing each output token to the same shifted input token.
#         full_seq = seq.unsqueeze(0)
#         with torch.no_grad():
#             output = model(full_seq, labels=full_seq)
#             loss_val = output.loss.item()
#             losses.append(loss_val)
        
#         # 5) Decode the generated tokens for the target portion
#         #    We only look at the newly generated tokens after context_ids
#         generated_ids = generated[0]#
#         predicted_target_ids = generated_ids[len(context_ids):]

#         # Decode both predicted target and true target tokens
#         pred_text = tokenizer.decode(predicted_target_ids, skip_special_tokens=True)
#         true_text = tokenizer.decode(target_ids, skip_special_tokens=True)

#         # Convert them back to numeric sequences
#         pred_numbers = decode_tokens_to_numbers(pred_text)
#         true_numbers = decode_tokens_to_numbers(true_text)

#         # 6) Compute Mean Squared Error if the lengths match
#         mse = float("inf")
#         if len(pred_numbers) == len(true_numbers) and len(pred_numbers) > 0:
#             mse = np.mean((np.array(pred_numbers) - np.array(true_numbers)) ** 2)
#             mses.append(mse)

#             try:
#                 pred_array = np.array(pred_numbers).reshape(-1, 2)
#                 true_array = np.array(true_numbers).reshape(-1, 2)

#                 plt.figure()
#                 plt.plot(true_array[:, 0], label="True Prey", )
#                 plt.plot(true_array[:, 1], label="True Predator",linestyle='dashed')
#                 plt.plot(pred_array[:, 0], label="Predicted Prey")
#                 plt.plot(pred_array[:, 1], label="Predicted Predator",linestyle='dashed')
#                 plt.title(f"Forecast Sequence {i}")
#                 plt.legend()
#                 wandb.log({f"forecast_seq_{i}": wandb.Image(plt)})
#                 plt.close()

#             except ValueError:
#                 print(f"Sequence {i}: could not reshape into (T,2). Skipping plot.")



    
#         wandb.log({
#                 "loss_per_sequence": loss_val,
#                 "mse_per_sequence": mse,
#             }, step=i+1)

#     avg_loss = np.mean(losses) if len(losses) > 0 else float("inf")
#     avg_mse = np.mean(mses) if len(mses) > 0 else float("inf")
    
#     wandb.log({
#     "avg_loss": avg_loss,
#     "avg_mse": avg_mse,
    
# })


#     return avg_loss,  avg_mse

# def evaluation(model, tokenizer, tokenized_data, context_ratio: float = 0.7):
#     """
#     Evaluates the model in a fully autoregressive manner without teacher forcing.
    
#     For each tokenized sequence:
#       1. Split the sequence into context and target parts using split_context_target.
#       2. Initialize the autoregressive generation with the context.
#       3. For each prediction step (i.e. for each token in the target):
#            a. Run the model on the current sequence (which is updated with previously predicted tokens).
#            b. Compute the cross-entropy loss for the predicted token against the ground truth.
#            c. Select the next token using greedy decoding (argmax) from the logits.
#            d. Append the predicted token to the current sequence so that it is used as input for the next step.
#       4. Build a loss curve based on the loss at each prediction step.
#       5. Decode the entire predicted token sequence (for the target part) and the true target tokens into text,
#          then convert them to numeric values and compute the Mean Squared Error (MSE).
#       6. Log the loss curves and forecast comparison plots using wandb.
    
#     Args:
#         model: The Qwen2.5-Instruct model.
#         tokenizer: The corresponding tokenizer.
#         tokenized_data (List[torch.Tensor]): A list of 1D tensors of token IDs.
#         context_ratio (float): The fraction of the sequence used as context.
    
#     Returns:
#         Tuple[float, float]: (avg_loss, avg_mse) averaged over all evaluated sequences.
#     """
#     num_eval = 10  # Evaluate the first 10 sequences for brevity
#     all_seq_losses = []
#     all_seq_mses = []
    
#     for i in range(num_eval):
#         # Get the full sequence and split into context and target parts
#         seq = tokenized_data[i].to(device)
#         window_size = len(seq)
#         context_ids, target_ids = split_context_target(seq, context_ratio)
        
#         # Initialize the autoregressive generation with the context tokens
#         current_sequence = context_ids.clone()  # this will be updated with predicted tokens
#         predicted_tokens = []  # to store predicted tokens (for the target portion)
#         token_losses = []      # to store loss for each predicted token
        
#         # For each token in the target, generate a new token using the previously predicted tokens as input
#         for t in range(len(target_ids)):
#             # Perform a forward pass on the current sequence (unsqueezed to have batch dimension)
#             with torch.no_grad():
#                 outputs = model(current_sequence.unsqueeze(0))
#                 # Obtain logits for the last token position
#                 logits = outputs.logits  # shape: (1, current_seq_len, vocab_size)
#                 last_logits = logits[0, -1, :]  # shape: (vocab_size,)
                
#                 # Compute cross-entropy loss for the predicted token against the ground truth token at this step
#                 # Note: ground_truth_token is from target_ids at position t
#                 ground_truth_token = target_ids[t].unsqueeze(0)  # shape: (1,)
#                 loss_token = torch.nn.functional.cross_entropy(last_logits.unsqueeze(0), ground_truth_token)
#                 token_losses.append(loss_token.item())
                
#                 # Greedy decoding: select the token with the highest probability
#                 predicted_token = torch.argmax(last_logits, dim=-1).unsqueeze(0)  # shape: (1,)
            
#             # Save the predicted token and append it to current_sequence for subsequent predictions
#             predicted_tokens.append(predicted_token.item())
#             current_sequence = torch.cat([current_sequence, predicted_token])
#             if current_sequence.size(0) > window_size:
#                current_sequence = current_sequence[-window_size:]
        
#         # Compute the average loss for the current sequence
#         avg_loss_seq = np.mean(token_losses)
#         all_seq_losses.append(avg_loss_seq)
        
#         # Decode the predicted target tokens and the true target tokens into text,
#         # then convert them to numeric sequences.
#         pred_ids = torch.tensor(predicted_tokens, device=device)
#         pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
#         true_text = tokenizer.decode(target_ids, skip_special_tokens=True)
#         pred_numbers = decode_tokens_to_numbers(pred_text)
#         true_numbers = decode_tokens_to_numbers(true_text)
        
#         mse = float("inf")
#         if len(pred_numbers) == len(true_numbers) and len(pred_numbers) > 0:
#             mse = np.mean((np.array(pred_numbers) - np.array(true_numbers)) ** 2)
#         all_seq_mses.append(mse)
        
#         # Plot and log the loss curve for this sequence
#         plt.figure()
#         plt.plot(token_losses, label="Token Loss")
#         plt.title(f"Loss Curve for Sequence {i}")
#         plt.xlabel("Prediction Step")
#         plt.ylabel("Cross-Entropy Loss")
#         plt.legend()
#         wandb.log({f"loss_curve_seq_{i}": wandb.Image(plt)})
#         plt.close()
        
#         # Plot and log the forecast comparison (decoded numeric values for target tokens)
#         try:
#             pred_array = np.array(pred_numbers).reshape(-1, 2)
#             true_array = np.array(true_numbers).reshape(-1, 2)
#             plt.figure()
#             plt.plot(true_array[:, 0], label="True Prey")
#             plt.plot(true_array[:, 1], label="True Predator", linestyle='dashed')
#             plt.plot(pred_array[:, 0], label="Predicted Prey")
#             plt.plot(pred_array[:, 1], label="Predicted Predator", linestyle='dashed')
#             plt.title(f"Forecast Comparison for Sequence {i}")
#             plt.legend()
#             wandb.log({f"forecast_seq_{i}": wandb.Image(plt)})
#             plt.close()
#         except Exception as e:
#             print(f"Sequence {i}: Plotting error: {e}")
        
#         wandb.log({
#             "loss_per_sequence": avg_loss_seq,
#             "mse_per_sequence": mse,
#         }, step=i+1)
    
#     avg_loss = np.mean(all_seq_losses) if len(all_seq_losses) > 0 else float("inf")
#     avg_mse = np.mean(all_seq_mses) if len(all_seq_mses) > 0 else float("inf")
#     wandb.log({
#         "avg_loss": avg_loss,
#         "avg_mse": avg_mse,
#     })
#     return avg_loss, avg_mse
