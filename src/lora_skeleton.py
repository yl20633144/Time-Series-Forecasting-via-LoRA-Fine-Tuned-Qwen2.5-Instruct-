import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator
from .preprocessor import load_and_preprocess
from .qwen import load_qwen
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt


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

    """
    Apply LoRA (Low-Rank Adaptation) to the query and value projection layers of a Qwen2.5-Instruct model.

    Args:
        model (transformers.PreTrainedModel): The base Qwen model to modify.
        r (int, optional): LoRA rank. Defaults to 4.
        alpha (int, optional): LoRA alpha scaling factor. If None, defaults to r.

    Returns:
        transformers.PreTrainedModel: The modified model with LoRA modules injected.
    """
    for layer in model.model.layers:
        layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=r, alpha=alpha)
        layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=r, alpha=alpha)
    return model




def load_data(tokenizer, path="lotka_volterra_data.h5", max_ctx_length=512, stride=256):
    """
    Load and tokenize the Lotka-Volterra dataset using LLMTIME scheme.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer from Qwen model.
        path (str): Path to the dataset HDF5 file. Defaults to "lotka_volterra_data.h5".
        max_ctx_length (int): Maximum context length for token sequences. Defaults to 512.
        stride (int): Sliding window stride size. Defaults to 256.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            Tokenized and padded tensors for train, validation, and test datasets.
    """
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


_,tokenizer=load_qwen()

def train_lora(model, train_input_ids, learning_rate=1e-5, batch_size=4, max_steps=10000):
    """
    Train the Qwen model with LoRA-adapted layers using the given training inputs.

    Args:
        model (transformers.PreTrainedModel): The Qwen model with LoRA applied.
        train_input_ids (torch.Tensor): Tokenized training data tensor.
        learning_rate (float): Learning rate for Adam optimizer. Defaults to 1e-5.
        batch_size (int): Training batch size. Defaults to 4.
        max_steps (int): Maximum training steps. Defaults to 10000.

    Returns:
        List[float]: A list of training losses at each step.
    """
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
            labels = batch.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            outputs = model(batch, labels=labels)

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
