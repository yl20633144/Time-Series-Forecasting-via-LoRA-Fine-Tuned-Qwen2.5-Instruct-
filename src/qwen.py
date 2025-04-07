import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_qwen():
    """
    Load the Qwen2.5-0.5B-Instruct model and tokenizer from HuggingFace.

    This function loads the Qwen2.5-Instruct model with all parameters frozen except for the bias in the 
    language modeling head (`lm_head.bias`), which is initialized to zeros and made trainable. This is 
    typically used as a simple form of tuning when full fine-tuning is computationally expensive.

    Returns:
        tuple:
            - model (transformers.PreTrainedModel): The Qwen2.5-Instruct model with frozen parameters except for `lm_head.bias`.
            - tokenizer (transformers.PreTrainedTokenizer): Corresponding tokenizer from HuggingFace.
    """
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
