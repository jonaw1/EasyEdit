from easyeditor import BaseEditor
from easyeditor import MEMITHyperParams
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from logger_config import setup_logger

logger = setup_logger()

# Set the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load hyperparameters
hparams = MEMITHyperParams.from_hparams("./hparams/MEMIT/gpt2-xl.yaml")
hparams.model_name = "gpt2-xl"

# Define prompts and targets
prompts = [
    "Ray Charles, the",
    "Grant Hill is a professional",
    "The law in Ikaalinen declares the language",
]
ground_truth = ["piano", "basketball", "Finnish"]
target_new = ["violin", "soccer", "Swedish"]
subject = ["Ray Charles", "Grant Hill", "Ikaalinen"]

# Instantiate the editor
logger.info("Instantiating the editor...")
editor = BaseEditor.from_hparams(hparams)

# Perform the edit on the selected device
logger.info("Performing the edit...")
metrics, edited_model, _ = editor.edit(
    prompts=prompts,
    ground_truth=ground_truth,
    target_new=target_new,
    subject=subject,
    sequential_edit=True,
)
logger.info(f"Edit metrics: {metrics}")

# Load the tokenizer and model
logger.info("Loading the tokenizer and base model...")
tokenizer = GPT2Tokenizer.from_pretrained("./hugging_cache/gpt2-xl")
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

# Move the model to the selected device
model = GPT2LMHeadModel.from_pretrained("./hugging_cache/gpt2-xl").to(device)

# Define prompts for evaluation
correct_prompts = [
    "Ray Charles, the",
    "The law in Ikaalinen declares the language of",
    "Grant Hill is a professional",
]

# Tokenize the prompts
batch = tokenizer(correct_prompts, return_tensors="pt", padding=True)

# Move input tensors to the selected device
input_ids = batch["input_ids"].to(device)
attention_mask = batch["attention_mask"].to(device)

# Generate pre-edit outputs
logger.info("Generating pre-edit outputs...")
pre_edit_outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=15,
)

# Generate post-edit outputs
logger.info("Generating post-edit outputs...")
post_edit_outputs = edited_model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=15,
)

# Decode and print results
max_length = batch["input_ids"].shape[-1]
for i in range(len(correct_prompts)):
    logger.info(f"Prompt: {correct_prompts[i]}")
    logger.info(
        f"Pre-Edit  Output: {tokenizer.decode(pre_edit_outputs[i][max_length:], skip_special_tokens=True)}"
    )
    logger.info(
        f"Post-Edit Output: {tokenizer.decode(post_edit_outputs[i][max_length:], skip_special_tokens=True)}"
    )
    logger.info("--" * 50)