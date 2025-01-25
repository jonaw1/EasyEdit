from easyeditor import BaseEditor
from easyeditor import MEMITHyperParams
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import os
from logger_config import setup_logger

logger = setup_logger()

# Set the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Define the local directory for the model
local_dir = "./hugging_cache/gpt2-xl"

# Create the directory if it doesn't exist
os.makedirs(local_dir, exist_ok=True)
logger.info(f"Ensured directory exists: {local_dir}")

# Download the model if it doesn't already exist
if not os.listdir(local_dir):  # Check if the directory is empty
    logger.info(f"Model not found at {local_dir}. Downloading...")
    model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
    model.save_pretrained(local_dir)
    tokenizer.save_pretrained(local_dir)
    logger.info(f"Model and tokenizer saved to {local_dir}")
else:
    logger.info(f"Model found at {local_dir}. Skipping download.")

# Load hyperparameters
hparams = MEMITHyperParams.from_hparams("./hparams/MEMIT/gpt2-xl.yaml")

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
tokenizer = GPT2Tokenizer.from_pretrained(local_dir)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

# Move the model to the selected device
model = GPT2LMHeadModel.from_pretrained(local_dir).to(device)

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
        f"Pre-Edit Output: {tokenizer.decode(pre_edit_outputs[i][max_length:], skip_special_tokens=True)}"
    )
    logger.info(
        f"Post-Edit Output: {tokenizer.decode(post_edit_outputs[i][max_length:], skip_special_tokens=True)}"
    )
    logger.info("--" * 50)
