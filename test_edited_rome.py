import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from logger_config import setup_logger
import os
import json
import random

logger = setup_logger()
logger.info(f"Running script {os.path.basename(__file__)}...")

MODEL_PATH = "./hugging_cache/gpt2-xl"
ROME_CACHE_DIR = "./rome_cache"
COUNTERFACT_PATH = "./data/counterfact.json"

# Load the base model and tokenizer
logger.info("Loading the base model and tokenizer...")
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)

# Move the model to the appropriate device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
model = model.to(device)

# Chose random case id (fact)
with open("used_case_ids.json", "r") as f:
    used_case_ids = json.load(f)

used_case_ids_list = used_case_ids.keys()
random_case_id = random.choice(list(used_case_ids_list))
logger.info(f"Using random case id: {random_case_id}")

# Load the fact from the dataset
with open(COUNTERFACT_PATH, "r") as f:
    counterfact = json.load(f)
cf = counterfact[random_case_id]
prompts = [cf["requested_rewrite"]["prompt"].format(cf["requested_rewrite"]["subject"])]

# Load the edited parameters from the .npz file
edited_params_path = f"{ROME_CACHE_DIR}/{random_case_id}.npz"

# Load the numpy array from the .npz file
logger.info(f"Loading edited parameters from {edited_params_path}...")
with np.load(edited_params_path) as data:
    edited_params = data[
        "arr"
    ]  # The key "arr" is used when saving with np.savez_compressed

# Convert the numpy array to a PyTorch tensor
logger.info("Converting edited parameters to PyTorch tensor...")
edited_params_tensor = torch.tensor(edited_params, device=device)

# Apply the edited parameters to the model
logger.info("Applying edited parameters to the model...")
with torch.no_grad():
    model.transformer.h[17].mlp.c_proj.weight.copy_(edited_params_tensor)

# Tokenize the prompts
batch = tokenizer(prompts, return_tensors="pt", padding=True)

# Move input tensors to the selected device
input_ids = batch["input_ids"].to(device)
attention_mask = batch["attention_mask"].to(device)


# Generate post-edit outputs
logger.info("Generating outputs...")
post_edit_outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=15,
)

logger.info("Prompt: " + prompts[0])
logger.info("Expected output: " + cf["requested_rewrite"]["target_new"]["str"])
max_length = batch["input_ids"].shape[-1]
logger.info(
    f"Generated output: {tokenizer.decode(post_edit_outputs[0][max_length:], skip_special_tokens=True)}"
)
