from logger_config import setup_logger
import json
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import requests
import numpy as np

USED_IDS_FILE = "used_case_ids.json"
MODEL_PATH = "./hugging_cache/gpt2-xl"
COUNTERFACT_URL = "https://rome.baulab.info/data/dsets/counterfact.json"
DATA_DIR = "./data"
COUNTERFACT_PATH = os.path.join(DATA_DIR, "counterfact.json")

logger = setup_logger()

# Set the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

with open(USED_IDS_FILE, "r") as f:
    used_case_ids = json.load(f)

num_ids = len(used_case_ids)
logger.info(f"There are {num_ids} ids in the used_case_ids file.")

# Download the model if it doesn't already exist
if not os.listdir(MODEL_PATH):  # Check if the directory is empty
    logger.info(f"Model not found at {MODEL_PATH}. Downloading...")
    model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    logger.info(f"Model and tokenizer saved to {MODEL_PATH}")
else:
    logger.info(f"Model found at {MODEL_PATH}. Skipping download.")

# Load the tokenizer and model
logger.info("Loading the tokenizer and base models...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

# Move the model to the selected device
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(device)
edited_model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(device)

# Download the dataset if it doesn't already exist
if not os.path.exists(COUNTERFACT_PATH):
    logger.info(f"Counterfact dataset not found at {COUNTERFACT_PATH}. Downloading...")
    response = requests.get(COUNTERFACT_URL)
    with open(COUNTERFACT_PATH, "wb") as f:
        f.write(response.content)
    logger.info(f"Counterfact dataset saved to {COUNTERFACT_PATH}")
else:
    logger.info(f"Counterfact dataset found at {COUNTERFACT_PATH}. Skipping download.")

# Open the dataset
with open(COUNTERFACT_PATH, "r") as f:
    counterfact = json.load(f)

successful_edit_ids = []
unsuccessful_edit_ids = []

for j, id in enumerate(used_case_ids):
    logger.info(f"({j + 1}/{num_ids}) Analyzing counterfact with case id {id}... {ground_truth} -> {target_new}...")

    # Updating edited model
    edited_weights_path = os.path.join("rome_cache", f"{id}.npz")
    logger.info(f"Loading edited weights from {edited_weights_path}...")
    loaded_params = np.load(edited_weights_path)
    params_e = loaded_params["arr"]

    # Convert the numpy array back to a PyTorch tensor
    params_e_tensor = torch.from_numpy(params_e).to(device)

    # Assign the loaded weights back to the edited model
    with torch.no_grad():
        edited_model.transformer.h[17].mlp.c_proj.weight.copy_(params_e_tensor)

    logger.info("Edited weights successfully loaded and assigned to the edited model.")

    cf = counterfact[id]
    original_prompt = cf["requested_rewrite"]["prompt"].format(
        cf["requested_rewrite"]["subject"]
    )
    paraphrase_prompts = cf["paraphrase_prompts"]
    ground_truth = cf["requested_rewrite"]["target_true"]["str"]
    target_new = cf["requested_rewrite"]["target_new"]["str"]

    batch = batch = tokenizer(paraphrase_prompts, return_tensors="pt", padding=True)

    # Generate pre-edit outputs
    logger.info("Generating pre-edit outputs...")
    pre_edit_outputs = model.generate(
        input_ids=batch["input_ids"].to(model.device),
        attention_mask=batch["attention_mask"].to(model.device),
        max_new_tokens=15,
    )

    # Generate post-edit outputs
    logger.info("Generating post-edit outputs...")
    post_edit_outputs = edited_model.generate(
        input_ids=batch["input_ids"].to(edited_model.device),
        attention_mask=batch["attention_mask"].to(edited_model.device),
        max_new_tokens=15,
    )

    max_length = batch["input_ids"].shape[-1]
    successful = True
    for i in range(len(paraphrase_prompts)):
        print(f"Paraphrase prompt: {paraphrase_prompts[i]}")
        print(
            f"Pre-Edit Output: {tokenizer.decode( pre_edit_outputs[i][max_length:], skip_special_tokens=True)}"
        )
        post_edit_output = tokenizer.decode(
            post_edit_outputs[i][max_length:], skip_special_tokens=True
        )
        print(f"Post-Edit Output: {post_edit_output}")
        if target_new not in post_edit_output:
            logger.info(f"{target_new} not in output, discarding counterfact with case id {id}...")
            successful = False
            unsuccessful_edit_ids.append(id)
            break
        print("--" * 50)

    if successful:
        successful_edit_ids.append(id)
        logger.info(f"Counterfact with case id {id} successful!")

logger.info(f"Successful edit IDs: {successful_edit_ids}")
logger.info(f"Unsuccessful edit IDs: {unsuccessful_edit_ids}")

# Save the successful and unsuccessful edit IDs to a JSON file
logger.info("Saving successful and unsuccessful edit IDs to JSON files...")

with open("successful_edit_ids.json", "w") as f:
    json.dump(successful_edit_ids, f)

with open("unsuccessful_edit_ids.json", "w") as f:
    json.dump(unsuccessful_edit_ids, f)
