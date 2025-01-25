import requests
import os
from logger_config import setup_logger
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from easyeditor import ROMEHyperParams, BaseEditor
import torch
import json
import random
import numpy as np

logger = setup_logger()

NUM_EDITS_PER_EXECUTION = 99

COUNTERFACT_URL = "https://rome.baulab.info/data/dsets/counterfact.json"
DATA_DIR = "./data"
COUNTERFACT_PATH = os.path.join(DATA_DIR, "counterfact.json")

MODEL_PATH = os.path.join("./hugging_cache", "gpt2-xl")

ROME_CACHE_DIR = "./rome_cache"

# Set the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Create the data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)
logger.info(f"Ensured directory exists: {DATA_DIR}")

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

# Create the model directory if it doesn't exist
os.makedirs(MODEL_PATH, exist_ok=True)
logger.info(f"Ensured directory exists: {MODEL_PATH}")

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

# Load hyperparameters
hparams = ROMEHyperParams.from_hparams("./hparams/ROME/gpt2-xl.yaml")

# Instantiate the editor
logger.info("Instantiating the editor...")
editor = BaseEditor.from_hparams(hparams)

# Load the tokenizer and model
logger.info("Loading the tokenizer and base model...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

# Move the model to the selected device
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(device)

with open("used_case_ids.json", "r") as f:
    used_case_ids = json.load(f)

counterfact_len = len(counterfact)

# Create ROME cache if it doesn't exist
os.makedirs(ROME_CACHE_DIR, exist_ok=True)
logger.info(f"Ensured directory exists: {ROME_CACHE_DIR}")

for i in range(NUM_EDITS_PER_EXECUTION):
    # Find case id (fact) that has not been used for editing before
    random_case_id = random.randint(0, counterfact_len - 1)
    while used_case_ids.get(random_case_id) == True:
        random_case_id = random.randint(0, counterfact_len - 1)

    used_case_ids[random_case_id] = True

    cf = counterfact[random_case_id]

    subjects = [cf["requested_rewrite"]["subject"]]
    prompts = [
        cf["requested_rewrite"]["prompt"].format(cf["requested_rewrite"]["subject"])
    ]
    gt = [cf["requested_rewrite"]["target_true"]["str"]]
    tn = [cf["requested_rewrite"]["target_new"]["str"]]

    logger.info(
        f"({i + 1}/{NUM_EDITS_PER_EXECUTION}) Editing model for: {prompts[0]}... {gt[0]} -> {tn[0]}..."
    )
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=gt,
        target_new=tn,
        subject=subjects,
        sequential_edit=True,
    )

    params_e = edited_model.transformer.h[17].mlp.c_proj.weight.detach().cpu().numpy()
    params_e = params_e.astype(np.float32)

    np.savez_compressed(f"{ROME_CACHE_DIR}/{random_case_id}.npz", arr=params_e)

with open("used_case_ids.json", "w") as f:
    json.dump(used_case_ids, f)
logger.info(f"Finished. There are now {len(used_case_ids.keys())} facts used")
