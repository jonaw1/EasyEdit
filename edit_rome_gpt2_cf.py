import requests
import os
from logger_config import setup_logger
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from easyeditor import ROMEHyperParams, BaseEditor
import torch
import json

logger = setup_logger()

EXAMPLE_RELATIONS = ["P37"]
NUM_EDITS_PER_RELATION = 3

COUNTERFACT_URL = "https://rome.baulab.info/data/dsets/counterfact.json"
DATA_DIR = "./data"
COUNTERFACT_PATH = os.path.join(DATA_DIR, "counterfact.json")

MODEL_PATH = os.path.join("./hugging_cache", "gpt2-xl")

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

# Group counterfacts by relation (category)
cf_per_relation = {}

for x in counterfact:
    relation = x["requested_rewrite"]["relation_id"]
    tmp = cf_per_relation.get(relation, [])
    tmp.append(x)
    cf_per_relation[relation] = tmp

# Load the tokenizer and model
logger.info("Loading the tokenizer and base model...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

# Move the model to the selected device
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(device)

for relation in EXAMPLE_RELATIONS:
    for cf in cf_per_relation[relation][:NUM_EDITS_PER_RELATION]:
        subjects = [cf["requested_rewrite"]["subject"]]
        prompts = [
            cf["requested_rewrite"]["prompt"].format(cf["requested_rewrite"]["subject"])
        ]
        gt = [cf["requested_rewrite"]["target_true"]["str"]]
        tn = [cf["requested_rewrite"]["target_new"]["str"]]

        logger.info(f"Editing model for: {prompts[0]}... {gt[0]} -> {tn[0]}...")
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            ground_truth=gt,
            target_new=tn,
            subject=subjects,
            sequential_edit=True,
        )

        # Tokenize the prompts
        batch = tokenizer(prompts, return_tensors="pt", padding=True)

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
        for i in range(len(prompts)):
            logger.info(f"Prompt: {prompts[i]}")
            logger.info(
                f"Pre-Edit Output: {tokenizer.decode(pre_edit_outputs[i][max_length:], skip_special_tokens=True)}"
            )
            logger.info(
                f"Post-Edit Output: {tokenizer.decode(post_edit_outputs[i][max_length:], skip_special_tokens=True)}"
            )
            logger.info("--" * 50)

