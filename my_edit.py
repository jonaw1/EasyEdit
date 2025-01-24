from easyeditor import BaseEditor
from easyeditor import MEMITHyperParams

hparams = MEMITHyperParams.from_hparams("./hparams/MEMIT/gpt2-xl.yaml")
hparams.model_name = "gpt2-xl"
prompts = [
    "Ray Charles, the",
    "Grant Hill is a professional",
    "The law in Ikaalinen declares the language",
]
ground_truth = ["piano", "basketball", "Finnish"]
target_new = ["violin", "soccer", "Swedish"]
subject = ["Ray Charles", "Grant Hill", "Ikaalinen"]

editor = BaseEditor.from_hparams(hparams)
metrics, edited_model, _ = editor.edit(
    prompts=prompts,
    ground_truth=ground_truth,
    target_new=target_new,
    subject=subject,
    sequential_edit=True,
)
print(metrics)

from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel
import torch

tokenizer = GPT2Tokenizer.from_pretrained("./hugging_cache/gpt2-xl")
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2LMHeadModel.from_pretrained("./hugging_cache/gpt2-xl").to(device)

correct_prompts = [
    "Ray Charles, the",
    "The law in Ikaalinen declares the language of",
    "Grant Hill is a professional",
]

batch = tokenizer(correct_prompts, return_tensors="pt", padding=True)

pre_edit_outputs = model.generate(
    input_ids=batch["input_ids"].to(device),
    attention_mask=batch["attention_mask"].to(device),
    max_new_tokens=15,
)

post_edit_outputs = edited_model.generate(
    input_ids=batch["input_ids"].to(device),
    attention_mask=batch["attention_mask"].to(device),
    max_new_tokens=15,
)

max_length = batch["input_ids"].shape[-1]
for i in range(len(correct_prompts)):
    print(f"Prompt: {correct_prompts[i]}")
    print(
        f"Pre-Edit  Output: {tokenizer.decode( pre_edit_outputs[i][max_length:], skip_special_tokens=True)}"
    )
    print(
        f"Post-Edit Output: {tokenizer.decode(post_edit_outputs[i][max_length:], skip_special_tokens=True)}"
    )
    print("--" * 50)
