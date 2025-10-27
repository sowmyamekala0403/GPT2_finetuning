# Fine-tuning GPT-2 model on WikiText-2 dataset (small subset)
# Author: Sowmya
# Description: Quick demo of GPT-2 fine-tuning using Hugging Face Transformers.

from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

print("Loading WikiText-2 dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Using only a small subset for faster training on CPU
train_data = dataset["train"].select(range(500))

print("Loading GPT-2 model and tokenizer...")
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=64
    )

print("Tokenizing dataset...")
tokenized_train = train_data.map(tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir="./results_small",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    save_total_limit=1,
    logging_steps=20,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    data_collator=data_collator
)

print("Starting fine-tuning...")
trainer.train()

trainer.save_model("./finetuned_gpt2_small")
print("Model fine-tuning complete. Saved to ./finetuned_gpt2_small")
