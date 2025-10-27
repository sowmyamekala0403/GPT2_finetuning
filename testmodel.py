# Text generation using the fine-tuned GPT-2 model
# Author: Sowmya
# Description: Loads the saved GPT-2 model and generates text based on a given prompt.

from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the fine-tuned model and tokenizer
print("Loading fine-tuned GPT-2 model...")
model_path = "./finetuned_gpt2_small"   # Path where you saved the model
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Input prompt
prompt_text = "The history of artificial intelligence began in the 1950s when"
inputs = tokenizer(prompt_text, return_tensors="pt")

# Generate continuation
print("Generating text...")
outputs = model.generate(
    **inputs,
    max_length=100,
    num_return_sequences=1,
    temperature=0.8,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

# Decode and display the result
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nGenerated Text:\n")
print(generated_text)
