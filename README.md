🧠 GPT-2 Fine-Tuning Project

This project demonstrates how to fine-tune the GPT-2 model using the Hugging Face transformers library.
It includes Python scripts for both training and testing the fine-tuned model on a custom dataset.

📁 Project Structure
Finetuning_GPT2/
│
├── finetune_gpt2.py         # Script to fine-tune GPT-2 on your dataset
├── test_model.py            # Script to test the fine-tuned model
├── requirements.txt         # Python dependencies
├── .gitignore               # Ignored files/folders (includes large model files & venv)
└── README.md                # Project documentation

⚙️ Setup Instructions
1️⃣ Clone the repository
git clone https://github.com/sowmyamekala0403/GPT2_finetuning.git
cd GPT2_finetuning

2️⃣ Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate     # (Windows)
# source .venv/bin/activate  # (Linux/Mac)

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ (Optional) Add your own dataset

Place your dataset file in the project directory and update its path inside the finetune_gpt2.py script.

🚀 How to Run
🔹 Fine-tune GPT-2
python finetune_gpt2.py

🔹 Test the fine-tuned model
python test_model.py

🧩 Key Features

Fine-tunes GPT-2 for custom text generation tasks

Uses Hugging Face Transformers and Datasets

Saves fine-tuned weights locally for reuse

Includes .gitignore to avoid committing large model files or virtual environments

Step-by-step debugging support for Git and GitHub errors

⚠️ Common Errors Faced and Fixes

Below are the actual errors faced during the project setup and GitHub upload — with explanations and fixes.

Error	Meaning / Cause	Fix / Solution
remote: error: File model.safetensors is 474.71 MB; exceeds 100 MB limit	GitHub does not allow files > 100 MB	Add the folder to .gitignore or use git-filter-repo to remove large files
error: failed to push some refs to 'https://github.com/... (non-fast-forward)	Your local branch is behind the GitHub branch	Run git pull origin main --rebase before pushing or use --force if overwriting
fatal: pathspec '.venv' did not match any files	You tried to remove something that’s not tracked by Git	Ensure the folder exists or already added to .gitignore
Aborting: Refusing to destructively overwrite repo history	git-filter-repo refused to modify a non-fresh repo	Use the --force flag: python -m git_filter_repo --path ... --invert-paths --force
Deletion of directory '.git/objects/..' failed	Windows locked Git’s internal folders	Close VS Code/File Explorer → retry or skip (n)
fatal: 'origin' does not appear to be a git repository	You didn’t add a GitHub remote yet	Run git remote add origin <repo-URL>
remote: GH001: Large files detected	Repository contains files >100MB even after deletion	Run git rm --cached <filename> and commit again, or use git-filter-repo
🧠 Example Usage

Here’s how you can load and generate text using your fine-tuned GPT-2 model:

from transformers import GPT2Tokenizer, GPT2LMHeadModel

model_path = "./finetuned_gpt2_small"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

input_text = "Once upon a time"
inputs = tokenizer.encode(input_text, return_tensors="pt")
outputs = model.generate(inputs, max_length=50, temperature=0.7)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

💡 Tips

Always commit and push only code files, not model weights or datasets.

Keep .gitignore updated to exclude:

.venv/
__pycache__/
finetuned_gpt2_small/
results_small/


To store models, use Google Drive, Hugging Face Hub, or Git LFS
