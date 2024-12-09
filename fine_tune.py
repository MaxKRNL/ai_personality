import pandas as pd
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from accelerate import Accelerator

# =====================================
# Configuration
# =====================================
CSV_FILE = "training_data/cleaned_tweets.csv"   # Path to your CSV file
TEXT_COLUMN = "clean_text"       # Column containing text
MODEL_NAME = "facebook/opt-350m"
OUTPUT_DIR = "./opt-fine-tuned"
MAX_LENGTH = 512
TEST_SIZE = 0.2

# =====================================
# Initialize Accelerator
# =====================================
accelerator = Accelerator()

# Check the device
print(f"Using device: {accelerator.device}")

# Load the tokenizer and model
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# =====================================
# Load dataset from CSV
# =====================================
print("Loading dataset from CSV...")
df = pd.read_csv(CSV_FILE, dtype=str)
# Filter out rows with empty text if necessary
df = df.dropna(subset=[TEXT_COLUMN])
df = df[df[TEXT_COLUMN].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)]

# Create a Hugging Face Dataset from the DataFrame
dataset = Dataset.from_pandas(df, preserve_index=False)

# =====================================
# Tokenization function
# =====================================
def tokenize_function(examples):
    # For causal LM, we just tokenize the text itself
    # If you want, you could add special tokens, or different formatting.
    return tokenizer(examples[TEXT_COLUMN], truncation=True, padding="max_length", max_length=MAX_LENGTH)

# Tokenize the entire dataset
print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

# =====================================
# Train-test split
# =====================================
print(f"Splitting dataset into train/test with test_size={TEST_SIZE}...")
train_test_split = tokenized_dataset.train_test_split(test_size=TEST_SIZE)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# =====================================
# Data Collator
# =====================================
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We're training a causal language model, no masked LM
)

# =====================================
# Training Arguments
# =====================================
# You can adjust these based on your needs and GPU memory.
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=8,   # Adjust based on GPU memory
    per_device_eval_batch_size=8,
    eval_strategy="steps",
    save_steps=500,
    eval_steps=500,
    logging_dir="./logs",
    learning_rate=3e-5,
    warmup_steps=500,
    weight_decay=0.01,
    fp16=True,   # Use mixed precision if supported
    report_to="none"  # Prevent reporting to external services
)

# =====================================
# Initialize Trainer
# =====================================
print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator
)

# =====================================
# Train the model
# =====================================
print("Starting fine-tuning...")
trainer.train()

# =====================================
# Save the fine-tuned model
# =====================================
print("Saving fine-tuned model and tokenizer...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Fine-tuning complete!")





