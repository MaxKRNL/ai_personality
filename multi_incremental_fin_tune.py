import os
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import math

print("CUDA available?", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU available. Check your NVIDIA driver and CUDA installation.")

# =====================================
# Configuration
# =====================================
MODEL_NAME = "facebook/opt-350m"
DATASET_NAME = "enryu43/twitter100m_tweets"
OUTPUT_BASE_DIR = "./opt-fine-tuned-twitter"
MAX_LENGTH = 256
TEST_SIZE = 0.2
TRAIN_EPOCHS = 1          # Reduced epochs for a lighter training step
LEARNING_RATE = 4e-5
PORTION_PERCENT = 0.05
SEED = 42

PER_DEVICE_TRAIN_BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH_SIZE = 8

# Specify which portion numbers to train in this run:
# For example, [1, 2] means train on the first portion and then the second portion in the same run.
PORTION_NUMBERS = [2, 3]

# =====================================
# Load Model and Tokenizer
# =====================================
print("Loading model and tokenizer...")
PREVIOUS_TRAINED_MODEL_DIR = "./opt-fine-tuned-twitter-portion-1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(PREVIOUS_TRAINED_MODEL_DIR)

# =====================================
# Load and Prepare Dataset
# =====================================
print(f"Loading dataset '{DATASET_NAME}'...")
dataset = load_dataset(DATASET_NAME, split="train")

print("Shuffling dataset...")
dataset = dataset.shuffle(seed=SEED)

print("Renaming 'tweet' column to 'input' and removing others...")
dataset = dataset.map(lambda x: {"input": x["tweet"]}, remove_columns=dataset.column_names)

dataset_length = len(dataset)
portion_size = math.ceil(dataset_length * PORTION_PERCENT)
num_portions = math.ceil(dataset_length / portion_size)

print(f"Total dataset length: {dataset_length}")
print(f"Portion size ({int(PORTION_PERCENT*100)}%): {portion_size}")
print(f"Number of portions: {num_portions}")

# Validate portion numbers
for p in PORTION_NUMBERS:
    if p < 1 or p > num_portions:
        raise ValueError(f"Invalid portion number {p}. It should be between 1 and {num_portions}.")

def tokenize_function(examples):
    return tokenizer(
        examples["input"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir=OUTPUT_BASE_DIR,
    overwrite_output_dir=True,
    num_train_epochs=TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    gradient_accumulation_steps=4,
    warmup_steps=500,
    weight_decay=0.01,
    fp16=True,
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_dir="./logs",
    save_total_limit=3,
    remove_unused_columns=False,
    report_to="none",
    seed=SEED,
    ddp_find_unused_parameters=False,
    no_cuda=False
)

print(f"Trainer device will be: {training_args.device}")

# Loop over the specified portions
for portion_number in PORTION_NUMBERS:
    start_idx = (portion_number - 1) * portion_size
    end_idx = min(portion_number * portion_size, dataset_length)
    
    print(f"\n=== Training on portion {portion_number}/{num_portions}: Using samples {start_idx} to {end_idx - 1} ===")
    
    portion_dataset = dataset.select(range(start_idx, end_idx))
    
    print("Splitting portion into train/test sets...")
    split_data = portion_dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)
    train_data = split_data["train"]
    test_data = split_data["test"]
    
    print("Tokenizing training data...")
    tokenized_train = train_data.map(tokenize_function, batched=True, remove_columns=["input"], num_proc=os.cpu_count())
    tokenized_test = test_data.map(tokenize_function, batched=True, remove_columns=["input"], num_proc=os.cpu_count())
    
    tokenized_train.set_format("torch")
    tokenized_test.set_format("torch")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=data_collator,
    )
    
    print(f"Starting training on portion {portion_number}...")
    trainer.train()
    
    portion_output_dir = f"{OUTPUT_BASE_DIR}-portion-{portion_number}"
    print(f"Saving model checkpoint for portion {portion_number} to: {portion_output_dir}")
    trainer.save_model(portion_output_dir)
    tokenizer.save_pretrained(portion_output_dir)
    
    # Update the model to the newly trained model
    model = AutoModelForCausalLM.from_pretrained(portion_output_dir)

print("Training completed for all specified portions!")
