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
MAX_LENGTH = 300
TEST_SIZE = 0.2
TRAIN_EPOCHS = 2
LEARNING_RATE = 4e-5
PORTION_PERCENT = 0.05
SEED = 42

PER_DEVICE_TRAIN_BATCH_SIZE = 16
PER_DEVICE_EVAL_BATCH_SIZE = 16

# =====================================
# Load Model and Tokenizer
# =====================================
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).cuda()

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
print(f"Portion size (5%): {portion_size}")
print(f"Number of 5% portions: {num_portions}")

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
    no_cuda=False
)

print(f"Trainer device will be: {training_args.device}")

for i in range(num_portions):
    start_idx = i * portion_size
    end_idx = min((i + 1) * portion_size, dataset_length)
    if start_idx >= dataset_length:
        break

    print(f"\n=== Portion {i+1}/{num_portions}: Using samples {start_idx} to {end_idx-1} ===")
    portion_dataset = dataset.select(range(start_idx, end_idx))
    
    print("Splitting portion into train/test sets...")
    split_data = portion_dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)
    train_data = split_data["train"]
    test_data = split_data["test"]
    
    print("Tokenizing training data...")
    tokenized_train = train_data.map(tokenize_function, batched=True, remove_columns=["input"], num_proc=os.cpu_count())
    
    print("Tokenizing testing data...")
    tokenized_test = test_data.map(tokenize_function, batched=True, remove_columns=["input"], num_proc=os.cpu_count())
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Starting training on the current 5% portion...")
    trainer.train()
    
    portion_output_dir = f"{OUTPUT_BASE_DIR}-portion-{i+1}"
    print(f"Saving model checkpoint for portion {i+1} to: {portion_output_dir}")
    model.save_pretrained(portion_output_dir)
    tokenizer.save_pretrained(portion_output_dir)

print("\nTraining on all portions completed successfully!")




