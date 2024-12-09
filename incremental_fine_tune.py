from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import math

# =====================================
# Configuration
# =====================================
MODEL_NAME = "./opt-fine-tuned"        # Path to your local fine-tuned model, or a base model
DATASET_NAME = "enryu43/twitter100m_tweets"
OUTPUT_BASE_DIR = "./opt-fine-tuned-twitter"
MAX_LENGTH = 300                       # Max sequence length for tokenization
TEST_SIZE = 0.2                        # Fraction of each portion used as test set
TRAIN_EPOCHS = 2                       # Number of epochs per portion
LEARNING_RATE = 5e-5
PORTION_PERCENT = 0.05                 # Train in increments of 5%
SEED = 42                              # Random seed for reproducibility

# Adjust batch size based on available GPU memory
PER_DEVICE_TRAIN_BATCH_SIZE = 16
PER_DEVICE_EVAL_BATCH_SIZE = 16

# =====================================
# Load Model and Tokenizer
# =====================================
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# =====================================
# Load and Prepare Dataset
# =====================================
print(f"Loading dataset '{DATASET_NAME}'...")
dataset = load_dataset(DATASET_NAME, split="train")

print("Shuffling dataset...")
dataset = dataset.shuffle(seed=SEED)

# Extract 'tweet' column as 'input' column, and remove everything else
print("Renaming 'tweet' column to 'input' and removing others...")
dataset = dataset.map(lambda x: {"input": x["tweet"]}, remove_columns=dataset.column_names)

# Calculate portion size (5% increments)
dataset_length = len(dataset)
portion_size = math.ceil(dataset_length * PORTION_PERCENT)

# Number of portions is how many 5% chunks we can get out of the dataset.
num_portions = math.ceil(dataset_length / portion_size)

print(f"Total dataset length: {dataset_length}")
print(f"Portion size (5%): {portion_size}")
print(f"Number of 5% portions to train on: {num_portions}")

# =====================================
# Tokenization Function
# =====================================
def tokenize_function(examples):
    """
    Tokenizes text from the 'input' field.
    Applies truncation and padding to a fixed length.
    """
    return tokenizer(
        examples["input"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

# =====================================
# Data Collator
# =====================================
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # For causal language modeling, we do not use MLM
)

# =====================================
# Training Arguments
# =====================================
training_args = TrainingArguments(
    output_dir=OUTPUT_BASE_DIR,
    overwrite_output_dir=True,
    num_train_epochs=TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    gradient_accumulation_steps=4,   # Accumulate gradients over 4 mini-batches
    warmup_steps=500,
    weight_decay=0.01,
    fp16=True,                 # Use mixed precision if supported by your GPU
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_dir="./logs",
    save_total_limit=3,
    remove_unused_columns=False,
    report_to="none",
    seed=SEED
)

# =====================================
# Training in 5% Portions
# =====================================
for i in range(num_portions):
    start_idx = i * portion_size
    end_idx = min((i + 1) * portion_size, dataset_length)

    # If the portion doesn't have enough samples (can happen at the end), break out.
    if start_idx >= dataset_length:
        break

    print(f"\n=== Portion {i+1}/{num_portions}: Using samples {start_idx} to {end_idx-1} ===")

    # Select the current 5% portion from the dataset
    portion_dataset = dataset.select(range(start_idx, end_idx))
    
    # Split the portion into train/test sets
    print("Splitting portion into train/test sets...")
    split_data = portion_dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)
    train_data = split_data["train"]
    test_data = split_data["test"]
    
    # Tokenize the training and testing data
    print("Tokenizing training data...")
    tokenized_train = train_data.map(tokenize_function, batched=True, remove_columns=["input"])
    
    print("Tokenizing testing data...")
    tokenized_test = test_data.map(tokenize_function, batched=True, remove_columns=["input"])
    
    # Initialize Trainer for this portion
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train on the current portion
    print("Starting training on the current 5% portion...")
    trainer.train()
    
    # Save the model after completing this portion
    portion_output_dir = f"{OUTPUT_BASE_DIR}-portion-{i+1}"
    print(f"Saving model checkpoint for portion {i+1} to: {portion_output_dir}")
    model.save_pretrained(portion_output_dir)
    tokenizer.save_pretrained(portion_output_dir)

    # Update the model to the newly trained weights before moving on to the next portion
    model = AutoModelForCausalLM.from_pretrained(portion_output_dir)

print("\nTraining on all portions completed successfully!")



