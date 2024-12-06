from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from datasets import load_dataset
from accelerate import Accelerator

# Initialize Hugging Face Accelerate
accelerator = Accelerator()

# Check device
print(f"Using device: {accelerator.device}")

# Load the Tokenizer and Model
model_name = "./opt-fine-tuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model to the appropriate device
model = accelerator.prepare(model)

# Load and Prepare Datasets
print("Loading datasets...")

# Load enryu43/twitter100m_tweets dataset and select the 'tweet' column (3rd column)
dataset = load_dataset("enryu43/twitter100m_tweets")["train"]

# Select only the `tweet` column
print("Selecting the tweet column...")
dataset = dataset.map(lambda x: {"text": x["tweet"]}, remove_columns=dataset.column_names)

# Split the dataset into train and test
print("Splitting the dataset...")
split_dataset = dataset.train_test_split(test_size=0.2)

# Tokenize the Dataset
print("Tokenizing the dataset...")
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=500,
    )

# Apply tokenization to both train and test splits
tokenized_dataset = split_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False  # No masked language modeling for causal models
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./opt-fine-tuned-2",  # Output directory
    num_train_epochs=1,              # Number of epochs
    per_device_train_batch_size=32,  # Adjust batch size as needed
    per_device_eval_batch_size=32,   # Batch size for evaluation
    eval_strategy="steps",           # Evaluate at regular intervals
    save_steps=500,
    eval_steps=500,
    logging_dir="./logs",            # Directory for logging
    logging_steps=100,
    learning_rate=1e-5,              # Learning rate
    warmup_steps=100,                # Warmup steps
    weight_decay=0.01,               # Regularization
    fp16=True,                       # Mixed precision for faster training
    save_total_limit=2,              # Limit checkpoints to save disk space
    report_to="none",                # Avoid unnecessary reports (e.g., WandB or TensorBoard)
)

# Prepare datasets using accelerate
train_dataset = accelerator.prepare(tokenized_dataset["train"])
eval_dataset = accelerator.prepare(tokenized_dataset["test"])

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Train the model
print("Starting training...")
trainer.train()

# Save the fine-tuned model
print("Saving the fine-tuned model...")
model.save_pretrained("./opt-fine-tuned-2")
tokenizer.save_pretrained("./opt-fine-tuned-2")

print("Training and saving completed!")






