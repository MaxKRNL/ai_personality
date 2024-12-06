from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from datasets import load_dataset
from accelerate import Accelerator

# Initialize Hugging Face Accelerate
accelerator = Accelerator()

# Check device
print(f"Using device: {accelerator.device}")

# Load the tokenizer and model
model_name = "facebook/opt-350m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Prepare model for Accelerate
model = accelerator.prepare(model)

# Load dataset
print("Loading dataset...")
dataset = load_dataset("text", data_files={"train": "training_data/training_data.txt"})

# Tokenize dataset
print("Tokenizing dataset...")
def tokenize_function(examples):
    if "text" not in examples:
        raise ValueError("Expected 'text' field in dataset examples.")
    # Apply truncation and padding during tokenization
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",  # Ensures consistent tensor dimensions
        max_length=500,        # Adjust as needed
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
split_dataset = tokenized_dataset["train"].train_test_split(test_size=0.2)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False  # No masked language modeling for causal models
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./opt-fine-tuned",  # Output directory
    overwrite_output_dir=True,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_dir="./logs",
    logging_steps=100,
    num_train_epochs=2,            # Number of epochs
    per_device_train_batch_size=16,  # Adjust batch size
    per_device_eval_batch_size=16,
    learning_rate=3e-5,
    warmup_steps=500,
    weight_decay=0.01,
    fp16=True,                     # Enable mixed precision
    save_total_limit=3,            # Limit checkpoints
    report_to="none",              # Disable reporting to external tools
    remove_unused_columns=False,   # Use all dataset columns
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    data_collator=data_collator,
)

# Train the model
print("Starting fine-tuning...")
trainer.train()

# Save the fine-tuned model
print("Saving the fine-tuned model...")
model.save_pretrained("./opt-fine-tuned")
tokenizer.save_pretrained("./opt-fine-tuned")

print("Fine-tuning completed!")



