from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from datasets import load_dataset

# Load tokenizer and model
model_name = "./opt-fine-tuned"  # Ensure this points to your local model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load and prepare dataset
print("Loading dataset...")
dataset = load_dataset("enryu43/twitter100m_tweets")["train"]

# Filter for the 'tweet' field (the column with the text data you want to train on)
print("Extracting 'tweet' column...")
dataset = dataset.map(lambda x: {"input": x["tweet"]}, remove_columns=dataset.column_names)

# Flatten dataset into train-test split
print("Splitting dataset...")
split_data = dataset.train_test_split(test_size=0.2, seed=42)

# Tokenize dataset
print("Tokenizing dataset...")
def tokenize_function(examples):
    return tokenizer(
        examples["input"],
        truncation=True,
        padding="max_length",
        max_length=500,  # Adjust as needed
    )

tokenized_data = split_data.map(tokenize_function, batched=True, remove_columns=["input"])

# Data collator for causal language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False  # Not using masked language modeling
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./opt-fine-tuned-twitter",  # Directory to save the model
    overwrite_output_dir=True,
    logging_steps=100,
    num_train_epochs=2,  # Adjust epochs as needed
    per_device_train_batch_size=16,  # Adjust batch size based on GPU memory
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    warmup_steps=500,
    weight_decay=0.01,
    fp16=True,  # Enable mixed precision training for faster performance
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_dir="./logs",  # Directory for logs
    save_total_limit=3,
    remove_unused_columns=False,
    report_to="none",  # Disable external reporting tools like WandB
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Start training
print("Starting fine-tuning...")
trainer.train()

# Save the fine-tuned model
print("Saving the fine-tuned model...")
model.save_pretrained("./opt-fine-tuned-twitter")
tokenizer.save_pretrained("./opt-fine-tuned-twitter")

print("Training and saving completed!")


