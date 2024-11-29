from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from datasets import load_dataset

# Load the tokenizer and model
model_name = "facebook/opt-350m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load dataset
dataset = load_dataset("text", data_files={"train": "training_data.txt"})

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset["train"].train_test_split(test_size=0.2)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False  # No masked language modeling for causal models
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./opt-fine-tuned",  # Output directory
    num_train_epochs=2,            # Number of epochs
    per_device_train_batch_size=16,  # Adjust batch size as needed
    eval_strategy="steps",
    save_steps=500,
    eval_steps=500,
    logging_dir="./logs",
    learning_rate=3e-5,            # Learning rate
    warmup_steps=500,              # Warmup steps
    weight_decay=0.01,             # Regularization
    fp16=True,                     # Mixed precision for faster training
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./opt-fine-tuned")
tokenizer.save_pretrained("./opt-fine-tuned")

