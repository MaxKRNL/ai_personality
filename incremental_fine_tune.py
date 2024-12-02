from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from datasets import load_dataset

# Load the Previously Fine-Tuned Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained("./opt-fine-tuned")
model = AutoModelForCausalLM.from_pretrained("./opt-fine-tuned")

# Load New Data 
print("Loading the BlueSky dataset...")
dataset = load_dataset("alpindale/two-million-bluesky-posts")

# Tokenize the Dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_dataset = tokenized_dataset["train"].train_test_split(test_size=0.2)

# Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False            # No masked language modeling for causal models
)

# Define Incremental Fine-Tuning Arguments
training_args = TrainingArguments(
    output_dir="./opt-fine-tuned-2",          # Directory for the updated model
    num_train_epochs=1,                       # Use fewer epochs for incremental updates
    per_device_train_batch_size=16,
    eval_strategy="epoch",              # Evaluate at the end of each epoch
    save_strategy="epoch",
    save_total_limit=2,
    logging_dir="./logs-2",
    learning_rate=1e-5,                       # Lower learning rate for fine-tuning
    warmup_steps=100,                         # Shorter warmup for incremental training
    weight_decay=0.01,                        # Regularization
    fp16=True,                                # Mixed precision
)

# Train the Model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
)

print("Starting incremental fine-tuning with multiple files...")
trainer.train()

# Save the Updated Model
model.save_pretrained("./opt-fine-tuned-2")
tokenizer.save_pretrained("./opt-fine-tuned-2")

print("Incremental fine-tuning with multiple files completed!")

