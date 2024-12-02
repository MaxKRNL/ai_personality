from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from datasets import load_dataset, concatenate_datasets

# Load the Previously Fine-Tuned Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained("./opt-fine-tuned")
model = AutoModelForCausalLM.from_pretrained("./opt-fine-tuned")

# Load and Prepare Datasets
print("Loading datasets...")

# Load datasets and map the "text" field
tweets = load_dataset("tweet_eval", "sentiment")["train"].map(
    lambda examples: {"text": examples["text"]}  # Ensure compatibility
)
bluesky = load_dataset("alpindale/two-million-bluesky-posts")["train"]
persona_chat = load_dataset("personachat")["train"].map(
    lambda examples: {"text": " ".join(examples["utterances"])}  # Join utterances into a single text
)

# Combine datasets
print("Combining datasets...")
combined_dataset = concatenate_datasets([tweets, bluesky, persona_chat])

# Tokenize the Dataset
print("Tokenizing the dataset...")
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

tokenized_dataset = combined_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)

# Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False  # No masked language modeling for causal models
)

# Define Incremental Fine-Tuning Arguments
training_args = TrainingArguments(
    output_dir="./opt-fine-tuned-2",  # Directory for the updated model
    num_train_epochs=1,              # Use fewer epochs for incremental updates
    per_device_train_batch_size=16,  # Adjust based on hardware
    eval_strategy="epoch",           # Evaluate at the end of each epoch
    save_strategy="epoch",
    save_total_limit=2,
    logging_dir="./logs-2",
    learning_rate=1e-5,              # Lower learning rate for fine-tuning
    warmup_steps=100,                # Shorter warmup for incremental training
    weight_decay=0.01,               # Regularization
    fp16=True,                       # Mixed precision
)

# Train the Model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
)

print("Starting incremental fine-tuning with combined datasets...")
trainer.train()

# Save the Updated Model
model.save_pretrained("./opt-fine-tuned-2")
tokenizer.save_pretrained("./opt-fine-tuned-2")

print("Incremental fine-tuning completed!")


