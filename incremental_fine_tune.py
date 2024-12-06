from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
from accelerate import Accelerator

# Initialize Hugging Face Accelerate
accelerator = Accelerator()
device = accelerator.device
print(f"Using device: {device}")

# Load the Tokenizer and Model
model_name = "facebook/opt-350m"  # Replace with your model path or name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Ensure the model is on GPU if available
model.to(device)

# Load your dataset (e.g., HuggingFaceTB/smoltalk)
print("Loading dataset...")
dataset = load_dataset("HuggingFaceTB/smoltalk", "smol-magpie-ultra")["train"]

# Preprocess the dataset to prepare input-output pairs
print("Preprocessing dataset...")
def preprocess_messages(example):
    messages = example["messages"]
    conversations = []
    for i in range(len(messages) - 1):
        if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
            user_input = messages[i]["content"]
            assistant_response = messages[i + 1]["content"]
            conversations.append({"input": user_input, "output": assistant_response})
    return conversations

# Flatten the dataset and create input-output pairs
processed_data = dataset.map(preprocess_messages, batched=True, remove_columns=dataset.column_names)
processed_data = processed_data.flatten()

# Convert input-output pairs into text for fine-tuning
print("Formatting dataset for training...")
def format_data(example):
    return {
        "text": f"<|user|> {example['input']} <|assistant|> {example['output']}"
    }

formatted_data = processed_data.map(format_data, remove_columns=["input", "output"])

# Split into train and test datasets
split_data = formatted_data.train_test_split(test_size=0.2, seed=42)

# Tokenize the dataset
print("Tokenizing dataset...")
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

tokenized_data = split_data.map(tokenize_function, batched=True, remove_columns=["text"])

# Data collator for causal language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./fine-tuned-model",  # Output directory
    num_train_epochs=2,              # Number of epochs
    per_device_train_batch_size=16,  # Batch size
    per_device_eval_batch_size=16,   # Evaluation batch size
    evaluation_strategy="epoch",     # Evaluate at the end of each epoch
    save_strategy="epoch",           # Save checkpoints at the end of each epoch
    logging_dir="./logs",            # Logging directory
    learning_rate=1e-5,              # Learning rate
    warmup_steps=100,                # Warmup steps
    weight_decay=0.01,               # Regularization
    fp16=True,                       # Mixed precision training
    save_total_limit=2,              # Limit saved checkpoints
    report_to="none",                # No external reporting
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Fine-tune the model
print("Starting training...")
trainer.train()

# Save the fine-tuned model
print("Saving fine-tuned model...")
model.save_pretrained("./fine-tuned-model")
tokenizer.save_pretrained("./fine-tuned-model")

print("Fine-tuning completed!")
