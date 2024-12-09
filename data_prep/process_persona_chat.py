import json

# Function to preprocess the dataset
def preprocess_personachat(input_file, output_file):
    print(f"Loading dataset from {input_file}...")
    with open(input_file, "r") as f:
        dataset = json.load(f)

    print("Preprocessing the Persona-Chat dataset...")
    conversations = []
    for example in dataset:
        history = []
        for message in example['messages']:
            # Check the key for message text
            if 'text' in message:
                history.append(message['text'])
            elif 'content' in message:
                history.append(message['content'])
            else:
                raise KeyError("Message does not contain 'text' or 'content'.")
        # Combine the history into a single conversation string
        conversations.append({"text": " ".join(history)})

    print(f"Processed {len(conversations)} conversations.")
    
    # Save the preprocessed data to a JSON file
    print(f"Saving preprocessed dataset to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(conversations, f, indent=4)
    print(f"Dataset saved to {output_file}.")

# Fix line-delimited JSON to valid JSON array
def fix_json_structure(input_file, output_file):
    with open(input_file, "r") as f:
        lines = f.readlines()  # Read each line as a separate JSON object

    # Parse each line into a JSON object
    data = [json.loads(line) for line in lines]

    # Write the JSON array to the output file
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)



if __name__ == "__main__":
    # Run the function
    fix_json_structure("personachat_train.json", "personachat_train_fixed.json")
    fix_json_structure("personachat_test.json", "personachat_test_fixed.json")

    # Preprocess the training dataset
    preprocess_personachat("personachat_train_fixed.json", "personachat_train_preprocessed.json")
    
    # Preprocess the test dataset
    preprocess_personachat("personachat_test_fixed.json", "personachat_test_preprocessed.json")
