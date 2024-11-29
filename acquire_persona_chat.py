from datasets import load_dataset
import json

# Function to download and save the dataset
def download_personachat():
    print("Downloading the Persona-Chat dataset...")
    dataset = load_dataset("HuggingFaceTB/smol-smoltalk")
    print(dataset)

    # Save the train and test splits as JSON files
    dataset['train'].to_json("personachat_train.json")
    dataset['test'].to_json("personachat_test.json")
    print("Dataset downloaded and saved as JSON files.")

if __name__ == "__main__":
    download_personachat()

