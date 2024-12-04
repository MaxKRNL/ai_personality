from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the fine-tuned model and tokenizer
model = AutoModelForCausalLM.from_pretrained("./opt-fine-tuned-2")
tokenizer = AutoTokenizer.from_pretrained("./opt-fine-tuned-2")

print("Welcome to the KRNL Chatbot!")
print("Type your query below and the bot will respond. Type 'exit' to quit.")

while True:
    # Get user input
    user_input = input("You: ")
    
    # Exit the loop if the user types 'exit'
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    
    # Format the prompt
    prompt = f"User: {user_input}\nBot:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Generate response
    output = model.generate(
        input_ids,
        max_length=100,
        temperature=0.6,  # Sampling temperature
        top_k=50,         # Top-k sampling
        top_p=0.9,        # Nucleus sampling
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,  # Ensure response ends with EOS token
        no_repeat_ngram_size=2,  # Prevents repetition
    )

    # Decode the response
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract only the bot's response (remove repeated prompt)
    # Find the position of "Bot:" in the generated response
    bot_response_start = response.find("Bot:") + len("Bot:")
    bot_response = response[bot_response_start:].strip()

    print(f"KRNLBot: {bot_response}")




