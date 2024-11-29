from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the fine-tuned model and tokenizer
model = AutoModelForCausalLM.from_pretrained("./opt-fine-tuned")
tokenizer = AutoTokenizer.from_pretrained("./opt-fine-tuned")

# Test prompt
prompt = "Say something about today condition."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# Generate response
output = model.generate(
    input_ids,
    max_length=100,
    temperature=0.9,  # Sampling temperature
    top_k=50,         # Top-k sampling
    top_p=0.9,        # Nucleus sampling
    do_sample=True,
)

response = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Response:", response)


