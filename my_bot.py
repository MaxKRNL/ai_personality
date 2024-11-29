# my_bot.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class MyBot:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('./fine-tuned-model')
        self.model = AutoModelForCausalLM.from_pretrained('./fine-tuned-model')

    def generate_response(self, prompt, **kwargs):
        inputs = self.tokenizer.encode(prompt + self.tokenizer.eos_token, return_tensors='pt')
        outputs = self.model.generate(
            inputs,
            max_length=kwargs.get('max_length', 150),
            pad_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=kwargs.get('no_repeat_ngram_size', 2),
            do_sample=kwargs.get('do_sample', True),
            top_k=kwargs.get('top_k', 50),
            top_p=kwargs.get('top_p', 0.95),
            temperature=kwargs.get('temperature', 0.7),
            # Additional parameters can be added here
        )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Return only the generated part
        return text[len(prompt):].strip()

