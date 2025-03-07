import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# 1. Load base model and tokenizer
print("Loading base model and tokenizer...")
base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

# 2. Load LoRA-adapted model
print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(base_model, "my_lora_finetuned")
model.eval()

class Seq2SeqModel:
    def __init__(self):
        print("Initializing Seq2SeqModel...")
        self.model = model
        self.tokenizer = tokenizer
        self.memory = []  # Use list instead of NumPy array
        self.history = []  # ✅ Add history attribute to store user inputs and responses

    def generate(self, q: str) -> str:
    #Generate response based only on the user's query.

        print(f"Generating response for: {q}")

        # Tokenize and move input to model device
        inputs = self.tokenizer(q, return_tensors="pt").to(self.model.device)

        # Generate response with proper settings
        outputs = self.model.generate(
            **inputs, 
            max_length=100,  # Keep it concise
            do_sample=True,  # More diverse outputs
            top_k=50,        # Avoid deterministic responses
            top_p=0.9        # Nucleus sampling
        )

        # Decode the response properly
        res = self.tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        return res if res.strip() else "[No Response]"

class App:
    def __init__(self):
        print("Initializing App...")
        self.model = Seq2SeqModel()

    def run(self):
        print("Starting interactive mode...")
        while True:
            q = input("Enter a question (or type 'exit' to quit): ")  # ✅ Ensuring input is called
            if q.lower() == "exit":
                print("Exiting...")
                break
            res = self.model.generate(q)
            print("AI:", res)

if __name__ == "__main__":
    print("Launching app...")
    my_app = App()
    my_app.run()
