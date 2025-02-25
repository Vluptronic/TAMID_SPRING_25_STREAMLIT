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
model = PeftModel.from_pretrained(base_model, "my_lora_out")

class Seq2SeqModel:
    def __init__(self):
        print("Initializing Seq2SeqModel...")
        self.model = model
        self.tokenizer = tokenizer
        self.memory = []  # Use list instead of NumPy array

    def generate(self, q: str) -> str:
        """Generate response based on query + memory context"""
        print(f"Generating response for: {q}")
        context = self.get_memories()

        # Tokenize the input
        inputs = self.tokenizer(f"{context}\n{q}", return_tensors="pt")
        
        # Generate response
        outputs = self.model.generate(**inputs)
        res = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Store embeddings for memory
        with torch.no_grad():
            embeddings = self.model.get_encoder()(**inputs).last_hidden_state
            self.memory.append(embeddings.cpu().numpy())  # Store as NumPy array

        return res

    def get_memories(self) -> str:
        """Retrieve memory embeddings as text representation"""
        return "\n".join(["[Stored Embedding]" for _ in self.memory])  # Avoid recursion

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
