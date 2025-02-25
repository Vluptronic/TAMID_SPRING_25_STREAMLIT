from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import numpy as np

from typing import Union

class Seq2SeqModel:

    def __init__(self):

        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", use_fast = True)
        self.memory = np.array([]) # ts (this) memory is kinda butt but works

    def generate(self, q: str) -> str:

        context = self.get_memories()

        inputs = self.tokenizer(f"{context}\n{q}", return_tensors = "pt")
        outputs = self.model.generate(**inputs)
        res = self.tokenizer.batch_decode(outputs, skip_special_tokens = True)[0]

        with torch.no_grad():
            embeddings = self.model.get_encoder()(**inputs).last_hidden_state
            np.append(self.memory, embeddings)
        
        return res

    def get_memories(self) -> str:

        res = ""
        
        for i in range(len(self.memory)):
            res += self.generate(self.memory[i]) + "\n"
        
        return res