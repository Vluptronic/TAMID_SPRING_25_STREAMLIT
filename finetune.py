from datasets import load_dataset

# Load dataset from Hugging Face
dataset = load_dataset("jimwang99/TinyStoriesV2-Tokenized")
# Select the tokenized input-output pairs
def preprocess_function(examples):
    return {
        "input_ids": examples["sentencepiece_tok32k"],  # Use the correct tokenized column
        "labels": examples["sentencepiece_tok32k"]      # Assuming output is also tokenized
    }

# Apply processing to all samples
tokenized_dataset = dataset.map(preprocess_function, batched=True)
from transformers import AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType

# Load base model (FLAN-T5)
base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

# Wrap model with PEFT (LoRA adapter)
peft_model = get_peft_model(base_model, lora_config)

# Training setup
training_args = TrainingArguments(
    output_dir="my_lora_finetuned",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    evaluation_strategy="no",
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],  # Use the tokenized dataset!
)

# Train the model
trainer.train()

# Save LoRA adapters
peft_model.save_pretrained("my_lora_finetuned")
