from datasets import load_dataset

# Load dataset from Hugging Face
dataset = load_dataset("jimwang99/TinyStoriesV2-Tokenized")
dataset.cleanup_cache_files()
# Select the tokenized input-output pairs
def preprocess_function(example):
    max_length = 176
    
    input_ids = example["sentencepiece_tok32k"][:max_length]
    input_ids += [0] * (max_length - len(input_ids))

    attention_mask = [1]*len(input_ids)
    # Extend with 0's to length 176 if needed
    attention_mask += [0]*(max_length - len(attention_mask))

    labels = example["sentencepiece_tok32k"][:max_length]
    labels += [-100]*(max_length - len(labels))
    
    # SHIFT: create decoder_input_ids by shifting the label tokens right
    decoder_input_ids = [0] + input_ids[:-1]  # Prepend pad token (0), drop last token

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "decoder_input_ids": decoder_input_ids
    }


# Apply strict mapping
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=False,
    remove_columns=dataset["train"].column_names
)


sample = tokenized_dataset["train"][0]
print(f"Input Length: {len(sample['input_ids'])}")
print(f"Label Length: {len(sample['labels'])}")
print(f"Attention Mask Length: {len(sample['attention_mask'])}")

import torch
from transformers import AutoModelForSeq2SeqLM, Trainer, TrainingArguments, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from transformers import DataCollatorForSeq2Seq
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,  # Set this to your actual tokenizer if needed
    padding=True,   # Ensures proper padding
)
# Load base model (FLAN-T5)
base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to("cuda")

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=4,
    lora_alpha=16,
    lora_dropout=0.3,
)

# Wrap model with PEFT (LoRA adapter)
peft_model = get_peft_model(base_model, lora_config)

peft_model.print_trainable_parameters()
# Training setup
training_args = TrainingArguments(
    output_dir="my_lora_finetuned",
    per_device_train_batch_size=8,
    lr_scheduler_type="linear",
    gradient_accumulation_steps=2,
    warmup_ratio=0.05,
    num_train_epochs=1.5,
    logging_steps=100,
    learning_rate=3e-4,
    label_smoothing_factor=0.2,
    label_names=["labels"],
    fp16=False,
    bf16=True,
)
peft_model.config.use_cache = False

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator  # Fixes inconsistent batch sizes
)

print(f"Tokenizer loaded: {tokenizer}")
print(f"Tokenizer pad token: {tokenizer.pad_token}")

# Train the model
trainer.train()

# Save LoRA adapters
peft_model.save_pretrained("my_lora_finetuned")
