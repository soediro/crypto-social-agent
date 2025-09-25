#!/usr/bin/env python3
"""
Hello World of Fine-Tuning - Simplest possible version
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import torch


def main():
    print("🎯 Hello World Fine-Tuning Starting...")

    # Check CUDA availability and setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Using device: {device}")
    if torch.cuda.is_available():
        print(f"📱 CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"🧠 CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 1. Load model
    model_name = "facebook/opt-350m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32)
    model = model.to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Minimal training data
    texts = [
        "Crypto market sentiment turns bullish when",
        "Blockchain technology enables secure",
        "DeFi platforms allow users to",
        "NFT marketplaces have revolutionized",
        "Smart contracts automatically execute when"
    ]

    dataset = Dataset.from_dict({"text": texts})

    # 3. Tokenize
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=64)

    tokenized_ds = dataset.map(tokenize_fn, batched=True)

    # 4. Train with minimal settings optimized for GTX 1050 Mobile
    training_args = TrainingArguments(
        output_dir="./hello_world_model",
        num_train_epochs=2,
        per_device_train_batch_size=1,  # Reduced for GTX 1050 Mobile memory
        gradient_accumulation_steps=2,  # Compensate for smaller batch size
        logging_steps=1,
        save_total_limit=1,
        fp16=device.type == "cuda",  # Enable half precision if using CUDA
        dataloader_pin_memory=False,  # Reduce memory usage
        report_to=[],  # Disable wandb/tensorboard logging
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
    )

    # 5. Train and save
    print("🏋️ Training...")
    trainer.train()
    trainer.save_model()
    print("✅ Training complete! Model saved.")

    # 6. Quick test
    test_input = "Crypto market sentiment turns bullish when"
    inputs = tokenizer(test_input, return_tensors="pt").to(device)
    outputs = model.generate(inputs.input_ids, max_length=15, pad_token_id=tokenizer.eos_token_id)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Test: {test_input} -> {result}")


if __name__ == "__main__":
    main()