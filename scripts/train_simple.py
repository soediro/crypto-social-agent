#!/usr/bin/env python3
"""
Hello World of Fine-Tuning - Simplest possible version
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset


def main():
    print("🎯 Hello World Fine-Tuning Starting...")

    # 1. Load model
    model_name = "facebook/opt-350m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

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

    # 4. Train with minimal settings
    training_args = TrainingArguments(
        output_dir="./hello_world_model",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        logging_steps=1,
        save_total_limit=1,
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
    inputs = tokenizer(test_input, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=15)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Test: {test_input} -> {result}")


if __name__ == "__main__":
    main()