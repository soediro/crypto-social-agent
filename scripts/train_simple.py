#!/usr/bin/env python3
"""
Hello World of Fine-Tuning - Simplest possible version
"""

import os
# Force CPU-only operation for GTX 1050 compatibility
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import torch


def main():
    print("🎯 Hello World Fine-Tuning Starting...")

    # Force CPU for GTX 1050 compatibility
    device = torch.device("cpu")
    print(f"🔥 Using device: {device} (forced for GTX 1050 compatibility)")
    if torch.cuda.is_available():
        print(f"📱 CUDA Available but disabled: {torch.cuda.get_device_name(0)}")
        print(f"🧠 CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 1. Load model
    model_name = "facebook/opt-350m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,  # Use float32 for GTX 1050 compatibility
        use_safetensors=True
    )
    model = model.to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. EMA Trading Strategy Training Data
    texts = [
        # Short-Term EMA Cross Strategies
        "when the 5 EMA crosses above both 8 EMA and 13 EMA with all three EMAs trending upward, enter a long position for bullish momentum in short-term trading",
        "when the 5 EMA crosses below both 8 EMA and 13 EMA with all three EMAs trending downward, enter a short position for bearish momentum in short-term trading",
        "when the 5 EMA crosses above the 8 EMA in volatile markets, enter a long position but set tight stop losses due to high responsiveness",
        "when the 5 EMA crosses below the 8 EMA in volatile markets, enter a short position but set tight stop losses due to high responsiveness",
        "when the 9 EMA crosses above the 21 EMA and price is above both EMAs, enter a long position for medium-term bullish trend",
        "when the 9 EMA crosses below the 21 EMA and price is below both EMAs, enter a short position for medium-term bearish trend",

        # Medium-Term EMA Cross Strategies
        "when the 12 EMA crosses above the 26 EMA with both EMAs sloping upward, enter a long position for medium-term trend following",
        "when the 12 EMA crosses below the 26 EMA with both EMAs sloping downward, enter a short position for medium-term trend following",
        "when the fast EMA crosses above the slow EMA and price breaks above recent highs, enter a long position with momentum confirmation",
        "when the fast EMA crosses below the slow EMA and price breaks below recent lows, enter a short position with momentum confirmation",

        # Long-Term EMA Cross Strategies
        "when the 50 EMA crosses above the 200 EMA creating a golden cross pattern, enter a long position for strong bullish trend confirmation",
        "when the 50 EMA crosses below the 200 EMA creating a death cross pattern, enter a short position for strong bearish trend confirmation",
        "when the 30 EMA crosses above the 200 EMA on daily timeframe, enter a long position for long-term trend following",
        "when the 30 EMA crosses below the 200 EMA on daily timeframe, enter a short position for long-term trend following",

        # Gold (XAUUSD) Specific Strategies
        "when the 9 EMA crosses above the 21 EMA in gold trading and price is above both EMAs, enter a long position with ATR-based stop loss",
        "when the 9 EMA crosses below the 21 EMA in gold trading and price is below both EMAs, enter a short position with ATR-based stop loss",
        "when the 12 EMA crosses above the 26 EMA in XAUUSD and Stochastic is oversold, enter a long position for confluence signal",
        "when the 12 EMA crosses below the 26 EMA in XAUUSD and Stochastic is overbought, enter a short position for confluence signal",
        "when the 8 EMA, 13 EMA, and 21 EMA are all aligned upward in gold trading, enter a long position for strong trend confirmation",
        "when the 8 EMA, 13 EMA, and 21 EMA are all aligned downward in gold trading, enter a short position for strong trend confirmation",

        # Entry Confirmation Rules
        "when a short-term EMA crosses above a long-term EMA and price is trading above both EMAs, confirm bullish bias before entering long",
        "when a short-term EMA crosses below a long-term EMA and price is trading below both EMAs, confirm bearish bias before entering short",
        "when EMA crossover occurs with increasing volume, the signal strength is higher for trade entry",
        "when EMA crossover occurs with decreasing volume, wait for volume confirmation before entering trade",
        "when price pulls back to the EMA line after a crossover and finds support, enter in the direction of the crossover",
        "when price pulls back to the EMA line after a crossover and finds resistance, enter in the direction of the crossover",

        # Exit Strategy Rules
        "when the fast EMA crosses back below the slow EMA after a bullish crossover, exit long positions or consider profit taking",
        "when the fast EMA crosses back above the slow EMA after a bearish crossover, exit short positions or consider profit taking",
        "when price moves significantly away from the EMA after entry, trail stop loss using the EMA as dynamic support/resistance",
        "when EMA alignment breaks down with EMAs starting to converge, consider reducing position size or exiting",

        # Risk Management Rules
        "when entering on EMA crossover signals, place stop loss below the most recent swing low for long positions",
        "when entering on EMA crossover signals, place stop loss above the most recent swing high for short positions",
        "when using short-term EMA crosses (5-8), set tighter stop losses due to increased signal frequency and noise",
        "when using long-term EMA crosses (50-200), allow wider stop losses due to slower signal generation but stronger trends",
        "when EMA crossover fails to generate follow-through price movement, exit quickly to minimize losses",

        # Timeframe-Specific Rules
        "when trading on 5-minute charts, use 5, 9, or 20-period EMAs for scalping strategies with quick entries and exits",
        "when trading on 1-hour charts, use 12, 26, or 50-period EMAs for day trading strategies with moderate hold times",
        "when trading on daily charts, use 50, 100, or 200-period EMAs for swing trading strategies with longer hold times",
        "when switching timeframes, adjust EMA periods proportionally to maintain similar sensitivity to price movements",

        # Market Condition Rules
        "when markets are trending strongly, EMA crossover strategies work best with high probability of follow-through",
        "when markets are ranging or choppy, EMA crossover signals generate more false signals requiring additional confirmation",
        "when volatility is high, use shorter EMA periods to capture quick price movements but expect more whipsaws",
        "when volatility is low, use longer EMA periods to filter noise and focus on more significant trend changes",

        # Forex-Specific Applications
        "when trading major forex pairs with EMA crosses, consider economic news and central bank policies for additional context",
        "when the price crosses above the EMA cluster in trending forex markets, enter long positions with the trend",
        "when the price crosses below the EMA cluster in trending forex markets, enter short positions with the trend",
        "when multiple currency pairs show similar EMA crossover patterns, it may indicate broader market sentiment shifts"
    ]

    dataset = Dataset.from_dict({"text": texts})

    # 3. Tokenize with labels for causal LM
    def tokenize_fn(examples):
        tokenized = tokenizer(examples["text"], truncation=True, padding=True, max_length=256)  # Increased for longer EMA strategies
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_ds = dataset.map(tokenize_fn, batched=True)

    # 4. Train with minimal settings optimized for GTX 1050 Mobile
    training_args = TrainingArguments(
        output_dir="./hello_world_model",
        num_train_epochs=2,
        per_device_train_batch_size=1,  # Reduced for GTX 1050 Mobile memory
        gradient_accumulation_steps=2,  # Compensate for smaller batch size
        logging_steps=1,
        save_total_limit=1,
        fp16=False,  # Disable FP16 for GTX 1050 compatibility
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
    tokenizer.save_pretrained("./hello_world_model")  # Save tokenizer explicitly
    print("✅ Training complete! Model and tokenizer saved.")

    # 6. Quick test
    test_input = "when the 9 EMA crosses above the 21 EMA"
    inputs = tokenizer(test_input, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=50,  # Increased for longer trading strategy completions
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Test: {test_input} -> {result}")


if __name__ == "__main__":
    main()