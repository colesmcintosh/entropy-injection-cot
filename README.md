# Entropy Injection Chain of Thought

An approach to improve language model reasoning by dynamically injecting Chain of Thought (CoT) prompts based on entropy measurements.

## Overview

This project implements a technique that monitors the entropy of a language model's token predictions during generation. When the model encounters high uncertainty (high entropy), the system automatically injects a Chain of Thought prompt to encourage step-by-step reasoning.

## How It Works

1. **Entropy Monitoring**: During text generation, the system calculates the entropy of the model's next token distribution at each step.
2. **Threshold Detection**: When entropy exceeds a predefined threshold (default: 4.0), indicating model uncertainty.
3. **CoT Injection**: The system injects a Chain of Thought prompt (e.g., "To determine the answer, let's breakdown the problem step by step, then provide a final answer.").
4. **Improved Reasoning**: This encourages the model to work through the problem methodically, leading to more structured reasoning.

## Implementation Details

- Uses Hugging Face Transformers library for model loading and tokenization
- Supports 8-bit quantization via BitsAndBytes for memory efficiency
- Implements nucleus sampling (top-p) for token generation
- Includes cooldown period after CoT injection to prevent multiple injections in quick succession
- Provides detailed logging of entropy values and generated tokens

## Example

When given the prompt "Is 9.9 greater than 9.11?", the model initially responds with uncertainty. When entropy spikes above the threshold, the system injects a CoT prompt, leading the model to break down the comparison step-by-step:

```
Step 112: Entropy 4.9825 exceeds threshold. Injecting CoT prompt.
```

The model then proceeds with a structured approach:
1. Determining the comparison needed
2. Comparing the numbers by expanding decimal places
3. Reaching a conclusion based on proper decimal comparison

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Load model and tokenizer
model_name = 'meta-llama/Llama-3.2-1B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name, token=YOUR_HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=YOUR_HF_TOKEN,
    device_map='auto',
    quantization_config=BitsAndBytesConfig(load_in_8bit=True)
)

# Run entropy-based CoT injection
prompt = "Is 9.9 greater than 9.11?"
result, entropies, tokens = entropy_based_cot_injection_with_logging(
    prompt, 
    entropy_threshold=4.0,
    max_length=150,
    max_cot_injections=1,
    cooldown_steps=5
)
print(result)
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- BitsAndBytes (for quantization)
- Accelerate

## License

This project is licensed under the terms of the MIT license.
