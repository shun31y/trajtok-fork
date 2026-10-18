## 1. Model and weights

- [x] 1.1 Verify official model/tokenizer metadata and add the Qwen2.5 registry entry.
- [x] 1.2 Add QKV-bias-aware conversion with a separate LM head and register the HF source.

## 2. Configuration and verification

- [x] 2.1 Add Stage1/Stage2 recipes and document model selection, weight preparation and inheritance.
- [x] 2.2 Verify HF/native logits parity, pretrained weight loading with LoRA, invalid weights, and Stage2 configuration inheritance on CPU.
- [x] 2.3 Run existing config and PEFT regression suites and OpenSpec/uv checks.
