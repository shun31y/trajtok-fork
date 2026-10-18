## ADDED Requirements

### Requirement: Select Qwen2.5-7B-Instruct

The system SHALL accept model.llm=qwen2.5_7b_instruct for Stage1 and preserve the official architecture, vocabulary, tokenizer, QKV bias and untied output head. Existing model defaults SHALL remain unchanged.

#### Scenario: Compose a Qwen2.5 experiment
- **WHEN** the Qwen2.5 Stage1 recipe is loaded
- **THEN** the effective model selects Qwen/Qwen2.5-7B-Instruct and existing PEFT/training settings remain effective

### Requirement: Convert pretrained weights faithfully

The system SHALL convert Qwen2.5 HF weights including all QKV biases and the independent LM head to native sequential blocks, and MUST reject unused or missing required parameters.

#### Scenario: Numerical parity
- **WHEN** a small HF Qwen2 model is converted and evaluated with the same causal input
- **THEN** native logits match HF logits within floating-point tolerance

### Requirement: Preserve the Stage1 model in Stage2

The system SHALL inherit Qwen2.5 model configuration and tokenizer from its Stage1 checkpoint, retaining compatibility with Stage2 LoRA.

#### Scenario: Stage2 configuration
- **WHEN** Stage2 is configured with a Qwen2.5 Stage1 checkpoint
- **THEN** its model dimensions, tokenizer, QKV bias and untied head are preserved and its configured LoRA adapters can be constructed
