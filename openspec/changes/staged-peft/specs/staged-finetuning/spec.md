## ADDED Requirements

### Requirement: Stage-specific trainable parameters

With PEFT enabled, the system SHALL train only vision_backbone.cross_attn and vision_backbone.image_projector in Stage1. Stage2 SHALL additionally train only LLM LoRA A/B parameters. All other parameters, including query_proj, segmenter, ViT, embeddings and LM head, MUST remain frozen and absent from optimizer groups.

#### Scenario: Optimizer update in each stage
- **WHEN** a backward pass and optimizer step run with each PEFT mode
- **THEN** gradients and updates occur only in the allowed trainable scopes

### Requirement: Preserve unrelated experiment settings

The system SHALL retain all existing recipe settings except the PEFT policy, corresponding fine-tuning flags and LoRA model architecture metadata. It SHALL leave PEFT-unset legacy CLI behavior unchanged.

#### Scenario: Compare PEFT and legacy recipes
- **WHEN** the same recipe is built with PEFT enabled and with peft=null
- **THEN** all data, optimizer, scheduler, duration, batch, compilation and evaluation settings are identical

### Requirement: LoRA configuration and initialization

The system SHALL expose rank, alpha, dropout and target modules through typed configuration, retain original Linear weight keys and initialize the adapter delta to zero, including after meta materialization. Defaults SHALL be rank 16, alpha 32, dropout 0 and fused attention input/output projections. Invalid settings or unmatched targets MUST fail.

#### Scenario: Initial adapter output
- **WHEN** a base Linear is adapted and initialized
- **THEN** its output is identical to the base output and base weight/bias names remain unchanged

### Requirement: Stage transition and resume

The system SHALL allow absent adapter keys only when initializing a new Stage2 run from a non-LoRA checkpoint, and MUST reject other missing keys. Resume SHALL restore all adapter and optimizer state and reject incompatible PEFT configuration. Sharded and unsharded checkpoints SHALL both support the transition.

#### Scenario: Stage1 initialization and Stage2 resume
- **WHEN** Stage2 initializes from Stage1 and later saves and resumes
- **THEN** Stage1 weights are retained, new adapters start with zero delta, and Stage2 adapter and optimizer state are restored

#### Scenario: Corrupt checkpoint
- **WHEN** a required base weight is missing or an adapter is missing during resume
- **THEN** loading fails instead of silently using uninitialized parameters
