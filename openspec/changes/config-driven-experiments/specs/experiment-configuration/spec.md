## ADDED Requirements

### Requirement: Deterministic typed recipe composition

The system SHALL load a typed, versioned experiment recipe using OmegaConf. It SHALL apply base files in listed order, then the experiment file, then explicit CLI overrides in order. Mappings SHALL merge recursively and lists SHALL replace. Unknown keys, invalid types, missing required values, unsupported schema versions and cyclic inheritance MUST fail before training initialization.

#### Scenario: Override learning rate
- **WHEN** a base, experiment file and CLI override each specify a connector learning rate
- **THEN** the CLI value is effective and its source is recorded

#### Scenario: Invalid or cyclic input
- **WHEN** a recipe contains a misspelled key or cyclic extends chain
- **THEN** the command fails with the offending key or file chain before initializing distributed execution

### Requirement: Portable file and environment resolution

The system SHALL resolve extends relative to each declaring YAML, normalize recipe-relative local paths, preserve URIs, and resolve declared environment interpolation. CLI-relative paths SHALL use the top-level experiment directory. Shared recipes MUST NOT embed host-specific NAS or author paths. Missing required environment variables MUST produce an actionable error.

#### Scenario: Launch from another working directory
- **WHEN** the same absolute experiment file is loaded from two working directories with the same environment
- **THEN** effective input and output paths are identical

### Requirement: Consistent stage configuration generation

The system SHALL compose the recipe before generating model, preprocessing, data, evaluation and duration settings. It SHALL reject conflicting structural trainer_overrides and validate dependent fields together. New recipes SHALL use explicit max_steps or epochs with an explicit train_examples count; these modes MUST be mutually exclusive.

#### Scenario: Change global batch size
- **WHEN** an epochs-based recipe changes global batch size using CLI
- **THEN** duration is recalculated using the final batch size and stop_at is consistent

#### Scenario: Inconsistent model override
- **WHEN** a trainer override attempts to independently change generated trajectory counts
- **THEN** validation rejects it and points to the public recipe field

### Requirement: Explicit Stage1 and Stage2 data selection

The system SHALL provide a PixMoCap-only Stage1 recipe and a VideoQA-only Stage2 recipe using the authoritative VIDEO_ACADEMIC_DATASETS names. Data mixture, evaluation selection and W&B enablement SHALL be explicit. Stage2 SHALL inherit the model from checkpoint configuration metadata and apply recipe video parameters without loading weights during configuration construction.

#### Scenario: VideoQA-only experiment
- **WHEN** the Stage2 VideoQA recipe is composed with evaluation disabled
- **THEN** the generated training mixture contains only the selected video datasets and no image, tracking or NLP evaluation tasks are added

#### Scenario: Missing checkpoint metadata
- **WHEN** Stage2 is configured without its required checkpoint config.yaml
- **THEN** configuration fails with the required path before any model weights are loaded

### Requirement: Legacy launcher compatibility

Existing CLI and shell launchers SHALL remain usable with their previous argument meanings and defaults. Shared builders SHALL preserve legacy configurations for equivalent inputs. CLI defaults MUST NOT override explicitly supplied YAML recipe values in the new launcher.

#### Scenario: Existing SFT shell invocation
- **WHEN** the existing sft.sh is invoked with its original arguments
- **THEN** its default 64 frames and 8 frames_per_clip remain effective
