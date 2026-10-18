## ADDED Requirements

### Requirement: CPU-only configuration inspection

The system SHALL expose --print-config and --export-config to validate and inspect the effective recipe and typed training configuration before distributed initialization. These modes MUST NOT initialize CUDA, process groups or W&B, construct models, read dataset bodies or weights, or download assets. They SHALL allow reading explicitly required checkpoint configuration metadata. Only explicit export destinations MAY be written; normal result directories MUST NOT be created by inspection.

#### Scenario: Inspect without a GPU or dataset
- **WHEN** an explicit-step Stage1 recipe is inspected on a host without CUDA or local datasets
- **THEN** the effective recipe and training configuration are printed without data or model access

#### Scenario: Inspect Stage2 using metadata
- **WHEN** a Stage2 recipe is inspected with a local checkpoint config fixture but no weights
- **THEN** model and video settings can be validated without opening weight files

### Requirement: Clearly bounded validation

The system SHALL validate recipe types and cross-field constraints before training. Checks depending on actual distributed world size SHALL run when topology becomes available. Inspection without a known topology MUST identify these checks as pending and MUST NOT claim full runtime readiness.

#### Scenario: Unknown world size
- **WHEN** a recipe is inspected outside torchrun without an explicit topology
- **THEN** static validation succeeds where valid and distributed batch compatibility is reported as unverified

### Requirement: Reproducible run records

For new training runs the system SHALL retain the existing resolved config.yaml and save input and composed recipes, source file order, explicit overrides, Git revision and dirty state, uv.lock hash, Python and key dependency versions, command and launch topology. Records MUST NOT capture arbitrary environment variables or credentials. Implicit overwriting of existing experiment outputs MUST be rejected.

#### Scenario: New experiment starts
- **WHEN** a validated recipe starts a new training run
- **THEN** the output contains the effective trainer configuration and experiment provenance linked to that run

#### Scenario: Output already exists
- **WHEN** a new run targets an existing experiment output without explicit reuse authorization
- **THEN** the launcher refuses to overwrite its configuration and manifest

### Requirement: Preserve original provenance on resume

The system SHALL preserve the original run record when resuming and append a separate resume record including the current effective settings. Existing trainer resume compatibility checks SHALL remain effective.

#### Scenario: Resume with changed optimizer settings
- **WHEN** an existing run is resumed with an explicit learning-rate override
- **THEN** the original manifest remains intact, the override appears in a resume record, and existing trainer configuration comparisons still execute
