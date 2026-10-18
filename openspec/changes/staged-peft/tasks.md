## 1. Configuration and scope

- [x] 1.1 Add typed PEFT/LoRA configuration and stage-specific YAML defaults, preserving unrelated values.
- [x] 1.2 Freeze the model and enable only TrajPerceiver/Projector plus Stage2 LLM adapters; preserve optimizer groups.

## 2. LoRA and checkpoint lifecycle

- [x] 2.1 Implement native LoRA for custom LLM blocks with preserved weight keys, zero delta and meta-safe initialization.
- [x] 2.2 Support Stage1→Stage2 checkpoint loading and strict Stage2 resume for sharded/unsharded formats.

## 3. Verification and documentation

- [x] 3.1 Test exact trainable scopes, actual gradients/updates, initialization, optimizer membership, checkpoint transition and resume on CPU.
- [x] 3.2 Verify unrelated recipe settings and existing config compatibility tests; document configuration and limitations.
