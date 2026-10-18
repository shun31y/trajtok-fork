## Why

Stage1 は TrajPerceiver と Projector の学習に限定し、Stage2 は同じ2モジュールに加えて凍結 LLM の LoRA を学習する。データ・学習率・batch・期間など、既存の実験条件は維持する。

## What Changes

- YAML の PEFT 設定を追加し、Stage1 は connector_only、Stage2 は connector_lora を既定にする。
- cross_attn / image_projector のみを通常学習し、segmenter / ViT / query_proj / embedding / LM head を凍結する。
- カスタム LLM の attention Linear に native LoRA を追加し、元の weight/bias キーを保持する。
- Stage1→Stage2 の初回読込だけ adapter の欠落を許可し、再開は adapter を含む重みと optimizer を復元する。

## Capabilities

### New Capabilities

- `staged-finetuning`: 学習対象制限、LLM LoRA、初期化・checkpoint と再開。

### Modified Capabilities

なし。既存実験設定の合成と記録形式はそのまま利用する。

## Impact

recipe / TrainConfig / LlmConfig、trainer の凍結処理、既存 checkpoint 読込。新しい外部依存は追加しない。既存 CLI の PEFT 未指定時の挙動と、PEFT 以外のレシピ値は保持する。
