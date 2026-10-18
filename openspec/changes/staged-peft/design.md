## Context

既存 connector グループには segmenter、query_proj、新規 embedding も含まれるため、ft_connector だけでは指定範囲に限定できない。LLM は Hugging Face モデルではなく Molmo 独自実装で、既定の Qwen3 設定は結合 QKV の att_proj を持つ sequential block を使用する。FSDP は meta 構築後に to_empty で実体化し、Stage2 は Stage1 の full-model checkpoint で初期化する。

## Goals / Non-Goals

Stage1: cross_attn と image_projector のみ。Stage2: 同じ2モジュールと LLM attention の LoRA A/B のみ。query_proj は独立した橋渡し Linear として凍結する。データ、lr、scheduler、batch、期間、FSDP、compile、評価条件は変更しない。モデル準備・GPU 訓練・adapter 専用 export は今回の対象外。

## Decisions

- PeftConfig を recipe と TrainConfig に、LoRAConfig を LlmConfig に保存する。モデル再構築時も adapter が復元される。
- カスタム LLM に対応する nn.Linear サブクラスで LoRA を実装する。元 weight/bias の名前とオブジェクトを保持し、lora_A / lora_B のみ追加する。rank 16、alpha 32、dropout 0、att_proj / attn_out を既定にする。
- A は Kaiming uniform、B はゼロ初期化し、初回出力は base と一致する。meta からの実体化後にも adapter を明示初期化する。
- generic ft フラグの処理後にモデル全体を凍結し、cross_attn / image_projector と必要な adapter だけを再度有効にする。optimizer は既存 requires_grad フィルタを使い、LoRA は既存 LLM グループの lr を使用する。
- Stage1→Stage2 新規実行では adapter 欠落だけを許可し、base 重みの欠落は拒否する。sharded / unsharded の両形式を扱う。PyTorch の broadcast + strict=False が missing_keys を報告しない場合に備え、unsharded では明示的にキーを検証する。
- Stage2 再開は同じ PEFT / LoRA 構成を要求し、全 adapter と optimizer を復元する。既存の full-model checkpoint 形式を保つ。
- 標準 YAML のみ既定を更新し、peft: null は従来の全体学習へ戻す。旧 CLI の未指定時の挙動を維持する。

## Risks / Trade-offs

- FSDP 実体化で初期値が失われる → checkpoint 初期化前に LoRA を再初期化する。
- adapter を欠落許可することで壊れた checkpoint を見逃す → 新規 Stage2 の adapter キーだけを許可し、再開は厳密にする。
- GPU は利用できない → CPU の実 LLM block 構築、凍結・勾配・更新、Gloo の checkpoint 保存/読込/optimizer 再開を検証する。GPU FSDP の実行は未検証と明示する。

参考: [LoRA 論文](https://arxiv.org/abs/2106.09685)、[PyTorch checkpoint API](https://docs.pytorch.org/docs/main/distributed/checkpoint.html)。
