## Why

Qwen2.5-7B-Instruct を TrajVLM の base LLM として設定可能にする。Qwen3 と構造が異なるため、設定レジストリだけでなく重み変換も追加する。

## What Changes

- model.llm=qwen2.5_7b_instruct を追加する。
- 公式の層数・次元・QKV bias・RMSNorm・RoPE・独立 LM head・tokenizer を設定する。
- 専用 HF→Molmo converter と Stage1/Stage2 派生レシピ、利用手順と CPU テストを追加する。

## Capabilities

### New Capabilities

- `qwen25-base-model`: Qwen2.5 の選択、重み変換と checkpoint による Stage2 継承。

### Modified Capabilities

なし。

## Impact

LLM レジストリ、pretrained-model 準備スクリプト、サンプル YAML。既存モデルの既定値、PEFT 設定、学習条件は維持する。7B 重みの実取得と GPU 訓練は行わない。
