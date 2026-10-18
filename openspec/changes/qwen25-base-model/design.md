## Context

Qwen3 converter は QKV bias なし、Q/K norm ありを前提にする。一方 Qwen2.5-7B-Instruct の公式設定は hidden_size=3584、28層、28 query heads / 4 KV heads、head_dim=128、intermediate_size=18944、vocab_size=152064、rope_theta=1e6、QKV bias あり、Q/K norm なし、weight tying なし。

根拠: [公式 config.json](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json)、[公式 tokenizer 設定](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/tokenizer_config.json)。

## Goals / Non-Goals

設定で Qwen2.5 を選択し、元モデルの計算を保って native LLM に変換する。既存 Qwen3 の既定値や学習条件を変更しない。モデル重みのダウンロードと GPU 検証は今回の対象外。

## Decisions

- `qwen2.5_7b_instruct` を LLMS・HF source・converter の各レジストリに追加する。ファイル名は pretrained_llms/qwen2.5-7b-instruct.pt。
- sequential block の既存 fused QKV を利用し、bias も Q/K/V 順に結合する。MLP は既存 SwiGLU に合わせ up/gate 順、出力 head は独立して保持する。余った source key を拒否する。
- tokenizer は Qwen/Qwen2.5-7B-Instruct。既存 message_format=qwen3 の ChatML 分岐は同じ user/assistant 境界形式なので共有し、学習用 prompt と special-token 処理を変えない。
- Stage1 の model.llm から設定し、Stage2 は該当 Stage1 checkpoint のモデル・tokenizer・projector 次元を継承する。Stage2 で別モデルへ置換しない。
- converter は既存スクリプトを使い、LLM loader が期待する .pt ファイルを生成するため --unsharded を明示する。

## Risks / Trade-offs

- レジストリ追加だけでは不正な変換を見逃す → 小型ランダム HF Qwen2 と native の logits を数値比較する。
- Stage2 LoRA と独立 head の不整合 → 変換済み重みのロード、adapter 初期化、Stage1→Stage2 設定継承を CPU で検証する。
- 実際の 7B 重みや GPU は未検証 → 利用手順に準備コマンドと検証範囲を記載する。
