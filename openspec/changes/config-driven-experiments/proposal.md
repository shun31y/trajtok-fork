## Why

TrajVLM は OmegaConf を利用しているが、実験条件が Python のデフォルト値・shell・CLI に分散している。Stage1 PixMoCap / Stage2 VideoQA の実験条件を YAML と Git で比較し、訓練を始めずに設定を確認できるようにする。

## What Changes

- YAML を入力とする共通の実験ランチャーと型付き設定を追加する。
- 基底 YAML → 実験 YAML → 明示 CLI override の順序で設定を合成する。
- 設定の合成後にモデル・データ・評価・学習期間を一貫して生成する。
- GPU・distributed 初期化より前に設定の検証・表示・書き出しを行えるようにする。
- Stage1 PixMoCap-only / Stage2 VideoQA-only の基準レシピと派生例を用意する。
- 入力レシピ・解決済み TrainConfig・Git/uv 情報を実行単位で保存する。
- 既存 CLI と shell の呼び出し互換性を維持する。

## Capabilities

### New Capabilities

- `experiment-configuration`: YAML 合成、CLI、型・整合性検証、ステージ別構成生成。
- `experiment-reproducibility`: GPU 不要の設定確認、実行記録、再開時の設定保持。

### Modified Capabilities

なし（既存の OpenSpec specs は未作成）。

## Impact

主な対象は `trajvlm/launch_scripts/trajvlm_{pretrain,sft}.py`、新規共通設定モジュール・ランチャー、`trajvlm/configs/`、利用手順とテスト。既存 `TrainConfig` と trainer の保存処理を活用する。Hydra などの追加フレームワーク、GPU カーネル整備、モデル移植、データ取得の変更、実際の訓練実行はこの変更の対象外。
