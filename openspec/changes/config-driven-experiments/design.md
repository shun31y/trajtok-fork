## Context

explore でソースを確認した結果:

| 現在の場所 | 挙動・問題 |
| --- | --- |
| `launch_scripts/trajvlm_pretrain.py:183` | 最初に `prepare_torchrun_environment()`。argparse 前に distributed 初期化する |
| 同 `:246` | duration 未指定時は `len(PixMoCap(...))` を読み、batch size から期間を算出する |
| 同 `:283` 以降 | CLI から caption/pointing/NLP の混合や評価を組み立てる |
| 同 `:414` | 組み立てた後の dotlist 上書き。batch size だけ変えると計算済み duration と食い違いうる |
| `launch_scripts/trajvlm_sft.py:114` | checkpoint の `config.yaml` から TrajVLM モデル設定を読み、動画設定を派生させる |
| 同 `:225` | checkpoint と mixture が位置引数。モデル・データ・評価を生成してから dotlist を適用する |
| `scripts/sft.sh` | 64 frames / 8 frames_per_clip。Python 直起動は 128 / 16 でデフォルトが異なる |
| `olmo/train/trainer_config.py:917` | 型付き YAML 読み込みと model type dispatch がある |
| `olmo/train/run_trainer.py:109` 以降 | dry_run でもモデルを構築する。config-only の代用にはならない |
| 同 `:342` | 新規訓練では出力先 `config.yaml` に解決済み設定を保存する |
| 両 TrajVLM entrypoint 冒頭 | model registry の monkey-patch があり、import 順序に注意が必要 |

`video_academic` はダウンローダーのグループ名であり、それを既存 SFT mixture 名として渡せるとは限らない。`get_training_mixture("molmo2")` は画像・pointing・NLP・tracking 等も含む。取得済みデータと選択した訓練・評価 mixture を一致させる必要がある。

## Goals / Non-Goals

**Goals:** 実験 YAML の比較・派生、決定的な設定合成、CPU 上での設定確認、Stage1/Stage2 の条件記録、既存起動方法との互換性。

**Non-Goals:** Hydra multirun/探索基盤、ジョブスケジューラ、全 Molmo 設定系の置換、モデル重みや DINOv3 の準備、データ取得完了の保証、GPU での学習確認。

## Decisions

### 1. OmegaConf と既存 TrainConfig を使う

Hydra を追加する案、完成済み TrainConfig YAML だけを直接ロードする案、既存 Python builder に実験レシピを渡す案を比較し、最後を採用する。Stage2 の checkpoint モデル継承と Stage1 の派生値算出を既存処理で扱え、実験 YAML を巨大な内部設定から切り離せる。最終生成物は従来の型付き TrainConfig とする。

予定する構造:

```text
trajvlm/configs/base/{pretrain,sft}.yaml
trajvlm/configs/experiments/{stage1_pixmo_cap,stage2_videoqa}.yaml
trajvlm/configs/experiments/stage2_videoqa_lr_ablation.yaml
trajvlm/olmo/experiment_config.py          # recipe schema / composition / validation
trajvlm/launch_scripts/trajvlm_experiment.py # config-only or training dispatch
```

公開 recipe には `schema_version`, `stage`, `run`, `model`, `data`, `training`, `evaluation`, `tracking`, `checkpoint`, `trainer_overrides` を設ける。ステージ固有の項目も structured schema で検証する。`trainer_overrides` は既存 TrainConfig の微調整用。データ構造・モデル構造・batch・期間等、他の設定を派生させる値はこの欄からの変更を拒否し、recipe の該当欄を唯一の入力にする。これにより生成後の上書きによる不整合を避ける。

### 2. 合成・派生・検証を順序づける

```text
extends の基底（列挙順） → 実験 YAML → --set key=value（指定順）
                           ↓
               recipe の型検証・環境変数解決
                           ↓
     stage builder（モデル、データ、評価、期間を一度だけ生成）
                           ↓
        許可した trainer_overrides → 最終 TrainConfig 検証
                           ↓
    --print-config / --export-config  または distributed 初期化 → trainer
```

`extends` は各 YAML ファイルのディレクトリ相対で解決し、循環継承を拒否する。mapping は再帰 merge、list は全置換。`${oc.env:MOLMO_DATA_DIR}` 等を使用し、環境は現在の `data_env.sh` から与える。共有レシピに NAS や作者環境の絶対パスを埋め込まない。通常の相対ローカルファイルパスは、その値を定義する YAML の位置を基準に正規化する（環境変数で与えるルートは絶対パスを要求）。URI は変更しない。`--set` の相対パスはトップレベルの実験 YAML 基準。

CLI は新しいランチャーに `--config FILE`、反復可能な `--set key=value`、`--print-config`、`--export-config FILE` を追加する。以下は実装後の予定コマンド:

```bash
uv run --project trajvlm --locked --extra train python \
  trajvlm/launch_scripts/trajvlm_experiment.py \
  --config trajvlm/configs/experiments/stage1_pixmo_cap.yaml --print-config

uv run --project trajvlm --locked --extra train torchrun --nproc-per-node=8 \
  trajvlm/launch_scripts/trajvlm_experiment.py \
  --config trajvlm/configs/experiments/stage2_videoqa.yaml \
  --set run.name=videoqa_lr1e5 --set training.connector_learning_rate=1e-5
```

torchrun の GPU/ノード数・MASTER_ADDR 等は起動トポロジーとして引き続き外部で指定し、実行メタデータに記録する。YAML に変更不能な torchrun 起動値を混在させない。

### 3. 設定確認は実行から分離する

CPU のプレビューは process group 初期化、CUDA 利用、モデル構築、データセット実体の読み込み、ダウンロード、W&B 初期化、結果ディレクトリ作成を行わない。`--export-config` の明示的な書き出し先のみ作成する。Stage2 のモデル継承には checkpoint の設定メタデータを読む必要があるため、その YAML は必要とし、重みファイルは読まない。メタデータ未配置時は必要なパスをエラー表示する。CPU テストでは小さな checkpoint config fixture を使う。

Stage1 の期間は `training.max_steps` または `training.epochs` と `data.train_examples` の明示指定を要求し、排他検証する。基準レシピには従来の約 22k steps に対応する明示ステップ数と根拠を記載し、実際に取得できた PixMoCap 件数で epochs が変わることを明記する。従来 CLI の自動件数算出は互換経路に残す。新 config 経路はダウンロード中のデータにも依存せずプレビューできる。

最終検証には未知キー・型・必須値・環境変数、正の batch/frames、学習率、混合重み、num_traj と前処理/バックボーンの一致、期間と stop_at の一致を含める。world size が未確定の CPU 確認では batch の分散整合性は未確認と表示し、訓練起動時に実 world size で検証する。未確認事項を「実行可能」と表示しない。

### 4. 基準実験のデータと評価を明示する

Stage1 は PixMoCap-only を用意し、pointing/NLP を暗黙に加えない。Stage2 は `VIDEO_ACADEMIC_DATASETS` を再利用した明示的な VideoQA mixture を用意する。Molmo2 全体 mixture は既存互換経路から選べる。

評価は `evaluation.enabled` と対象を YAML に持ち、最初のレシピでは明示的に無効化する。既存の画像評価や追跡評価が VideoQA-only の実行に自動追加されないようにする。初期 SFT レシピの動画条件は既存 shell に合わせ 64 frames / 8 frames_per_clip とし、Python 既定値 128 / 16 との違いを記載する。W&B は初期レシピで無効とし、有効時のみ project/entity 等を要求する。

### 5. 記録と再開

trainer が保存する `config.yaml` の形式を保ち、入力 recipe、合成・解決済み recipe、入力ファイル一覧/override、Git SHA と dirty 状態、uv.lock SHA256、Python/主要依存バージョン、実行 CLI と分散トポロジーを追加保存する。環境変数全体や認証情報を保存しない。

新規実行は既存出力への暗黙の上書きを禁止する。自動生成名は用いず、明示 run.name / output_dir で比較・再利用を容易にする。再開時は既存 trainer の互換性チェックを保ち、初回の manifest を保持した上で各再開を別記録にする。再開に渡した変更後の設定も追跡できるようにする。

### 6. 既存起動との互換性

最初に両 entrypoint から設定構築を副作用のない builder に分離する。TrajVLM 型登録を共通化し、Stage1/2 の import 順による registry 汚染をなくす。旧 argparse は旧デフォルト値と引数解釈を維持して共通 builder を呼ぶ。新 recipe の値を旧 CLI のデフォルト値で上書きしない。既存 `pretrain.sh` / `sft.sh` のデフォルト差を勝手に統一しない。

## Risks / Trade-offs

- 設定構築中の隠れた I/O → builder 抽出と I/O 禁止テスト、checkpoint config 読み込みを注入可能にする。
- モデル型の dispatch が壊れる → TrajVlmConfig round-trip と import 順序の回帰テスト。
- 旧 CLI と YAML の条件がずれる → 同一入力のモデル/データ/optimizer/scheduler 等の比較テスト。
- 任意の内部 override は安全性と両立しにくい → 派生に影響する欄を拒否し、recipe の公開項目に寄せる。
- CPU 検証はデータ実在性や GPU カーネル互換性を保証しない → プレビューと実行前環境確認を明確に分ける。

## Migration Plan

型登録/builder 抽出 → recipe loader → CPU preview → stage recipes → manifest → ドキュメントの順で実装する。旧 CLI を常に使えるよう残し、新ランチャーを選択的に導入する。問題時は旧ランチャーへ戻せ、checkpoint 形式は変更しない。

## Open Questions

実装を阻害する未決事項はなし。Stage2 全 video_academic のうち実際に取得できる subset と、最終的な steps / epochs / GPU トポロジーは、データ準備後に実験 YAML で確定する。将来の sweep、実験 DB、クラスタ用設定は今回の基盤が安定した後に検討する。
