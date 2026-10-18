## 1. 設定構築を実行から分離

- [ ] 1.1 両 entrypoint の現在の代表的な設定を CPU fixture で記録し、shell/Python の既定値差を整理する。
- [ ] 1.2 TrajVlmConfig/前処理型と registry 登録を共通化し、Stage1/2 を両順序で import した際の型解決を検証する。
- [ ] 1.3 Stage1/2 の副作用のない builder を抽出する。checkpoint 設定読取と legacy の件数取得を注入可能にし、distributed 初期化を訓練開始時へ移す。

## 2. 型付き recipe と合成

- [ ] 2.1 schema_version と stage ごとの recipe schema、公開設定と許可する trainer_overrides を定義する。
- [ ] 2.2 extends の順序付き merge、循環検出、list 置換、環境変数と定義元相対パスの解決を実装する。
- [ ] 2.3 --config と反復 --set を実装し、全入力合成後に派生値を生成する。未知キー、型、必須値、ステップ/epochs、混合重み、前処理とモデルの整合性を検証する。

## 3. ステージ別の基準実験

- [ ] 3.1 PixMoCap-only レシピと明示 duration を定義し、pointing/NLP・W&B・評価の有効無効を明示する。
- [ ] 3.2 VIDEO_ACADEMIC_DATASETS を再利用する VideoQA-only mixture を追加し、checkpoint config 継承と frames 条件の反映を確認する。
- [ ] 3.3 基底、Stage1、Stage2、学習率だけ変更する派生 YAML のサンプルを追加する。共有設定からホスト絶対パスを除く。

## 4. CPU での検証・表示

- [ ] 4.1 新ランチャーに --print-config / --export-config を実装し、合成 recipe と最終 TrainConfig を表示・保存する。
- [ ] 4.2 config-only 時に CUDA/process group/model/data/weight/W&B/download が呼ばれないことを禁止 stub で検証する。Stage2 はローカル config fixture を使う。
- [ ] 4.3 不正 YAML、循環、未設定環境変数、継承元相対パス、異なる cwd、CLI 優先順位、期間の再計算、未知トポロジー時の表示をテストする。

## 5. 実行記録と互換性

- [ ] 5.1 rank 0 による recipe/manifest 保存を追加し、既存 config.yaml を維持する。Git dirty と uv.lock ハッシュを記録し、秘密値を含めない。
- [ ] 5.2 出力先の衝突防止と、初回 manifest を保持する resume 記録を実装する。
- [ ] 5.3 旧 CLI と新 YAML の同等入力についてモデル、データ、optimizer、scheduler、期間の設定を比較する。既存 shell の引数転送を確認する。

## 6. 利用手順と完了条件

- [ ] 6.1 README/UV_SETUP に CPU preview、単一/複数 GPU 起動、Stage1→Stage2、派生、resume、データと GPU 検証の限界を記載する。
- [ ] 6.2 uv の固定環境で上記 CPU テスト、サンプル preview と export/reload を通し、既存設定との差分を確認する。GPU 訓練は別段階と明示する。
