# 変更履歴

ラマン分光分析アプリケーションの重要な変更点はこのファイルに記録します。

形式は [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) に基づき、バージョニングは [Semantic Versioning](https://semver.org/spec/v2.0.0.html) に従います。

> 注: このページは翻訳・整備中です。項目名や固有名詞（手法名など）は英語表記のままになる場合があります。

## [Unreleased]

### ドキュメント
- Read the Docs 向けの包括的なドキュメント構成
- 主要機能のユーザーガイド
- 分析手法リファレンス
- 開発者向け API ドキュメント
- 日本語翻訳（進行中）

## [1.0.0-alpha] - 2026-01-24

### Added

#### コア機能
- PySide6/Qt6 によるデスクトップアプリ
- 多言語対応（英語・日本語）
- ワークスペースを用いたプロジェクト管理
- 複数データセットを扱うデータパッケージ管理
- グループベースのサンプル整理

#### 前処理（40+ 手法）
- **Baseline Correction**: AsLS, AirPLS, Polynomial, Whittaker, FABC, Butterworth High-Pass
- **Smoothing**: Savitzky-Golay, Gaussian, Moving Average, Median Filter
- **Normalization**: Vector, Min-Max, Area, SNV, MSC, Quantile, PQN, Rank Transform
- **Derivatives**: 1st and 2nd order Savitzky-Golay
- **Feature Engineering**: Peak Ratio, Wavelet Transform
- **Advanced**: Convolutional Denoising Autoencoder (CDAE)
- **Pipeline System**: 前処理パイプラインの保存・読み込み・共有
- **Real-time Preview**: 適用前に効果を確認

#### 分析手法
- **Exploratory**:
  - Principal Component Analysis (PCA)（ローディング、スクリープロット）
  - UMAP
  - t-SNE
  - Hierarchical Clustering（デンドログラム）
  - K-means Clustering（エルボー法）
- **Statistical**:
  - Pairwise tests（t-test, Mann-Whitney U, Wilcoxon）
  - Multi-group comparisons（ANOVA）
  - Correlation analysis（Pearson, Spearman, Kendall）
  - Band ratio analysis（範囲カスタマイズ可能）
  - Peak detection and identification
- **Visualization**:
  - Interactive heatmaps
  - Waterfall plots
  - Overlaid spectra（グループ色分け）
  - Peak scatter plots
  - Correlation matrices

#### 機械学習
- **Algorithms**: SVM, Random Forest, XGBoost, Logistic Regression, Linear Regression
- **Validation**: GroupKFold（患者単位分割）, LOPOCV, Stratified K-Fold, Hold-out
- **Evaluation**: ROC/AUC, confusion matrix, classification report, calibration curve
- **Interpretability**: permutation importance, feature importance（波数対応）
- **Export**: pickle / ONNX

#### ビルドシステム
- Windows ポータブル版（単体実行ファイル）
- Windows インストーラ（NSIS）
- PowerShell によるビルド自動化
- 実行ファイル向けのテスト

### Fixed

#### 2026年1月
- **分析ページの安定性** (2026-01-23):
  - 安全なキャンセルのための2段階停止
  - PCA空間での外れ値検出を高速化
  - 相関ヒートマップの目盛りクリップ修正
  - バンドルフォント登録（EN/JA 描画の安定化）
  - MLドロップダウンの黒いポップアップスタイル修正
  - バンド比プロットの埋め込み改善（PathPatch 保持）

- **ML 評価の改善** (2026-01-22):
  - 評価サマリータブ
  - データリーク警告
  - データセット単位の指標

- **Grouped Mode 分析** (2026-01-21):
  - グループモードPCAのタブ構成修正
  - 大規模データでの同期による高速化

- **ML UI 改善** (2026-01-20):
  - MLページのUI再設計
  - ドラッグ＆ドロップによるグループ管理
  - i18n 充実
  - 大規模データセットでのパフォーマンス改善

#### 2025年10月
- **パラメータ型検証** (2025-10-15):
  - FABC ベースライン補正の整数変換修正
  - 2段階型検証
  - 40手法の検証（100% pass）
  - 互換性維持

- **前処理 UI/UX** (2025-10-08):
  - パイプライン eye ボタンのクラッシュ修正
  - 微分オーダーの空フィールド修正
  - feature engineering の enumerate バグ修正
  - deep learning モジュールの構文エラー修正

- **UI 調整** (2025-10-07):
  - 入力データセットのレイアウト改善
  - パイプライン選択の視認性改善
  - 追加ボタン色（青→緑）
  - セクション見出しの統一

### Changed

- **ドキュメント構造**: 公開 docs/ とローカル .docs/ を分離
- **README 構成**: 問題/FAQ を README から切り出し
- **ビルド**: PyInstaller 6.16.0+ へ更新

### Security

- **型検証**: パラメータ型チェックにより不正入力を抑止
- **パス検証**: ファイル操作時のパス検証でディレクトリトラバーサルを抑止

## [0.1.0] - 2025-10-01

### Added
- 初期アルファリリース
- 基本前処理パイプライン
- PCA 分析
- 機械学習（簡易）

## リリースノート

### バージョン 1.0.0-alpha

本リリースは、富山大学の卒業研究として開発された最初のアルファ版です。

**Status**: Alpha - 機能は一通り揃っていますが、検証と改善を継続中です。

**推奨**:
- 研究室
- 学術機関
- 手法開発・検証

**非推奨**:
- 臨床診断用途（未承認）
- 医療システム本番運用

**既知の制限**:
1. 一部前処理は追加検証が必要
2. deep learning 機能は GPU が推奨
3. 大規模データ（>5000 スペクトル）は最適化が必要な場合あり
4. 追加機能は今後予定

**今後の機能** (v1.1.0):
- 追加のスペクトル分離手法（NMF, ICA）
- バッチ処理強化
- REST API
- CLI
- 追加フォーマット対応（SPC, WDF）※計画中（現時点では未実装）
- 動画チュートリアル
- 日本語ドキュメントの完成
- マレー語翻訳

## Contributors

（整備中）
