# ラマン分光分析アプリケーション ドキュメント

リアルタイムラマンスペクトル分類システムの完全なドキュメント

---

## 📋 目次

```{toctree}
:maxdepth: 2
:caption: クイックスタート

getting-started/index
installation
quick-start
```

```{toctree}
:maxdepth: 2
:caption: ユーザーガイド

user-guide/index
user-guide/interface-overview
user-guide/importing-data
user-guide/preprocessing
user-guide/analysis
user-guide/machine-learning
user-guide/best-practices
user-guide/exporting-results
user-guide/project-management
```

```{toctree}
:maxdepth: 2
:caption: 分析手法

analysis-methods/index
analysis-methods/preprocessing
analysis-methods/exploratory
analysis-methods/statistical
analysis-methods/machine-learning
analysis-methods/comparison-tables
```

```{toctree}
:maxdepth: 2
:caption: API リファレンス

api/index
api/core
api/pages
api/components
api/functions
api/widgets
```

```{toctree}
:maxdepth: 2
:caption: 開発ガイド

dev-guide/index
dev-guide/architecture
dev-guide/contributing
dev-guide/build-system
dev-guide/testing
```

```{toctree}
:maxdepth: 1
:caption: 追加リソース

changelog
faq
troubleshooting
glossary
references
roadmap
```

```{toctree}
:hidden:

getting-started
installation
quick-start

analysis-methods/index
api/index
dev-guide/index

user-guide/index
user-guide/importing-data
user-guide/data-import
user-guide/preprocessing
user-guide/analysis
user-guide/machine-learning
```

## 🚀 クイックスタート

初めてご利用の方は、こちらからお始めください：

- **[クイックスタートガイド](quick-start.md)** - 5分でアプリケーションの使用を開始
- **[インストール](installation.md)** - 詳細なインストール手順
- **[はじめに](getting-started/index.md)** - 基本的なワークフローとチュートリアル

---

## 📚 ドキュメントセクション

### ユーザーガイド

エンドユーザー向けの完全な使用説明書

- **[ユーザーガイド概要](user-guide/index.md)** - アプリケーション機能の概要
- **[データのインポート](user-guide/importing-data.md)** - CSVファイルとスペクトルのロード
- **[前処理](user-guide/preprocessing.md)** - スペクトルの前処理と準備
- **[分析](user-guide/analysis.md)** - 探索的分析と統計
- **[機械学習](user-guide/machine-learning.md)** - モデルのトレーニングと評価
- **[結果のエクスポート](user-guide/exporting-results.md)** - データと図のエクスポート
- **[プロジェクト管理](user-guide/project-management.md)** - 作業の保存と整理

### 分析手法

すべての分析手法の詳細なドキュメント

- **[前処理手法](analysis-methods/preprocessing.md)** - 40以上の前処理アルゴリズム
- **[探索的分析](analysis-methods/exploratory.md)** - PCA、UMAP、t-SNE、クラスタリング
- **[統計分析](analysis-methods/statistical.md)** - 仮説検定と相関分析
- **[機械学習](analysis-methods/machine-learning.md)** - SVM、Random Forest、XGBoost
- **[比較表](analysis-methods/comparison-tables.md)** - 手法選択ガイド

### API リファレンス

開発者向けの完全なAPI仕様

- **[コアモジュール](api/core.md)** - アプリケーションのコアとユーティリティ
- **[ページ](api/pages.md)** - アプリケーションページとワークフロー
- **[コンポーネント](api/components.md)** - 再利用可能なUIコンポーネント
- **[関数](api/functions.md)** - 処理関数とアルゴリズム
- **[ウィジェット](api/widgets.md)** - カスタムQtウィジェット

### 開発ガイド

貢献者と開発者向けのガイド

- **[アーキテクチャ](dev-guide/architecture.md)** - システム設計と構造
- **[貢献ガイド](dev-guide/contributing.md)** - 開発ワークフローと標準
- **[ビルドシステム](dev-guide/build-system.md)** - ビルドとデプロイメント
- **[テストガイド](dev-guide/testing.md)** - テスト戦略とベストプラクティス

---

## 🎯 主な機能

### データ処理

- **複数形式対応**: CSV、TXT、Excel
- **バッチ処理**: 複数のスペクトルを同時に処理
- **グループ管理**: サンプルの整理と比較

### 前処理

40以上の前処理手法：

- **ベースライン補正**: AsLS、AirPLS、多項式、Whittaker
- **スムージング**: Savitzky-Golay、ガウシアン、移動平均
- **正規化**: ベクトルノルム、SNV、MSC、分位点正規化
- **微分**: 一次微分、二次微分
- **高度な手法**: ウェーブレット、CDAE、FABC

### 分析

包括的な分析ツール：

- **次元削減**: PCA、UMAP、t-SNE
- **クラスタリング**: K-means、階層的、DBSCAN
- **統計検定**: t検定、ANOVA、相関分析
- **機械学習**: SVM、Random Forest、XGBoost、ロジスティック回帰

### 可視化

インタラクティブで高品質な図：

- **スペクトルプロット**: 個別、平均、比較
- **分析結果**: スコアプロット、デンドログラム、ヒートマップ
- **評価指標**: 混同行列、ROC曲線、学習曲線
- **エクスポート**: PNG、PDF、SVG、Excel

---

## 💡 ユースケース

### 研究

- スペクトルデータの探索的分析
- 化合物の識別と分類
- 統計的仮説検定
- 出版品質の図の生成

### 教育

- ラマン分光法の実践的学習
- データ分析手法の実演
- インタラクティブな教材
- 学生プロジェクト

### 産業

- 品質管理とプロセス監視
- リアルタイム分類
- バッチ分析とレポート作成
- 自動化ワークフロー

---

## 🔧 システム要件

### 最小要件

- **OS**: Windows 10/11、macOS 11+、Linux (Ubuntu 20.04+)
- **メモリ**: 4 GB RAM
- **ストレージ**: 500 MB 空き容量
- **Python**: 3.10以上（ソースからの実行時）

### 推奨要件

- **メモリ**: 8 GB RAM以上
- **プロセッサ**: マルチコアCPU
- **ストレージ**: 1 GB 空き容量
- **GPU**: 深層学習機能用（オプション）

---

## 📖 学習リソース

### チュートリアル

- **[基本ワークフロー](getting-started/basic-workflow.md)** - ステップバイステップガイド
- **[前処理の例](getting-started/preprocessing-tutorial.md)** - 前処理パイプラインの構築
- **[分析の例](getting-started/analysis-tutorial.md)** - PCAとクラスタリング
- **[MLの例](getting-started/ml-tutorial.md)** - モデルのトレーニングと評価

### ビデオガイド

- アプリケーションの概要（5分）
- データのインポートとグループ化（3分）
- 前処理パイプラインの構築（10分）
- 機械学習モデルのトレーニング（15分）

### サンプルデータ

- **組み込みサンプル**: 練習用のデモデータセット
- **ダウンロード**: [サンプルスペクトル](https://example.com/samples)
- **フォーマット**: CSV、適切なラベル付き

---

## 🆘 ヘルプとサポート

### ドキュメント

- **[FAQ](faq.md)** - よくある質問
- **[トラブルシューティング](troubleshooting.md)** - 一般的な問題の解決
- **[用語集](glossary.md)** - 用語と定義

### コミュニティ

- **GitHub Issues**: [バグ報告と機能リクエスト](https://github.com/your-org/raman-app/issues)
- **Discussions**: [質問とディスカッション](https://github.com/your-org/raman-app/discussions)
- **Email**: support@example.com

### 貢献

プロジェクトへの貢献を歓迎します！

- **[貢献ガイド](dev-guide/contributing.md)** - 始め方
- **[Code of Conduct](https://github.com/your-org/raman-app/CODE_OF_CONDUCT.md)** - コミュニティガイドライン
- **[ロードマップ](roadmap.md)** - 計画中の機能

---

## 📝 ライセンス

このプロジェクトはMITライセンスの下でライセンスされています。詳細は[LICENSE](https://github.com/your-org/raman-app/blob/main/LICENSE)をご覧ください。

---

## 🔗 関連リンク

- **[GitHubリポジトリ](https://github.com/your-org/raman-app)**
- **[リリースノート](https://github.com/your-org/raman-app/releases)**
- **[変更履歴](https://github.com/your-org/raman-app/blob/main/CHANGELOG.md)**
- **[ウェブサイト](https://raman-app.example.com)**

---

## 📊 ドキュメント構造

```
docs/
├── index.md                    # このページ
├── quick-start.md             # クイックスタートガイド
├── installation.md            # インストール手順
├── getting-started/           # 初心者向けチュートリアル
│   ├── index.md
│   ├── basic-workflow.md
│   ├── preprocessing-tutorial.md
│   ├── analysis-tutorial.md
│   └── ml-tutorial.md
├── user-guide/                # 完全なユーザーガイド
│   ├── index.md
│   ├── importing-data.md
│   ├── preprocessing.md
│   ├── analysis.md
│   ├── machine-learning.md
│   ├── exporting-results.md
│   └── project-management.md
├── analysis-methods/          # 手法ドキュメント
│   ├── index.md
│   ├── preprocessing.md
│   ├── exploratory.md
│   ├── statistical.md
│   └── machine-learning.md
├── api/                       # API リファレンス
│   ├── index.md
│   ├── core.md
│   ├── pages.md
│   ├── components.md
│   ├── functions.md
│   └── widgets.md
└── dev-guide/                 # 開発者ガイド
    ├── index.md
    ├── architecture.md
    ├── contributing.md
    ├── build-system.md
    └── testing.md
```

---

## 🌐 言語

ドキュメントは以下の言語で利用可能です：

- **English** - [Documentation Home](https://raman-spectroscopy-analysis.readthedocs.io/en/latest/)
- **日本語** - このページ

---

## 🔄 ドキュメントのバージョン

- **最新版**: v1.0.0
- **最終更新**: 2026年1月24日
- **アプリケーションバージョン**: 1.0.0

---

**ラマン分光分析アプリケーションをお選びいただきありがとうございます！**

質問やフィードバックがございましたら、お気軽にお問い合わせください。
