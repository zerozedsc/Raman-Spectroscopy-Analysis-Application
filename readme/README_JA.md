# ラマン分光分析アプリケーション (Raman Spectroscopy Analysis Application)
## 完全版ドキュメント (日本語)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](../LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PySide6](https://img.shields.io/badge/GUI-PySide6-green.svg)](https://www.qt.io/qt-for-python)

**バージョン:** 1.0.0  
**最終更新:** 2026年1月  
**言語:** 日本語

---

## 目次

1. [はじめに](#はじめに)
2. [インストール](#インストール)
3. [使い方 (Getting Started)](#使い方)
4. [機能一覧](#機能一覧)
5. [ユーザーインターフェース](#ユーザーインターフェース)
6. [前処理手法 (Preprocessing)](#前処理手法)
7. [解析手法 (Analysis)](#解析手法)
8. [開発 (Development)](#開発)
9. [プロジェクトへの貢献](#プロジェクトへの貢献)
10. [トラブルシューティング](#トラブルシューティング)
11. [APIリファレンス](#apiリファレンス)
12. [ライセンス](#ライセンス)

---

## はじめに

### プロジェクトについて

本ラマン分光分析アプリケーションは、ラマン分光法を用いた**リアルタイム分類および疾患検出**のために設計された包括的なデスクトップソフトウェアです。本プロジェクトは、**富山大学 臨床フォトニクスおよび情報工学研究室**の指導の下、**卒業研究**として開発されました。

<div align="center">
  <img src="images/app-overview.png" alt="アプリケーション概要" width="800"/>
</div>

### 研究背景

#### 現状の課題

ラマン分光分析における現在の課題：

1. **手作業による処理**
   - 研究者がMATLABやPythonスクリプトを用いて手動でスペクトルを処理する必要がある
   - 時間がかかり、人為的ミスのリスクがある
   - プログラミングの専門知識が必要

2. **高額な商用ソフトウェア**
   - 既存の医療/生物学的分光ソフトウェアは高価なライセンスが必要
   - カスタマイズ性が低い
   - 特定のベンダーに依存してしまう

3. **オープンソースソリューションの不足**
   - 利用可能なオープンソースGUIアプリケーションが少ない
   - コミュニティ主導の開発が限定的
   - 最新の機械学習ツールとの統合が不十分

#### プロジェクトの目標

本プロジェクトは以下の目標を掲げています：

1. **包括的な分析ツールの提供**
   - 完全な前処理パイプラインの実装
   - 従来の手法と最新の分類アルゴリズムの両方をサポート
   - カスタムパイプライン設定の実現

2. **使いやすいソフトウェアの開発**
   - プログラマーでなくても使える直感的なGUI
   - リアルタイム処理と可視化のサポート
   - クロスプラットフォーム対応

3. **研究および臨床利用の促進**
   - 医療応用のための説明可能性（Explainability）の実装
   - 詳細な結果解釈の提供
   - 臨床意思決定ワークフローの支援

### 学術情報

**学生:** ムハマド ヘルミ ビン ロザイン (Muhamad Helmi bin Rozain)  
**学籍番号:** 12270294  
**所属:** 富山大学 工学部  
**研究室:** [臨床フォトニクスおよび情報工学研究室](http://www3.u-toyama.ac.jp/medphoto/)

**指導教員:**
- 大嶋　佑介 先生 (Yusuke Oshima)
- 竹谷　皓規 先生 (Hironori Taketani)

### 主な機能

- ✅ **40種類以上の前処理手法** - 研究で検証されたアルゴリズム
- ✅ **リアルタイム解析** - インタラクティブな可視化と分類
- ✅ **モダンなGUI** - 直感的なPySide6/Qt6インターフェース
- ✅ **多言語対応** - 日本語と英語をサポート
- ✅ **オープンソース** - 学術および商用利用可能なMITライセンス
- ✅ **クロスプラットフォーム** - Windows, macOS, Linux対応

---

## インストール

### システム要件

**最小要件:**
- **OS:** Windows 10/11, macOS 10.14+, または Linux (Ubuntu 18.04+)
- **Python:** 3.8 以上
- **RAM:** 4 GB (大規模データセットの場合は8 GB推奨)
- **ストレージ:** 500 MB の空き容量
- **ディスプレイ:** 1280x720 解像度 (1920x1080 推奨)

**オプション要件:**
- **GPU:** CUDA対応NVIDIA GPU (ディープラーニング機能用)

### インストール方法

#### 方法 1: ソースコードからのインストール (開発者向け推奨)

```bash
# 1. リポジトリをクローン
git clone https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application.git
cd Raman-Spectroscopy-Analysis-Application

# 2. 仮想環境の作成
python -m venv .venv

# 3. 仮想環境のアクティベート
# Windowsの場合:
.venv\Scripts\activate
# macOS/Linuxの場合:
source .venv/bin/activate

# 4. 依存関係のインストール
pip install -r requirements.txt

# 5. アプリケーションの実行
python main.py
```

#### 方法 2: UVパッケージマネージャーの使用 (ユーザー向け推奨)

```bash
# 1. UVのインストール
pip install uv

# 2. クローンとディレクトリ移動
git clone https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application.git
cd Raman-Spectroscopy-Analysis-Application

# 3. 環境作成と依存関係インストール
uv venv
uv pip install -e .

# 4. アプリケーションの実行
uv run python main.py
```

#### 方法 3: ポータブル実行ファイル (Windowsのみ)

Python環境をインストールせずに利用する場合：

1. [Releases](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/releases) から最新のポータブル版をダウンロード
2. ZIPファイルを解凍
3. `RamanApp.exe` を実行

**特徴:**
- ✅ インストール不要
- ✅ すべての依存関係を同梱
- ✅ 単一の実行ファイル (50-80 MB)
- ✅ USBメモリから実行可能

#### 方法 4: インストーラー (Windowsのみ)

Windowsへ恒久的にインストールする場合：

1. [Releases](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/releases) からインストーラーをダウンロード
2. `.exe` インストーラーを実行
3. ウィザードに従ってインストール
4. スタートメニューまたはデスクトップショートカットから起動

**特徴:**
- ✅ 一般的なインストール体験
- ✅ スタートメニューへの統合
- ✅ プロジェクトファイルの関連付け
- ✅ 簡単なアンインストール

### インストールの確認

インストール後、以下のコマンドで動作確認を行ってください：

```bash
# 簡単なテストを実行
python -c "import PySide6; print('PySide6 OK')"
python -c "import ramanspy; print('RamanSPy OK')"
python -c "import numpy; print('NumPy OK')"

# またはアプリケーションを起動
python main.py
```

---

## 使い方

### 初回起動

アプリケーションを初めて起動すると：

1. **言語選択**
   - 日本語または英語を選択します
   - 後から設定で変更可能です

2. **ウェルカム画面**
   - 主要機能の概要
   - クイックチュートリアル（オプション）

3. **最初のプロジェクト作成**
   - 「新規プロジェクト」ボタンをクリック
   - プロジェクト名と説明を入力
   - 保存場所を選択

<div align="center">
  <img src="images/first-launch.png" alt="初回起動画面" width="700"/>
</div>

### 基本ワークフロー

#### 1. プロジェクトの作成または開く

```
ホーム画面 → 新規プロジェクト
- プロジェクト名を入力 (例: "MGUS_Classification")
- 説明を追加 (任意)
- 保存場所を選択
- 「作成」をクリック
```

#### 2. スペクトルデータの読み込み

```
データパッケージ画面 → データのインポート
- 対応フォーマット: CSV, Excel, TXT, .spc
- 単一ファイルまたは一括インポート
- フォーマット自動検出
```

#### 3. スペクトルの前処理

```
前処理画面 (Preprocessing)
- パイプラインに前処理ステップを追加
- 各ステップのパラメータを設定
- リアルタイムで結果をプレビュー
- 処理済みデータをエクスポート
```

#### 4. 結果の解析

```
解析画面 (Analysis)
- 解析手法を選択 (PCA, クラスタリング等)
- 解析するデータセットを選択
- インタラクティブな可視化を確認
- 結果をエクスポート
```

### クイック例: 単一スペクトルの前処理

```python
# GUI操作の概念的な流れ:

1. データの読み込み
   - ファイル → インポート → "sample.csv" を選択
   
2. 前処理ステップの追加
   - "+" ボタンをクリック
   - "Baseline Correction" → "ASLS" を選択
   - lambda=1e6, p=0.05 に設定
   
3. 追加ステップ
   - "+" を再度クリック
   - "Normalization" → "Vector Norm" を選択
   - norm_type="L2" に設定
   
4. 結果のプレビュー
   - 処理前後の比較が自動的に表示されます
   - 必要に応じてパラメータを調整
   
5. 適用とエクスポート
   - "Apply to All"（すべてに適用）をクリック
   - エクスポート → "Save Processed Data"
```

---

## 機能一覧

### 1. 前処理パイプライン

#### 概要

複数の処理手法を連鎖させ、リアルタイムプレビューとパラメータ調整が可能なパイプライン機能です。

<div align="center">
  <img src="images/preprocessing-pipeline-detail.png" alt="前処理パイプライン" width="750"/>
</div>

#### 利用可能なカテゴリ

**ベースライン補正 (Baseline Correction)**
- **ASLS** (Asymmetric Least Squares)
- **Polynomial Baseline** (多項式フィッティング)
- **IASLS** (Improved ASLS)
- **Butterworth High-Pass Filter**

**正規化 (Normalization)**
- **Vector Normalization** (L1, L2, Max)
- **MinMax Scaling**
- **Z-Score Standardization**
- **Quantile Normalization**
- **Probabilistic Quotient Normalization (PQN)**
- **Rank Transform**

**平滑化・微分 (Smoothing & Derivatives)**
- **Savitzky-Golay Smoothing**
- **Savitzky-Golay Derivatives** (1次, 2次微分)
- **Moving Average** (移動平均)
- **Gaussian Filter**

**特徴量エンジニアリング**
- **Peak-Ratio Features** (MGUS/MM分類用ピーク比)
- **Peak Detection and Integration** (ピーク検出・積分)
- **Spectral Binning**

**ディープラーニング**
- **Convolutional Autoencoder (CDAE)**
- ノイズ除去とベースライン補正の統合処理

**高度な手法**
- **Cosmic Ray Removal** (宇宙線除去)
- **Spike Detection** (スパイク検出)
- **Noise Reduction** (ノイズ低減)

#### パイプラインの機能

- **ドラッグ＆ドロップ並べ替え** - 処理順序を簡単に変更
- **ステップの有効化/無効化** - 削除せずに一時的にオフにする
- **パラメータの保存** - 設定は自動的に保存されます
- **バッチ処理** - 複数のスペクトルにパイプラインを適用
- **パイプラインのエクスポート** - 設定を保存して共有

### 2. 解析手法

#### 探索的データ解析 (Exploratory Analysis)

**主成分分析 (PCA)**
- 次元圧縮
- 分散説明率の確認
- バイプロット可視化
- ローディング分析
- スコア分布と統計検定

<div align="center">
  <img src="images/pca-analysis.png" alt="PCA解析結果" width="700"/>
</div>

**t-SNE**
- 非線形次元圧縮
- クラスタの可視化
- Perplexityの最適化

**UMAP**
- 最新の次元圧縮手法
- t-SNEより高速
- 局所構造と大域構造の両方を保持

#### クラスタリング解析

**K-Means クラスタリング**
- クラスタ数の自動選択 (エルボー法)
- シルエット分析
- クラスタ可視化

**階層型クラスタリング (Hierarchical Clustering)**
- デンドログラム可視化
- 複数のリンケージ手法
- カットオフ最適化

#### 統計解析

**ANOVA (分散分析)**
- 一元配置および二元配置分散分析
- 事後検定 (Tukey HSD)
- 効果量の計算

**相関分析**
- ピアソンおよびスピアマン相関
- ヒートマップ可視化
- 有意性検定

**マン・ホイットニーのU検定**
- ノンパラメトリック比較
- 効果量 (Cohen's d)
- 信頼区間

### 3. 可視化ツール

#### インタラクティブ・プロット

すべてのプロットは以下の操作に対応しています：
- **ズーム** - マウスホイールまたはボックス選択
- **パン** - クリックしてドラッグ
- **エクスポート** - PNG, SVG, PDFとして保存
- **カスタマイズ** - 色、マーカー、ラベルの変更

<div align="center">
  <img src="images/interactive-plots.png" alt="インタラクティブな可視化機能" width="650"/>
</div>

#### プロットの種類

- **折れ線グラフ** - スペクトルの重ね書き
- **散布図** - PCA/t-SNE結果
- **ヒートマップ** - 相関行列
- **箱ひげ図 (Box Plots)** - 統計比較
- **バイオリン図** - 分布の可視化
- **デンドログラム** - 階層クラスタリング

---

## ユーザーインターフェース

### メイン画面レイアウト

<div align="center">
  <img src="images/ui-layout-annotated.png" alt="UIレイアウト" width="800"/>
</div>

#### 1. ナビゲーションバー (左側)

- **Home** - プロジェクト管理
- **Data Package** - データのインポートと整理
- **Preprocessing** - 前処理パイプラインの設定
- **Analysis** - 解析手法の実行
- **Workspace** - プロジェクトファイルの確認
- **Settings** - アプリケーション設定

#### 2. メインコンテンツエリア (中央)

アクティブなページのコンテンツを表示：
- データテーブル
- インタラクティブプロット
- パラメータコントロール
- 結果の可視化

#### 3. サイドバー (右側)

コンテキストに応じた情報を表示：
- 最近のプロジェクト (Home画面)
- データセットリスト (Preprocessing画面)
- 解析履歴 (Analysis画面)
- パラメータヒント

#### 4. ステータスバー (下部)

- 現在のプロジェクト名
- 処理ステータス
- メモリ使用量
- エラー通知

### Preprocessing画面

#### パイプラインの構築手順

**ステップ 1: 手法の追加**
```
"+" をクリック → カテゴリを選択 → 手法を選択
```

**ステップ 2: パラメータ設定**
```
スライダー、入力欄、ドロップダウンで調整
プレビューの即時更新を確認
```

**ステップ 3: 並べ替え（必要に応じて）**
```
ステップをドラッグして順序を変更
順序は重要です！（例：正規化の前にベースライン補正を行うなど）
```

**ステップ 4: テストと調整**
```
ステップのオン/オフを切り替えて比較
プレビューを見ながらパラメータを微調整
```

**ステップ 5: 適用**
```
"Apply to Selected" → 選択したデータセットのみ処理
"Apply to All" → プロジェクト全体を処理
"Export Pipeline" → 設定を保存
```

---

## 前処理手法

### ベースライン補正

#### ASLS (Asymmetric Least Squares)

**目的:** ピークを保持しつつ、ベースライン（蛍光バックグラウンド）を除去します。

**パラメータ:**
- `lambda` (λ): 滑らかさ (1e3 ～ 1e10)
  - 低い値 = データに追従
  - 高い値 = 滑らかなベースライン
  - 推奨値: ラマンスペクトルの場合 1e6

- `p`: 非対称性 (0.001 ～ 0.1)
  - ピークとベースラインの重み付けを制御
  - 推奨値: 生体試料の場合 0.05

**用途:**
- 強い蛍光を持つ生体試料
- ブロードなバックグラウンドを持つ鉱物試料
- 平坦でないベースラインを持つあらゆる試料

### 正規化

#### PQN (Probabilistic Quotient Normalization)

**目的:** サンプルの希釈効果を補正します。

**アルゴリズム概要:**
1. 参照スペクトル（通常は中央値スペクトル）を選択
2. 各波数点での商（quotient）を計算
3. 商の中央値を用いて各スペクトルをスケーリング

**用途:**
- 体液分析（血液、尿）
- 密度の異なる細胞培養サンプル

### ディープラーニング

#### Convolutional Autoencoder (CDAE)

**目的:** ディープラーニングを用いた統合的なノイズ除去とベースライン補正。

**特徴:**
- データから最適な前処理を学習
- 複雑なノイズパターンに対応
- 学習後はパラメータ調整不要

**要件:**
- PyTorchがインストールされていること
- トレーニングデータ（100スペクトル以上推奨）

---

## 解析手法

### 主成分分析 (PCA)

#### 概要

高次元のスペクトルデータを、分散を最大化する少数の主成分に圧縮します。

<div align="center">
  <img src="images/pca-explained.png" alt="PCAの解説" width="700"/>
</div>

#### 主なパラメータ

- **n_components:** 計算する主成分の数（デフォルト: 3）
- **scaling:** スケーリング手法（StandardScaler推奨）
- **show_ellipses:** 95%信頼楕円の表示有無

#### 結果の解釈

- **スコアプロット:** サンプル間の類似性を表示。近いほど類似しています。
- **ローディングプロット:** 各主成分に寄与する波数（ピーク）を表示。
- **スクリープロット:** 各主成分の寄与率（情報の保持量）を表示。

### t-SNE & UMAP

#### 目的

データの局所的な構造を保持しつつ、クラスタリング傾向を可視化します。

**PCAとの違い:**
- PCAは線形手法（大域的な構造を重視）
- t-SNE/UMAPは非線形手法（局所的な類似性を重視）
- 複雑なデータセットの分類において、より明確な分離を示すことがあります。

---

## 開発

### 開発環境のセットアップ

#### 必要なもの

- Python 3.8 以上
- Git
- テキストエディタまたはIDE (VS Code, PyCharm 推奨)

#### クローンとセットアップ

```bash
# リポジトリのクローン
git clone https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application.git
cd Raman-Spectroscopy-Analysis-Application

# 仮想環境の作成
python -m venv .venv

# アクティベート (Windows)
.venv\Scripts\activate

# アクティベート (macOS/Linux)
source .venv/bin/activate

# 開発モードでインストール
pip install -e ".[dev]"
```

---

## プロジェクトへの貢献

研究コミュニティからの貢献を歓迎します！

### 貢献方法

1. **リポジトリをフォーク**
2. **ブランチを作成** (`git checkout -b feature/your-feature`)
3. **変更をコミット** (`git commit -m "feat: 新機能の追加"`)
4. **プッシュ** (`git push origin feature/your-feature`)
5. **プルリクエストを作成**

### 歓迎される貢献

- 🐛 バグ修正
- 📖 ドキュメントの改善
- 🧪 未テスト機能へのテスト追加
- 🌍 翻訳の改善
- ✨ 新しい前処理・解析手法の追加

---

## トラブルシューティング

### よくある問題

#### "ModuleNotFoundError: No module named 'PySide6'"

**解決策:**
```bash
pip install PySide6
# または
pip install -r requirements.txt
```

#### アプリケーションが起動しない

**解決策:**
1. `logs/` ディレクトリのログファイルを確認してください
2. デバッグモードで起動して詳細を確認します:
   ```bash
   python main.py --debug
   ```

#### データ読み込みエラー "Unsupported file format"

**解決策:**
1. 対応フォーマット (.csv, .xlsx, .txt, .spc) か確認してください
2. ファイルが破損していないか確認してください
3. エンコーディング（UTF-8推奨）を確認してください

---

## ライセンス

本プロジェクトは **MIT License** の下で公開されています。

```
MIT License

Copyright (c) 2024-2026 Muhamad Helmi bin Rozain
Laboratory for Clinical Photonics and Information Engineering
University of Toyama
```
詳細は [LICENSE](../LICENSE) ファイルをご参照ください。

---

## 謝辞

### 学術的支援

**富山大学**
- 研究施設および計算リソースの提供

**臨床フォトニクスおよび情報工学研究室**
- ウェブサイト: http://www3.u-toyama.ac.jp/medphoto/
- 臨床応用に関するガイダンス
- 分光装置へのアクセスと研究協力

**指導教員:**
- **大嶋　佑介 先生** (Yusuke Oshima) - 技術指導およびプロジェクト監督
- **竹谷　皓規 先生** (Hironori Taketani) - 臨床的知見および検証

---

## 引用

本ソフトウェアを研究で使用する場合は、以下の引用をお願いいたします：

```bibtex
@software{helmi2024raman,
  author = {Rozain, Muhamad Helmi bin},
  title = {Raman Spectroscopy Analysis Application: 
           Real-Time Classification Software for Disease Detection},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application},
  version = {1.0.0},
  institution = {University of Toyama, 
                 Laboratory for Clinical Photonics and Information Engineering}
}
```

---

## お問い合わせ

**開発者:** ムハマド ヘルミ ビン ロザイン (学生ID: 12270294)  
**所属:** 富山大学・臨床フォトニクスおよび情報工学研究室

**連絡先:**
- GitHub: [@zerozedsc](https://github.com/zerozedsc)
- 研究室: http://www3.u-toyama.ac.jp/medphoto/

**不具合報告:**
- [GitHub Issues](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/issues) までお願いします。

---

<div align="center">
  <p><strong>Raman Spectroscopy Analysis Application をご利用いただきありがとうございます！</strong></p>
  <p>科学・医療研究コミュニティのために ❤️ を込めて開発されました</p>
  <p>
    <a href="http://www3.u-toyama.ac.jp/medphoto/">臨床フォトニクスおよび情報工学研究室</a> •
    <a href="https://www.u-toyama.ac.jp/">富山大学</a>
  </p>
</div>
