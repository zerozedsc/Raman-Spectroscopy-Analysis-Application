# インストールガイド

ラマン分光分析アプリケーションのインストール方法

---

## 📦 インストールオプション

3つのインストール方法から選択できます：

| 方法                 | 対象者         | 難易度 | 所要時間 |
| -------------------- | -------------- | ------ | -------- |
| **実行可能ファイル** | エンドユーザー | 簡単   | 2分      |
| **Pythonパッケージ** | 開発者         | 中級   | 5-10分   |
| **ソースから**       | 貢献者         | 上級   | 10-15分  |

---

## 方法1: 実行可能ファイル（推奨）

エンドユーザーに最適 - Pythonのインストール不要

### Windows

#### ステップ1: ダウンロード

[**最新リリースをダウンロード**](https://github.com/your-org/raman-app/releases/latest)

- `RamanApp-windows.zip`（ポータブル版）
- `RamanApp-installer.exe`（インストーラー版）

#### ステップ2: インストール

**ポータブル版**:
```powershell
# ZIPを解凍
Expand-Archive -Path RamanApp-windows.zip -DestinationPath C:\RamanApp

# 実行
C:\RamanApp\RamanApp.exe
```

**インストーラー版**:
1. `RamanApp-installer.exe`をダブルクリック
2. インストールウィザードに従う
3. スタートメニューのショートカットをクリック

#### 必要条件

- Windows 10 以降
- 4 GB RAM（8 GB推奨）
- 500 MB ディスク空き容量

#### トラブルシューティング

**問題**: "Windows によってPCが保護されました"

**解決策**:
1. "詳細情報"をクリック
2. "実行"をクリック

**問題**: "VCRUNTIME140.dll が見つかりません"

**解決策**:
[Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)をインストール

### macOS

#### ステップ1: ダウンロード

[**最新リリースをダウンロード**](https://github.com/your-org/raman-app/releases/latest)

- `RamanApp-macos.dmg`

#### ステップ2: インストール

```bash
# DMGをマウント
open RamanApp-macos.dmg

# Applicationsフォルダにドラッグ
# アプリケーションを開く
```

#### 必要条件

- macOS 11 (Big Sur) 以降
- 4 GB RAM（8 GB推奨）
- 500 MB ディスク空き容量

#### トラブルシューティング

**問題**: "「RamanApp.app」は破損しているため開けません"

**解決策**:
```bash
# ターミナルで実行
xattr -cr /Applications/RamanApp.app
```

**問題**: "開発元が未確認のため開けません"

**解決策**:
1. システム環境設定 → セキュリティとプライバシー
2. "このまま開く"をクリック

### Linux

#### ステップ1: ダウンロード

[**最新リリースをダウンロード**](https://github.com/your-org/raman-app/releases/latest)

- `RamanApp-linux.tar.gz`（すべてのディストリビューション）
- `RamanApp-x86_64.AppImage`（AppImage）
- `raman-app_1.0.0_amd64.deb`（Debian/Ubuntu）

#### ステップ2: インストール

**tar.gz版**:
```bash
# 解凍
tar -xzf RamanApp-linux.tar.gz

# 実行
cd RamanApp
./RamanApp
```

**AppImage版**:
```bash
# 実行可能にする
chmod +x RamanApp-x86_64.AppImage

# 実行
./RamanApp-x86_64.AppImage
```

**DEB版（Ubuntu/Debian）**:
```bash
# インストール
sudo dpkg -i raman-app_1.0.0_amd64.deb

# 依存関係を修正
sudo apt-get install -f

# 実行
raman-app
```

#### 必要条件

- Ubuntu 20.04+、Fedora 35+、または同等
- 4 GB RAM（8 GB推奨）
- 500 MB ディスク空き容量

#### 依存関係

```bash
# Ubuntu/Debian
sudo apt-get install libxcb-xinerama0 libxcb-icccm4 libxcb-image0

# Fedora
sudo dnf install xcb-util-wm xcb-util-image
```

---

## 方法2: Pythonパッケージ

開発者とPythonユーザー向け

### 前提条件

- Python 3.10以上
- pip または uv パッケージマネージャー

**Pythonバージョンの確認**:
```bash
python --version
# または
python3 --version
```

### UVを使用（推奨）

**UV**は高速なPythonパッケージマネージャーです。

#### ステップ1: UVをインストール

**macOS/Linux**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**:
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

#### ステップ2: 仮想環境を作成

```bash
# プロジェクトディレクトリを作成
mkdir raman-analysis
cd raman-analysis

# 仮想環境を作成
uv venv

# 有効化
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

#### ステップ3: アプリケーションをインストール

```bash
# PyPIからインストール（利用可能な場合）
uv pip install raman-app

# またはGitHubから
uv pip install git+https://github.com/your-org/raman-app.git
```

#### ステップ4: 実行

```bash
# コマンドラインから
raman-app

# またはPythonモジュールとして
python -m raman_app
```

### pipを使用

#### ステップ1: 仮想環境を作成

```bash
# 仮想環境を作成
python -m venv venv

# 有効化
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

#### ステップ2: インストール

```bash
# PyPIから
pip install raman-app

# またはGitHubから
pip install git+https://github.com/your-org/raman-app.git

# 開発依存関係を含める
pip install raman-app[dev]
```

#### ステップ3: 実行

```bash
raman-app
```

### オプション機能

**機械学習（拡張）**:
```bash
uv pip install raman-app[ml]
# 含まれるもの: XGBoost, UMAP
```

**深層学習**:
```bash
uv pip install raman-app[deep-learning]
# 含まれるもの: PyTorch, CDAE用
```

**すべての機能**:
```bash
uv pip install raman-app[ml,deep-learning,dev]
```

---

## 方法3: ソースから

貢献者と開発者向け

### ステップ1: リポジトリをクローン

```bash
# HTTPS経由でクローン
git clone https://github.com/your-org/raman-app.git
cd raman-app

# またはSSH
git clone git@github.com:your-org/raman-app.git
cd raman-app
```

### ステップ2: 仮想環境をセットアップ

**UVを使用（推奨）**:
```bash
# 仮想環境を作成
uv venv

# 有効化
# Windows:
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# 依存関係をインストール
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt

# 編集可能モードでインストール
uv pip install -e .
```

**pipを使用**:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

### ステップ3: pre-commitフックをインストール

```bash
# pre-commitをインストール
uv pip install pre-commit

# フックをインストール
pre-commit install

# すべてのファイルでテスト実行
pre-commit run --all-files
```

### ステップ4: 実行

**開発モード（ホットリロード付き）**:
```bash
python dev_runner.py
```

**通常モード**:
```bash
python main.py
```

### ステップ5: テストを実行

```bash
# すべてのテストを実行
pytest

# カバレッジ付き
pytest --cov=. --cov-report=html

# HTMLレポートを開く
# Linux/macOS:
open htmlcov/index.html
# Windows:
start htmlcov/index.html
```

---

## インストールの確認

インストールが成功したか確認します：

### 方法1: アプリケーションを起動

```bash
# 実行可能ファイル
./RamanApp  # または .exe をダブルクリック

# Pythonパッケージ
raman-app

# ソースから
python main.py
```

✅ アプリケーションウィンドウが開くはずです

### 方法2: バージョンを確認

```bash
# コマンドライン
raman-app --version

# Python
python -c "import raman_app; print(raman_app.__version__)"
```

出力例:
```
Raman Analysis Application v1.0.0
```

### 方法3: サンプルデータをロード

1. アプリケーションを開く
2. ホーム → "サンプルデータをロード"
3. スペクトルがロードされることを確認

✅ サンプルスペクトルが表示されるはずです

---

## システム要件

### 最小要件

| コンポーネント | 要件                               |
| -------------- | ---------------------------------- |
| **OS**         | Windows 10, macOS 11, Ubuntu 20.04 |
| **CPU**        | 2 GHz デュアルコア                 |
| **RAM**        | 4 GB                               |
| **ストレージ** | 500 MB 空き容量                    |
| **画面**       | 1280×720                           |
| **Python**     | 3.10以上（ソースからの場合）       |

### 推奨要件

| コンポーネント | 要件                   |
| -------------- | ---------------------- |
| **RAM**        | 8 GB以上               |
| **CPU**        | 4 GHz クアッドコア     |
| **ストレージ** | 1 GB 空き容量（SSD）   |
| **画面**       | 1920×1080              |
| **GPU**        | CUDA対応（深層学習用） |

### ソフトウェア依存関係

**必須**:
- NumPy ≥ 1.24.0
- SciPy ≥ 1.11.0
- PyQt6 ≥ 6.5.0
- scikit-learn ≥ 1.3.0
- Matplotlib ≥ 3.7.0
- Pandas ≥ 2.0.0

**オプション**:
- XGBoost ≥ 2.0.0（拡張ML用）
- UMAP-learn ≥ 0.5.3（非線形削減用）
- PyTorch ≥ 2.0.0（深層学習用）

---

## 次のステップ

インストール後：

1. **[クイックスタートガイド](quick-start.md)を読む** - 5分でアプリを学ぶ
2. **[はじめに](getting-started.md)を読む** - セットアップと最初の分析の流れ
3. **サンプルデータで練習** - 組み込みサンプルを使用
4. **自分のデータをロード** - 実際のスペクトルで開始

---

## トラブルシューティング

### 一般的な問題

#### 問題: アプリケーションが起動しない

**確認事項**:
1. システム要件を満たしているか
2. すべての依存関係がインストールされているか
3. ファイルパーミッションが正しいか

**解決策**:
```bash
# 依存関係を再インストール
uv pip install --upgrade --force-reinstall raman-app

# またはソースから再インストール
cd raman-app
uv pip install -e . --force-reinstall
```

#### 問題: インポートエラー

**エラー**: `ModuleNotFoundError: No module named 'xxx'`

**解決策**:
```bash
# 不足しているパッケージをインストール
uv pip install xxx

# またはすべての依存関係を再インストール
uv pip install -r requirements.txt
```

#### 問題: PyQt6エラー

**エラー**: `qt.qpa.plugin: Could not load the Qt platform plugin`

**解決策（Linux）**:
```bash
# 必要なライブラリをインストール
sudo apt-get install libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1

# または
sudo apt-get install --reinstall libqt6widgets6
```

**解決策（macOS）**:
```bash
# PyQt6を再インストール
uv pip uninstall PyQt6
uv pip install PyQt6
```

#### 問題: パフォーマンスが遅い

**原因**: 不十分なリソースまたは大きなデータセット

**解決策**:
1. システムRAMを確認（8 GB以上推奨）
2. バックグラウンドアプリケーションを閉じる
3. より小さなデータセットで開始
4. バッチ処理を有効化

#### 問題: GPU/CUDAエラー（オプション機能）

**エラー**: CUDA関連エラー

**解決策**:
```bash
# CPU専用モードを使用
# PyTorchをCPU版で再インストール
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### プラットフォーム固有の問題

#### Windows

**問題**: 長いパス名エラー

**解決策**:
1. レジストリエディタを開く
2. `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem`に移動
3. `LongPathsEnabled`を`1`に設定

#### macOS

**問題**: "損傷している"警告

**解決策**:
```bash
sudo xattr -rd com.apple.quarantine /Applications/RamanApp.app
```

#### Linux

**問題**: パーミッション拒否

**解決策**:
```bash
chmod +x RamanApp
# または
chmod +x RamanApp-x86_64.AppImage
```

---

## アンインストール

### 実行可能ファイル

**Windows**:
- インストーラー使用: コントロールパネル → プログラムと機能
- ポータブル版: フォルダを削除

**macOS**:
```bash
rm -rf /Applications/RamanApp.app
```

**Linux**:
```bash
# DEB
sudo dpkg -r raman-app

# 手動
rm -rf /path/to/RamanApp
```

### Pythonパッケージ

```bash
# アンインストール
uv pip uninstall raman-app

# 仮想環境を削除
rm -rf venv  # または .venv
```

### ソースから

```bash
# リポジトリを削除
rm -rf raman-app
```

---

## 更新

### 実行可能ファイル

1. 最新バージョンをダウンロード
2. 古いバージョンをアンインストール
3. 新しいバージョンをインストール

**注意**: 設定とプロジェクトは保持されます

### Pythonパッケージ

```bash
# 最新版に更新
uv pip install --upgrade raman-app

# 特定のバージョンに更新
uv pip install raman-app==1.1.0
```

### ソースから

```bash
cd raman-app
git pull origin main
uv pip install -e . --upgrade
```

---

## サポート

インストールに問題がある場合：

1. **[トラブルシューティングガイド](troubleshooting.md)を確認**
2. **[FAQ](faq.md)を検索**
3. **[GitHub Issues](https://github.com/your-org/raman-app/issues)を検索**
4. **新しいissueを作成**（見つからない場合）

**issueを報告する際は以下を含めてください**:
- オペレーティングシステムとバージョン
- Pythonバージョン（該当する場合）
- インストール方法
- エラーメッセージ
- 実行した手順

---

## 関連リンク

- **[クイックスタート](quick-start.md)** - 今すぐ使い始める
- **[ユーザーガイド](user-guide/index.md)** - 完全なドキュメント
- **[GitHub](https://github.com/your-org/raman-app)** - ソースコード
- **[リリース](https://github.com/your-org/raman-app/releases)** - すべてのバージョン

---

**最終更新**: 2026年1月24日 | **バージョン**: 1.0.0
