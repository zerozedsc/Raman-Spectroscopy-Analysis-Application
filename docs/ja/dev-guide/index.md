# 開発者ガイド

ラマン分光分析アプリケーションの貢献者と開発者向けの完全なガイド

---

## 📖 概要

このセクションでは、アプリケーションの開発、テスト、デプロイに必要なすべての情報を提供します。初めて貢献する方から経験豊富な開発者まで、プロジェクトに貢献するために必要な知識を得ることができます。

---

## 🎯 このガイドの使い方

### 初めての貢献者

プロジェクトに初めて貢献する場合：

1. **[貢献ガイド](contributing.md)** - 開発ワークフローと標準を学ぶ
2. **[アーキテクチャ](architecture.md)** - システム設計を理解する
3. **[ビルドシステム](build-system.md)** - ローカル開発環境をセットアップ
4. **[テストガイド](testing.md)** - テストの書き方を学ぶ

### 経験豊富な開発者

すでに開発経験がある場合：

1. **[アーキテクチャ](architecture.md)** - 設計パターンとモジュール構成
2. **[API リファレンス](../api/index.md)** - すべてのAPIの詳細
3. **[貢献ガイド](contributing.md)** - プルリクエストプロセス
4. **[ビルドシステム](build-system.md)** - CI/CDとデプロイメント

---

## 📚 ガイドセクション

### 1. アーキテクチャ

**内容**: システム設計と構造の完全なドキュメント

**トピック**:
- システム概要と4つのアーキテクチャレイヤー
- 6つのデザインパターン（MVC、Observer、Registry、Factory、Strategy、Command）
- モジュール構成と依存関係
- データフローとワークフロー
- 状態管理とイベント駆動設計
- プラグインアーキテクチャと拡張ポイント
- パフォーマンス最適化戦略

[詳細を見る →](architecture.md)

### 2. 貢献ガイド

**内容**: 開発ワークフロー、コーディング標準、プルリクエストプロセス

**トピック**:
- 環境のセットアップ（前提条件、インストール、pre-commitフック）
- 開発ワークフロー（Git Flow、ブランチ戦略、コミット規約）
- コーディング標準（PEP 8、Black、Ruff、型ヒント、docstrings）
- 機能追加ワークフロー（前処理手法、分析手法、ML、ページ、ウィジェット）
- テスト要件（カバレッジ目標、テスト実行）
- ドキュメント標準
- プルリクエストプロセス（チェックリスト、レビュー、承認）
- リリースプロセス（セマンティックバージョニング、Git Flowリリース）

[詳細を見る →](contributing.md)

### 3. ビルドシステム

**内容**: ビルド、パッケージング、デプロイメントの完全なガイド

**トピック**:
- 開発環境（UVパッケージマネージャー）
- プロジェクト設定（完全なpyproject.toml）
- 依存関係管理（コア、オプション、開発依存関係）
- アプリケーションのビルド（PyInstaller設定、ビルドコマンド）
- プラットフォーム別ビルド（Windows/macOS/Linux、アイコン、署名、公証）
- インストーラーの作成（NSIS、DMG、DEB）
- CI/CDパイプライン（完全なGitHub Actionsワークフロー）
- ドキュメントのビルド（Sphinx、ReadTheDocs）
- トラブルシューティング（プラットフォーム別の問題）

[詳細を見る →](build-system.md)

### 4. テストガイド

**内容**: テスト戦略、テストの書き方、ベストプラクティス

**トピック**:
- テスト戦略（テストピラミッド、目標、カバレッジ目標）
- テスト構造（ディレクトリ構成、命名規則）
- ユニットテスト（20以上の完全な例：前処理、分析、ML）
- 統合テスト（パイプライン、ワークフロー、ページ間連携）
- UIテスト（ウィジェット、ページ、ダイアログ with pytest-qt）
- テストフィクスチャ（共通フィクスチャ、パラメータ化）
- コードカバレッジ（設定、レポート）
- 継続的テスト（pre-commitフック、GitHub Actions、pytest-watch）
- ベストプラクティス（6つのプラクティス with DO/DON'T例）
- トラブルシューティング（ランダム失敗、UI問題、遅いテスト）

[詳細を見る →](testing.md)

---

## 🚀 クイックスタート

### 開発環境のセットアップ（10分）

```bash
# 1. リポジトリをクローン
git clone https://github.com/your-org/raman-app.git
cd raman-app

# 2. UVをインストール（macOS/Linux）
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex

# 3. 仮想環境を作成
uv venv

# 4. 仮想環境を有効化
# Windows:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# 5. 依存関係をインストール
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt
uv pip install -e .

# 6. pre-commitフックをインストール
uv pip install pre-commit
pre-commit install

# 7. アプリケーションを実行
python main.py

# 8. テストを実行
pytest --cov=. --cov-report=html
```

✅ **完了！** 開発環境が準備できました。

---

## 🔧 開発ツール

### 必須ツール

| ツール           | 目的           | インストール                                       |
| ---------------- | -------------- | -------------------------------------------------- |
| **UV**           | パッケージ管理 | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **Git**          | バージョン管理 | [git-scm.com](https://git-scm.com/)                |
| **Python 3.10+** | ランタイム     | [python.org](https://www.python.org/)              |

### 推奨ツール

| ツール      | 目的         | インストール                                            |
| ----------- | ------------ | ------------------------------------------------------- |
| **VS Code** | エディタ     | [code.visualstudio.com](https://code.visualstudio.com/) |
| **Black**   | フォーマット | `uv pip install black`                                  |
| **Ruff**    | リンター     | `uv pip install ruff`                                   |
| **pytest**  | テスト       | `uv pip install pytest`                                 |
| **mypy**    | 型チェック   | `uv pip install mypy`                                   |

### VS Code拡張機能

```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-python.black-formatter",
    "charliermarsh.ruff",
    "ms-toolsai.jupyter"
  ]
}
```

---

## 📋 プロジェクト構造

```
raman-app/
├── main.py                    # アプリケーションエントリーポイント
├── utils.py                   # ユーティリティ関数
├── dev_runner.py              # 開発モード実行
├── pyproject.toml             # プロジェクト設定
├── requirements.txt           # コア依存関係
├── requirements-dev.txt       # 開発依存関係
│
├── pages/                     # アプリケーションページ
│   ├── home_page.py
│   ├── data_package_page.py
│   ├── preprocess_page.py
│   ├── analysis_page.py
│   └── machine_learning_page.py
│
├── components/                # 再利用可能なコンポーネント
│   ├── app_tabs.py
│   ├── page_registry.py
│   └── widgets/
│
├── functions/                 # ビジネスロジック
│   ├── preprocess/            # 前処理関数（40以上）
│   ├── ML/                    # 機械学習アルゴリズム
│   └── visualization/         # 可視化関数
│
├── configs/                   # 設定管理
│   ├── configs.py
│   └── user_settings.py
│
├── assets/                    # 静的アセット
│   ├── icons/
│   ├── fonts/
│   └── locales/
│
├── tests/                     # テスト
│   ├── unit/
│   ├── integration/
│   └── ui/
│
├── docs/                      # ドキュメント
│   ├── user-guide/
│   ├── analysis-methods/
│   ├── api/
│   └── dev-guide/
│
└── build_scripts/             # ビルドスクリプト
    ├── build_portable.sh
    ├── build_installer.ps1
    └── test_build_executable.py
```

---

## 💡 一般的な開発タスク

### 新しい前処理手法の追加

```bash
# 1. 関数を作成
vim functions/preprocess/my_method.py

# 2. レジストリに登録
vim functions/preprocess/registry.py

# 3. テストを作成
vim tests/unit/test_my_method.py

# 4. テストを実行
pytest tests/unit/test_my_method.py

# 5. ドキュメントを更新
vim docs/analysis-methods/preprocessing.md

# 6. コミット
git add .
git commit -m "feat(preprocess): add my_method preprocessing"

# 7. プルリクエストを作成
git push origin feature/my-method
```

**所要時間**: 30-60分

### 新しい分析手法の追加

```bash
# 1. 分析関数を作成
vim functions/analysis/my_analysis.py

# 2. AnalysisPageに統合
vim pages/analysis_page.py

# 3. テストを作成
vim tests/unit/test_my_analysis.py

# 4. コミット
git commit -m "feat(analysis): add my_analysis method"
```

**所要時間**: 20-30分

### 新しいページの追加

```bash
# 1. ページクラスを作成
vim pages/my_page.py

# 2. PageRegistryに登録
vim components/page_registry.py

# 3. AppTabsに追加
vim components/app_tabs.py

# 4. テストを作成
vim tests/ui/test_my_page.py

# 5. ドキュメントを作成
vim docs/user-guide/my-feature.md
```

**所要時間**: 60-90分

### 新しいウィジェットの追加

```bash
# 1. ウィジェットクラスを作成
vim components/widgets/my_widget.py

# 2. テストを作成
vim tests/ui/test_my_widget.py

# 3. ドキュメントを更新
vim docs/api/widgets.md
```

**所要時間**: 30-45分

---

## 🎨 コーディング標準

### Pythonスタイル

```python
# PEP 8準拠、若干の修正あり

# 1. インポート
from typing import List, Optional
import numpy as np
from PyQt6.QtWidgets import QWidget

# 2. 定数
MAX_ITERATIONS = 100
DEFAULT_LAMBDA = 100000

# 3. 関数
def process_spectrum(
    spectrum: np.ndarray,
    method: str = "asls",
    **params
) -> Optional[np.ndarray]:
    """
    スペクトルを処理する
    
    Parameters
    ----------
    spectrum : np.ndarray
        入力スペクトル
    method : str
        処理手法
    **params
        追加パラメータ
    
    Returns
    -------
    Optional[np.ndarray]
        処理済みスペクトル
    """
    # 実装
    pass

# 4. クラス
class PreprocessingPipeline:
    """前処理パイプラインを管理"""
    
    def __init__(self):
        self.steps: List[tuple] = []
    
    def add_step(self, method: str, params: dict):
        """ステップを追加"""
        self.steps.append((method, params))
    
    def apply(self, spectrum: np.ndarray) -> np.ndarray:
        """パイプラインを適用"""
        result = spectrum.copy()
        for method, params in self.steps:
            result = self._apply_method(result, method, params)
        return result
```

### 命名規則

| 要素             | 規則       | 例                   |
| ---------------- | ---------- | -------------------- |
| **モジュール**   | snake_case | `preprocessing.py`   |
| **クラス**       | PascalCase | `PreprocessPage`     |
| **関数**         | snake_case | `apply_asls()`       |
| **定数**         | UPPER_CASE | `MAX_ITERATIONS`     |
| **変数**         | snake_case | `spectrum_data`      |
| **プライベート** | _prefix    | `_internal_method()` |

### Docstrings

```python
def apply_asls(
    spectrum: np.ndarray,
    lambda_: float = 100000,
    p: float = 0.01,
    max_iter: int = 10
) -> np.ndarray:
    """
    Asymmetric Least Squares ベースライン補正を適用
    
    Parameters
    ----------
    spectrum : np.ndarray
        入力スペクトル (n_points,)
    lambda_ : float, optional
        平滑化パラメータ、デフォルト100000
    p : float, optional
        非対称性パラメータ、デフォルト0.01
    max_iter : int, optional
        最大反復回数、デフォルト10
    
    Returns
    -------
    np.ndarray
        ベースライン補正済みスペクトル
    
    References
    ----------
    Eilers, P. H. C., & Boelens, H. F. M. (2005).
    Baseline Correction with Asymmetric Least Squares Smoothing.
    Leiden University Medical Centre Report, 1, 5.
    
    Examples
    --------
    >>> spectrum = np.random.rand(1000) + np.linspace(0, 100, 1000)
    >>> corrected = apply_asls(spectrum, lambda_=100000, p=0.01)
    >>> baseline = spectrum - corrected
    """
    # 実装
    pass
```

---

## 🧪 テスト戦略

### テストピラミッド

```
        ┌─────────┐
        │ UIテスト │  ← 少数（遅い、壊れやすい）
        ├─────────┤
        │ 統合テスト│  ← 中程度
        ├─────────┤
        │ユニット │  ← 多数（高速、安定）
        │テスト   │
        └─────────┘
```

### カバレッジ目標

| カテゴリ             | 目標  |
| -------------------- | ----- |
| **関数**             | ≥ 80% |
| **ビジネスロジック** | ≥ 80% |
| **ページ**           | ≥ 70% |
| **ウィジェット**     | ≥ 70% |
| **全体**             | ≥ 75% |

### テスト実行

```bash
# すべてのテスト
pytest

# カバレッジ付き
pytest --cov=. --cov-report=html

# 特定のファイル
pytest tests/unit/test_preprocessing.py

# 特定のテスト
pytest tests/unit/test_preprocessing.py::test_asls

# 並列実行
pytest -n auto

# 冗長モード
pytest -v

# 最初の失敗で停止
pytest -x
```

---

## 🔄 Git ワークフロー

### ブランチ戦略（Git Flow）

```
main (production)
  ├── develop (next release)
  │   ├── feature/new-preprocessing
  │   ├── feature/improved-ui
  │   └── feature/ml-algorithm
  ├── hotfix/critical-bug
  └── release/1.1.0
```

### コミットメッセージ規約

```bash
# 形式: <type>(<scope>): <subject>

# 例:
git commit -m "feat(preprocess): add wavelet denoising"
git commit -m "fix(analysis): correct PCA variance calculation"
git commit -m "docs(api): update preprocessing functions"
git commit -m "test(ml): add Random Forest unit tests"
git commit -m "refactor(ui): simplify page navigation"
```

**タイプ**:
- `feat`: 新機能
- `fix`: バグ修正
- `docs`: ドキュメント
- `style`: フォーマット
- `refactor`: リファクタリング
- `perf`: パフォーマンス改善
- `test`: テスト
- `build`: ビルドシステム
- `ci`: CI設定
- `chore`: その他の変更

---

## 📊 パフォーマンス最適化

### ベンチマーク

```python
import time
import numpy as np

def benchmark_preprocessing(method, spectrum, **params):
    """前処理手法のベンチマーク"""
    start = time.time()
    for _ in range(100):
        result = method(spectrum, **params)
    end = time.time()
    
    avg_time = (end - start) / 100
    print(f"Average time: {avg_time*1000:.2f} ms")
    return avg_time

# 使用例
spectrum = np.random.rand(1000)
benchmark_preprocessing(apply_asls, spectrum, lambda_=100000)
```

### プロファイリング

```python
import cProfile
import pstats

# プロファイリング実行
cProfile.run('run_analysis()', 'profile_stats')

# 結果の表示
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative')
stats.print_stats(20)  # トップ20を表示
```

---

## 🆘 サポート

### ドキュメント

- **[アーキテクチャ](architecture.md)** - システム設計
- **[貢献ガイド](contributing.md)** - 開発ワークフロー
- **[ビルドシステム](build-system.md)** - ビルドとデプロイ
- **[テストガイド](testing.md)** - テスト戦略

### コミュニティ

- **[GitHub Issues](https://github.com/your-org/raman-app/issues)** - バグとタスク
- **[GitHub Discussions](https://github.com/your-org/raman-app/discussions)** - 質問と議論
- **Email**: dev@example.com

### プルリクエスト

プルリクエストを作成する準備ができたら：

1. **[貢献ガイド](contributing.md)** を確認
2. **チェックリスト**を完了
3. **テスト**が通ることを確認
4. **レビュー**を依頼

---

## 🎯 ロードマップ

### v1.1.0（計画中）

- [ ] 深層学習前処理（CDAE）
- [ ] リアルタイムスペクトル取得
- [ ] プラグインシステム
- [ ] パフォーマンス最適化

### v1.2.0（検討中）

- [ ] Webインターフェース
- [ ] クラウド統合
- [ ] 共同作業機能
- [ ] モバイルアプリ

---

## 🌟 貢献者

プロジェクトへの貢献に感謝します！

- すべての貢献者は [CONTRIBUTORS.md](https://github.com/your-org/raman-app/blob/main/CONTRIBUTORS.md) に記載されます
- 重要な貢献は [CHANGELOG.md](https://github.com/your-org/raman-app/blob/main/CHANGELOG.md) に記録されます

---

**貢献をお待ちしています！** 🚀

質問やアイデアがあれば、お気軽にお問い合わせください。

---

**最終更新**: 2026年1月24日 | **バージョン**: 1.0.0
