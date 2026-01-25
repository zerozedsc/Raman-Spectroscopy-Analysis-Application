# API リファレンス

ラマン分光分析アプリケーションの完全なAPI仕様

---

## 📖 概要

このAPIリファレンスは、開発者がアプリケーションのコンポーネント、関数、およびクラスを理解し使用するための完全なドキュメントです。

---

## 🎯 対象読者

### アプリケーション開発者

カスタム機能を追加または拡張したい開発者：

- **[コアモジュール](core.md)** - アプリケーションの基礎
- **[ページ](pages.md)** - UIページとワークフロー
- **[コンポーネント](components.md)** - 再利用可能なUIコンポーネント

### アルゴリズム開発者

新しい分析手法や前処理アルゴリズムを追加したい研究者：

- **[関数モジュール](functions.md)** - 処理関数とアルゴリズム
- **前処理レジストリ** - 準備中（カスタム前処理手法の登録）
- **分析プラグイン** - 準備中（カスタム分析手法）

### UI開発者

カスタムウィジェットやビューを作成したい開発者：

- **[ウィジェット](widgets.md)** - カスタムQtウィジェット
- **可視化コンポーネント** - 準備中（グラフとプロット）
- **ダイアログ** - 準備中（モーダルウィンドウ）

---

## 📚 APIセクション

### コアモジュール

アプリケーションの基本構造と機能

**モジュール**:
- `main.py` - アプリケーションエントリーポイント
- `utils.py` - ユーティリティ関数
- `dev_runner.py` - 開発モード実行
- `splash_screen.py` - スプラッシュスクリーン

[詳細を見る →](core.md)

### ページモジュール

アプリケーションの各ページとそのロジック

**ページ**:
- `HomePage` - ホーム画面
- `DataPackagePage` - データ管理
- `PreprocessPage` - スペクトル前処理
- `AnalysisPage` - データ分析
- `MachineLearningPage` - 機械学習
- `WorkspacePage` - ワークスペース管理

[詳細を見る →](pages.md)

### コンポーネントモジュール

再利用可能なUIコンポーネント

**コンポーネント**:
- `AppTabs` - タブナビゲーション
- `PageRegistry` - ページ登録システム
- `Toast` - 通知システム
- `SpectrumViewer` - スペクトル表示
- `PipelineBuilder` - パイプライン構築

[詳細を見る →](components.md)

### 関数モジュール

データ処理と分析関数

**カテゴリ**:
- **前処理**: 40以上の前処理関数
- **分析**: PCA、UMAP、クラスタリング
- **統計**: 検定と相関分析
- **機械学習**: 分類アルゴリズム
- **可視化**: プロットとグラフ

[詳細を見る →](functions.md)

### ウィジェットモジュール

カスタムQtウィジェット

**ウィジェット**:
- `ParameterWidget` - パラメータ入力
- `FloatParameterWidget` - 浮動小数点入力
- `ChoiceParameterWidget` - 選択肢入力
- `MatplotlibWidget` - グラフ表示
- `ResultsPanel` - 結果表示

[詳細を見る →](widgets.md)

---

## 🔧 基本的な使用方法

### アプリケーションの起動

```python
from main import main

if __name__ == "__main__":
    main()
```

### カスタムページの作成

```python
from pages.base_page import BasePage
from components.page_registry import PageRegistry

class CustomPage(BasePage):
    """カスタムページの例"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """UIをセットアップ"""
        # UIコンポーネントを追加
        pass
    
    def get_state(self):
        """ページの状態を取得"""
        return {}
    
    def set_state(self, state):
        """ページの状態を設定"""
        pass

# ページを登録
PageRegistry.register_page(
    "custom",
    CustomPage,
    "カスタムページ",
    icon="custom_icon.png"
)
```

### カスタム前処理関数の追加

```python
from functions.preprocess.registry import MethodRegistry
import numpy as np

def custom_preprocessing(spectrum, param1=10, param2=0.5):
    """
    カスタム前処理関数
    
    Parameters
    ----------
    spectrum : np.ndarray
        入力スペクトル
    param1 : int
        パラメータ1の説明
    param2 : float
        パラメータ2の説明
    
    Returns
    -------
    np.ndarray
        処理済みスペクトル
    """
    # カスタム処理ロジック
    processed = spectrum * param2 + param1
    return processed

# レジストリに登録
MethodRegistry.register(
    name="custom_method",
    function=custom_preprocessing,
    category="Custom",
    description="カスタム前処理手法",
    parameters={
        "param1": {"type": "int", "default": 10, "min": 1, "max": 100},
        "param2": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0}
    }
)
```

### カスタムウィジェットの作成

```python
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import pyqtSignal

class CustomWidget(QWidget):
    """カスタムウィジェットの例"""
    
    # カスタムシグナル
    value_changed = pyqtSignal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """UIをセットアップ"""
        layout = QVBoxLayout()
        self.label = QLabel("カスタムウィジェット")
        layout.addWidget(self.label)
        self.setLayout(layout)
    
    def set_value(self, value):
        """値を設定"""
        self.label.setText(str(value))
        self.value_changed.emit(value)
    
    def get_value(self):
        """値を取得"""
        return self.label.text()
```

---

## 📊 アーキテクチャ概要

### レイヤー構造

```
┌─────────────────────────────────────┐
│   プレゼンテーション層 (Presentation)│
│   - ページ (Pages)                  │
│   - ウィジェット (Widgets)          │
│   - ダイアログ (Dialogs)            │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│   コンポーネント層 (Component)       │
│   - AppTabs                         │
│   - PageRegistry                    │
│   - SpectrumViewer                  │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│   ビジネスロジック層 (Business Logic)│
│   - 前処理関数                      │
│   - 分析関数                        │
│   - 機械学習関数                    │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│   データ層 (Data)                   │
│   - ファイルI/O                     │
│   - 設定管理                        │
│   - モデル保存                      │
└─────────────────────────────────────┘
```

### データフロー

```
1. データロード
   User → DataPackagePage → data_loader → DataFrame

2. 前処理
   User → PreprocessPage → Pipeline → preprocess_functions → processed_data

3. 分析
   User → AnalysisPage → analysis_functions → results → Visualization

4. 機械学習
   User → MachineLearningPage → ML_functions → model → predictions
```

### イベント駆動アーキテクチャ

```python
# Qtシグナル/スロットパターン
class DataPage(BasePage):
    # シグナル定義
    data_loaded = pyqtSignal(object)
    groups_changed = pyqtSignal(dict)
    
    def load_data(self, filepath):
        """データをロード"""
        data = load_csv(filepath)
        self.data_loaded.emit(data)  # シグナルを発火
    
    def create_group(self, name, samples):
        """グループを作成"""
        self.groups[name] = samples
        self.groups_changed.emit(self.groups)  # シグナルを発火

# 別のページでシグナルを受信
class AnalysisPage(BasePage):
    def __init__(self, parent=None):
        super().__init__(parent)
        # データページのシグナルに接続
        data_page.data_loaded.connect(self.on_data_loaded)
        data_page.groups_changed.connect(self.on_groups_changed)
    
    def on_data_loaded(self, data):
        """データロード時に呼ばれる"""
        self.data = data
        self.update_ui()
    
    def on_groups_changed(self, groups):
        """グループ変更時に呼ばれる"""
        self.groups = groups
        self.update_group_selector()
```

---

## 🎨 デザインパターン

### 1. Model-View パターン

```python
# Model: データとビジネスロジック
class PreprocessingModel:
    def __init__(self):
        self.pipeline = []
        self.data = None
    
    def add_step(self, method, params):
        """パイプラインにステップを追加"""
        self.pipeline.append((method, params))
    
    def apply_pipeline(self, data):
        """パイプラインを適用"""
        result = data.copy()
        for method, params in self.pipeline:
            result = method(result, **params)
        return result

# View: UIと表示ロジック
class PreprocessingView(QWidget):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.setup_ui()
    
    def setup_ui(self):
        """UIをセットアップ"""
        # UIコンポーネント
        pass
    
    def update_display(self):
        """モデルの状態を表示"""
        # ビューを更新
        pass

# Controller: ユーザー入力の処理
class PreprocessPage(BasePage):
    def __init__(self):
        super().__init__()
        self.model = PreprocessingModel()
        self.view = PreprocessingView(self.model)
    
    def on_add_step(self):
        """ステップ追加ボタンが押された"""
        method, params = self.view.get_selected_method()
        self.model.add_step(method, params)
        self.view.update_display()
```

### 2. レジストリパターン

```python
class MethodRegistry:
    """前処理手法のレジストリ"""
    
    _methods = {}
    
    @classmethod
    def register(cls, name, function, category, **metadata):
        """手法を登録"""
        cls._methods[name] = {
            "function": function,
            "category": category,
            "metadata": metadata
        }
    
    @classmethod
    def get(cls, name):
        """手法を取得"""
        return cls._methods.get(name)
    
    @classmethod
    def list_by_category(cls, category):
        """カテゴリ別に手法をリスト"""
        return [
            name for name, info in cls._methods.items()
            if info["category"] == category
        ]

# 使用例
MethodRegistry.register(
    "asls",
    asls_baseline,
    "Baseline Correction",
    description="Asymmetric Least Squares",
    parameters={"lambda": 100000, "p": 0.01}
)
```

### 3. ファクトリーパターン

```python
class ParameterWidgetFactory:
    """パラメータウィジェットのファクトリー"""
    
    @staticmethod
    def create(param_type, name, **kwargs):
        """パラメータタイプに応じてウィジェットを作成"""
        if param_type == "float":
            return FloatParameterWidget(name, **kwargs)
        elif param_type == "int":
            return IntParameterWidget(name, **kwargs)
        elif param_type == "choice":
            return ChoiceParameterWidget(name, **kwargs)
        elif param_type == "bool":
            return BoolParameterWidget(name, **kwargs)
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

# 使用例
widget = ParameterWidgetFactory.create(
    "float",
    "lambda",
    default=100000,
    min=1,
    max=1000000
)
```

---

## 🔌 拡張ポイント

### カスタム前処理手法の追加

```python
# 1. 関数を定義
def custom_method(spectrum, param1, param2):
    """カスタム前処理手法"""
    # 実装
    return processed_spectrum

# 2. レジストリに登録
MethodRegistry.register(
    name="custom_method",
    function=custom_method,
    category="Custom",
    description="カスタム手法の説明",
    parameters={
        "param1": {"type": "float", "default": 1.0},
        "param2": {"type": "int", "default": 10}
    }
)

# 3. 自動的にUIに表示される
```

### カスタム分析手法の追加

```python
# 1. 分析関数を定義
def custom_analysis(data, param1, param2):
    """カスタム分析手法"""
    # 実装
    return results

# 2. AnalysisPageに統合
class AnalysisPage(BasePage):
    def __init__(self):
        super().__init__()
        self.register_method("custom_analysis", custom_analysis)
    
    def run_custom_analysis(self):
        """カスタム分析を実行"""
        results = custom_analysis(self.data, param1, param2)
        self.display_results(results)
```

### カスタムエクスポート形式の追加

```python
class CustomExporter:
    """カスタムエクスポートクラス"""
    
    @staticmethod
    def export(data, filepath, **options):
        """データをカスタム形式でエクスポート"""
        # エクスポートロジック
        pass

# エクスポートレジストリに登録
ExportRegistry.register(
    format="custom",
    exporter=CustomExporter,
    extensions=[".custom"],
    description="カスタムフォーマット"
)
```

---

## 📖 詳細ドキュメント

各モジュールの詳細なAPIドキュメントは、以下のページをご覧ください：

- **[コアモジュール](core.md)** - アプリケーションの基礎
- **[ページ](pages.md)** - UIページとワークフロー
- **[コンポーネント](components.md)** - 再利用可能なUIコンポーネント
- **[関数](functions.md)** - 処理関数とアルゴリズム (60以上の関数)
- **[ウィジェット](widgets.md)** - カスタムQtウィジェット

---

## 🔗 関連リソース

### 開発ガイド

- **[アーキテクチャ](../dev-guide/architecture.md)** - システム設計の詳細
- **[貢献ガイド](../dev-guide/contributing.md)** - 開発ワークフロー
- **[テストガイド](../dev-guide/testing.md)** - テスト戦略

### ユーザードキュメント

- **[ユーザーガイド](../user-guide/index.md)** - アプリケーションの使用方法
- **[分析手法](../analysis-methods/index.md)** - 手法の詳細

### 外部リソース

- **[PyQt6ドキュメント](https://www.riverbankcomputing.com/static/Docs/PyQt6/)** - Qt API
- **[NumPyドキュメント](https://numpy.org/doc/)** - 数値計算
- **[scikit-learnドキュメント](https://scikit-learn.org/stable/)** - 機械学習

---

## 💡 ベストプラクティス

### コーディング規約

```python
# Googleスタイルのdocstring
def process_spectrum(
    spectrum: np.ndarray,
    method: str,
    **params
) -> np.ndarray:
    """
    スペクトルを処理する
    
    Parameters
    ----------
    spectrum : np.ndarray
        入力スペクトル (n_points,)
    method : str
        処理手法の名前
    **params
        手法固有のパラメータ
    
    Returns
    -------
    np.ndarray
        処理済みスペクトル (n_points,)
    
    Raises
    ------
    ValueError
        未知の手法が指定された場合
    
    Examples
    --------
    >>> spectrum = np.random.rand(1000)
    >>> processed = process_spectrum(spectrum, "asls", lambda=100000)
    """
    # 実装
    pass
```

### エラーハンドリング

```python
from typing import Optional

class PreprocessingError(Exception):
    """前処理エラーの基底クラス"""
    pass

class InvalidParameterError(PreprocessingError):
    """無効なパラメータエラー"""
    pass

def safe_preprocessing(
    spectrum: np.ndarray,
    method: str,
    **params
) -> Optional[np.ndarray]:
    """
    安全な前処理実行
    
    エラーをキャッチしてログに記録
    """
    try:
        result = apply_method(spectrum, method, **params)
        return result
    except InvalidParameterError as e:
        logger.error(f"Invalid parameter: {e}")
        return None
    except PreprocessingError as e:
        logger.error(f"Preprocessing failed: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return None
```

### 型ヒント

```python
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import numpy.typing as npt

# 型エイリアス
Spectrum = npt.NDArray[np.float64]
SpectraArray = npt.NDArray[np.float64]  # shape: (n_spectra, n_points)
Parameters = Dict[str, Union[int, float, str, bool]]

def apply_pipeline(
    spectra: SpectraArray,
    pipeline: List[Tuple[str, Parameters]]
) -> Optional[SpectraArray]:
    """
    パイプラインを適用
    
    Parameters
    ----------
    spectra : SpectraArray
        入力スペクトル配列
    pipeline : List[Tuple[str, Parameters]]
        (手法名, パラメータ) のリスト
    
    Returns
    -------
    Optional[SpectraArray]
        処理済みスペクトル配列、またはNone（エラー時）
    """
    result = spectra.copy()
    for method_name, params in pipeline:
        result = apply_method(result, method_name, **params)
        if result is None:
            return None
    return result
```

---

## 🆘 サポート

### ドキュメント

- **[ユーザーガイド](../user-guide/index.md)** - 機能の使用方法
- **[開発ガイド](../dev-guide/index.md)** - 開発とテスト
- **[FAQ](../faq.md)** - よくある質問

### コミュニティ

- **[GitHub Issues](https://github.com/your-org/raman-app/issues)** - バグ報告
- **[GitHub Discussions](https://github.com/your-org/raman-app/discussions)** - 質問と議論
- **Email**: dev@example.com

---

**最終更新**: 2026年1月24日 | **バージョン**: 1.0.0
