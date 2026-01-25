# トラブルシューティングガイド

ラマン分光分析アプリケーションの一般的な問題と解決方法

---

## 📋 目次

- {ref}`インストール問題 <troubleshooting-install>`
- {ref}`起動とクラッシュ <troubleshooting-startup>`
- {ref}`データロード問題 <troubleshooting-data-load>`
- {ref}`前処理エラー <troubleshooting-preprocess>`
- {ref}`分析エラー <troubleshooting-analysis>`
- {ref}`機械学習問題 <troubleshooting-ml>`
- {ref}`パフォーマンス問題 <troubleshooting-performance>`
- {ref}`UI/表示問題 <troubleshooting-ui>`
- {ref}`エクスポート問題 <troubleshooting-export>`
- {ref}`プラットフォーム固有の問題 <troubleshooting-platform>`

---

(troubleshooting-install)=
## インストール問題

### 問題1: "Python not found"エラー

**症状**:
```
'python' is not recognized as an internal or external command
```

**原因**: Pythonがインストールされていないか、PATHに追加されていません

**解決策**:

**Windows**:
```powershell
# 1. Python 3.10以上をダウンロード
https://www.python.org/downloads/

# 2. インストール時に「Add Python to PATH」にチェック

# 3. 確認
python --version
```

**macOS**:
```bash
# Homebrewを使用
brew install python@3.10

# 確認
python3 --version
```

**Linux**:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3.10 python3-pip

# 確認
python3 --version
```

### 問題2: "Permission denied"インストールエラー

**症状**:
```
ERROR: Could not install packages due to an EnvironmentError: [Errno 13] Permission denied
```

**原因**: システムディレクトリへの書き込み権限がありません

**解決策**:

**オプション1: 仮想環境を使用（推奨）**
```bash
# 仮想環境を作成
python -m venv venv

# 有効化
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# インストール
pip install raman-app
```

**オプション2: ユーザーモードでインストール**
```bash
pip install --user raman-app
```

**オプション3: sudoを使用（推奨しない）**
```bash
# Linux/macOS のみ
sudo pip install raman-app
```

### 問題3: UVのインストールが失敗する

**症状**:
```
Failed to install UV package manager
```

**解決策**:

**Windows**:
```powershell
# PowerShellを管理者として実行
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

# 再度実行
irm https://astral.sh/uv/install.ps1 | iex

# または手動ダウンロード
https://github.com/astral-sh/uv/releases
```

**macOS/Linux**:
```bash
# curlが利用できない場合
brew install curl

# または wgetを使用
wget -qO- https://astral.sh/uv/install.sh | sh

# PATH に追加
export PATH="$HOME/.cargo/bin:$PATH"
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
```

### 問題4: 依存関係の競合

**症状**:
```
ERROR: Cannot install raman-app because these package versions have conflicting dependencies
```

**解決策**:

```bash
# 1. 新しい仮想環境を作成
python -m venv clean_env
source clean_env/bin/activate  # または clean_env\Scripts\activate

# 2. pipを更新
pip install --upgrade pip setuptools wheel

# 3. 依存関係を1つずつインストール
pip install numpy scipy pandas
pip install PyQt6 matplotlib
pip install scikit-learn

# 4. アプリケーションをインストール
pip install raman-app

# または requirements.txt から
pip install -r requirements.txt
```

**特定のバージョンを指定**:
```bash
# 互換性のあるバージョンを使用
pip install numpy==1.24.0 scipy==1.11.0 scikit-learn==1.3.0
```

---

(troubleshooting-startup)=
## 起動とクラッシュ

### 問題5: アプリケーションが起動しない

**症状**: ダブルクリックしても何も起こらない

**診断ステップ**:

```bash
# ステップ1: ターミナルから実行してエラーを確認
python main.py

# ステップ2: 依存関係を確認
pip list | grep -E "PyQt6|numpy|scipy|sklearn|matplotlib|pandas"

# ステップ3: Pythonバージョンを確認
python --version  # 3.10以上が必要
```

**一般的な解決策**:

**解決策A: PyQt6の問題**
```bash
# PyQt6を再インストール
pip uninstall PyQt6 PyQt6-Qt6 PyQt6-sip
pip install PyQt6
```

**解決策B: 環境変数**
```bash
# Linux/macOS
export QT_DEBUG_PLUGINS=1
python main.py

# Windows (PowerShell)
$env:QT_DEBUG_PLUGINS=1
python main.py
```

**解決策C: ディスプレイ設定**
```bash
# Linux リモート接続の場合
export DISPLAY=:0
xhost +

# または Xvfb を使用
xvfb-run python main.py
```

### 問題6: 起動後すぐにクラッシュ

**症状**: スプラッシュスクリーン表示後にクラッシュ

**ログを確認**:

```bash
# ログファイルの場所
# Windows
type %APPDATA%\RamanApp\logs\app.log

# macOS
cat ~/Library/Logs/RamanApp/app.log

# Linux
cat ~/.local/share/RamanApp/logs/app.log
```

**一般的なエラーと解決策**:

**エラー1: "Qt platform plugin"エラー**
```
This application failed to start because no Qt platform plugin could be initialized
```

解決策:
```bash
# Linux
sudo apt-get install libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0

# PyQt6を再インストール
pip install --force-reinstall PyQt6
```

**エラー2: "OpenGL"エラー**
```
Could not initialize OpenGL
```

解決策:
```bash
# ソフトウェアレンダリングを使用
export QT_QUICK_BACKEND=software
python main.py

# またはアプリ設定
設定 → グラフィックス → 
「ソフトウェアレンダリングを使用」
```

**エラー3: "Segmentation fault"**
```
Segmentation fault (core dumped)
```

解決策:
```bash
# 1. すべての依存関係を更新
pip install --upgrade numpy scipy matplotlib PyQt6

# 2. 競合するパッケージを削除
pip uninstall PyQt5  # PyQt5とPyQt6の競合

# 3. システムライブラリを更新（Linux）
sudo apt-get update
sudo apt-get upgrade
```

### 問題7: ランダムなクラッシュ

**症状**: 使用中に不定期にクラッシュ

**診断**:

```text
# デバッグモードで実行
python main.py --debug

# または
python -m pdb main.py
```

**一般的な原因**:

1. **メモリ不足**
   ```bash
   # メモリ使用量を確認
   # Linux
   free -h
   
   # macOS
   vm_stat
   
   # Windows (PowerShell)
   Get-Process python | Select-Object WS
   ```
   
   解決策: データサイズを減らす、RAMを増やす

2. **スレッディング問題**
   ```text
   # 設定で並列処理を無効化
   設定 → パフォーマンス → 
   「並列処理を使用」のチェックを外す
   ```

3. **破損した設定ファイル**
   ```bash
   # 設定をリセット
   # Windows
   rd /s "%APPDATA%\RamanApp"
   
   # macOS/Linux
   rm -rf ~/.config/RamanApp/
   ```

---

(troubleshooting-data-load)=
## データロード問題

### 問題8: CSVファイルが読み込めない

**症状**: "Failed to load file"エラー

**診断**:

```python
# Pythonで手動確認
import pandas as pd
df = pd.read_csv('your_file.csv')
print(df.head())
print(df.shape)
print(df.dtypes)
```

**一般的な問題と解決策**:

**問題A: エンコーディングエラー**
```
UnicodeDecodeError: 'utf-8' codec can't decode byte
```

解決策:
```text
# 異なるエンコーディングを試す
# CSVをUTF-8で再保存

# Excel で:
名前を付けて保存 → CSV UTF-8 (カンマ区切り)

# Python で変換:
import pandas as pd
df = pd.read_csv('file.csv', encoding='latin1')
df.to_csv('file_utf8.csv', encoding='utf-8', index=False)
```

**問題B: 区切り文字の問題**
```
データが1列にまとめられている
```

解決策:
```python
# セミコロン区切りの場合
df = pd.read_csv('file.csv', sep=';')

# タブ区切りの場合
df = pd.read_csv('file.txt', sep='\t')

# 自動検出
df = pd.read_csv('file.csv', sep=None, engine='python')
```

**問題C: 小数点の問題**
```
数値がNaNになる
```

解決策:
```text
# カンマが小数点の場合（ヨーロッパ形式）
df = pd.read_csv('file.csv', decimal=',')

# アプリ内設定
設定 → データ → 小数点記号: カンマ
```

**問題D: ヘッダーの問題**
```
列名が正しく認識されない
```

解決策:
```text
# 正しいフォーマット:
Wavenumber,Sample1,Sample2
400,0.123,0.145
401,0.134,0.156

# 間違ったフォーマット（複数のヘッダー行）:
# Experiment 1
# Date: 2026-01-24
Wavenumber,Sample1,Sample2
400,0.123,0.145
```

最初のヘッダー行のみを残し、コメント行を削除

### 問題9: "Memory Error" データロード時

**症状**: 大きなファイルのロード時にメモリエラー

**解決策**:

**方法1: チャンク読み込み**
```python
# 大きなファイルをチャンクで読み込む
import pandas as pd

chunks = []
for chunk in pd.read_csv('large_file.csv', chunksize=1000):
    # 必要な処理
    processed = chunk[chunk['Wavenumber'].between(500, 2000)]
    chunks.append(processed)

df = pd.concat(chunks)
```

**方法2: 波数範囲を制限**
```
データをロード → 
フィルタ → 波数範囲 → 
範囲を指定（例: 500-2000 cm⁻¹）
```

**方法3: ダウンサンプリング**
```
データをロード → 
リサンプル → 
ステップ: 2（2ポイントごと）または 3
```

**方法4: データ型の最適化**
```text
# float64 → float32に変換
df = df.astype('float32')

# アプリ設定
設定 → データ → 
「float32を使用（メモリ節約）」
```

### 問題10: スペクトルが正しく表示されない

**症状**: ロード後のスペクトルがおかしい

**確認項目**:

1. **波数の順序**
   ```text
   # 昇順であるべき
   wavenumbers = [400, 401, 402, ..., 2000]
   
   # 降順の場合は反転
   データ → 変換 → 波数を反転
   ```

2. **単位**
   ```
   波数単位: cm⁻¹
   強度単位: 任意（自動スケーリング）
   ```

3. **外れ値**
   ```text
   # 異常な値を確認
   データ → 統計 → 
   最小値、最大値、中央値を確認
   
   # 外れ値を除去またはクリップ
   データ → フィルタ → 
   「外れ値を除去」
   ```

---

(troubleshooting-preprocess)=
## 前処理エラー

### 問題11: ベースライン補正が機能しない

**症状**: 適用後もベースラインが残る

**診断**:

```
前処理 → プレビュー → 
青線（元）と緑線（処理後）を比較
```

**解決策**:

**問題A: lambdaが大きすぎる**
```
現在: lambda = 10000000
→ ベースラインが過度に平坦化、ピークも除去

推奨: lambda = 100000
```

**問題B: pが大きすぎる**
```
現在: p = 0.5
→ 非対称性が弱い、ピークがベースラインとして扱われる

推奨: p = 0.01
```

**問題C: 反復回数が不足**
```
設定 → AsLS → 
max_iter: 10（デフォルト）→ 20に増やす
```

**問題D: データが適さない**
```
# 負の値がある場合
データ → 変換 → 
「負の値をゼロに設定」

# または異なる手法を試す
AsLS → AirPLS または Whittaker
```

### 問題12: スムージング後にピークが消える

**症状**: Savitzky-Golay適用後にピークが消失

**原因**: ウィンドウサイズが大きすぎる

**解決策**:

```text
# 現在の設定
window = 51  # 大きすぎる！
polyorder = 3

# 推奨設定
window = 11  # ピーク幅の約1/3
polyorder = 2 または 3

# 経験則
ピーク幅が約30ポイント → window = 9-11
ピーク幅が約10ポイント → window = 5-7
```

**段階的調整**:
```
1. window = 5 で開始（ノイズ多め）
2. window = 7
3. window = 11
4. window = 15（滑らかだがピーク保持）
5. window = 21（過度に滑らか）

最適なバランスを見つける
```

### 問題13: 正規化が期待通りに機能しない

**症状**: 正規化後もスペクトルのスケールが大きく異なる

**診断**:

```text
# 正規化前後の統計を確認
データ → 統計 → 
各スペクトルの min, max, mean を確認
```

**問題と解決策**:

**問題A: 外れ値の影響**
```
1つの外れ値がスケーリングを歪める

解決策:
1. 外れ値を除去
   データ → フィルタ → 外れ値除去
2. ロバストな正規化を使用
   ベクトルノルム → 分位点正規化
```

**問題B: 間違った正規化手法**
```
目的に応じた選択:

探索的分析 → ベクトルノルム
定量分析 → SNV または MSC
比較分析 → 最大値正規化
```

**問題C: 負の値**
```
SNV は負の値があると失敗する可能性

解決策:
1. ベースライン補正を先に実行
2. または Min-Max正規化を使用
```

### 問題14: "Singular matrix"エラー

**症状**:
```
numpy.linalg.LinAlgError: Singular matrix
```

**原因**: 行列が特異（逆行列が存在しない）

**一般的な状況**:

1. **重複データ**
   ```
   解決策:
   データ → 重複を削除
   ```

2. **定数列（分散ゼロ）**
   ```text
   # すべてのスペクトルで同じ値を持つ波数
   
   解決策:
   前処理 → 特徴量選択 → 
   「低分散特徴量を除去」
   閾値: 0.001
   ```

3. **線形従属**
   ```
   いくつかの列が他の列の線形結合
   
   解決策:
   分析前にPCAで次元削減:
   前処理 → PCA → 
   n_components: 50（またはデータサイズに応じて）
   ```

4. **サンプル数 < 特徴量数**
   ```
   5サンプル vs 1000波数 → 問題
   
   解決策:
   - より多くのサンプルを収集
   - PCAで次元削減
   - 波数範囲を制限
   ```

---

(troubleshooting-analysis)=
## 分析エラー

### 問題15: PCAで"explained variance"が低い

**症状**: 最初の2成分で説明分散が50%未満

**原因と解決策**:

**原因A: データのノイズが多い**
```
解決策:
1. 前処理を改善
   - より強いスムージング
   - より良いベースライン補正
2. 外れ値を除去
3. より多くの成分を使用（5-10）
```

**原因B: データが複雑**
```
多くの独立した変動源がある

これは正常な場合もある:
- 生物学的サンプルの自然な変動
- 複雑な混合物
- 多数の化合物

対応:
- より多くの成分を使用
- 非線形手法を試す（UMAP）
```

**原因C: 前処理が不十分**
```
解決策:
1. ベースライン補正を確認
2. 正規化を確認
3. スケーリングを追加
```

### 問題16: クラスタリングで意味のない結果

**症状**: すべてのサンプルが1つのクラスターに、または各サンプルが別々のクラスター

**診断**:

```
1. スコアプロット（PCA/UMAP）を確認
   → 視覚的な分離があるか？

2. デンドログラム（階層的）を確認
   → 自然なグループがあるか？

3. シルエットスコアを確認
   → スコア > 0.5 が良好
```

**解決策**:

**問題A: パラメータが不適切**
```
K-means:
- k が大きすぎる → 減らす
- k が小さすぎる → 増やす
→ エルボー法で最適なkを決定

DBSCAN:
- eps が大きすぎる → すべて1クラスター
- eps が小さすぎる → すべて外れ値
→ k-距離プロットで調整
```

**問題B: 前処理が不適切**
```
クラスタリングはスケールに敏感

解決策:
1. 適切な正規化（StandardScaler）
2. PCAで前処理（オプション）
```

**問題C: データが実際にクラスターを持たない**
```
すべてのサンプルが連続的な分布

対応:
- 階層的クラスタリングを使用
- デンドログラムで構造を確認
- 連続的な変動として扱う
```

### 問題17: 統計検定で"すべてのp値が有意"

**症状**: 数千の波数すべてでp < 0.05

**原因**: 多重比較問題

**解決策**:

```
分析 → 統計検定 → 
多重比較補正を有効化:

オプション1: Bonferroni（保守的）
  α_adjusted = 0.05 / n_tests
  例: 1000検定 → α = 0.00005

オプション2: FDR（False Discovery Rate）
  より緩やか、より多くの発見
  q値 < 0.05 を使用

オプション3: FWERコントロール
  Holm-Bonferroni法
```

**結果の解釈**:
```
補正後:
- 有意な波数が少数 → 真の差
- 依然として多数が有意 → 強い差がある
- 有意な波数なし → 差があるがわずか
```

---

(troubleshooting-ml)=
## 機械学習問題

### 問題18: モデルの精度が非常に低い（<60%）

**症状**: トレーニング精度もテスト精度も低い → **過小適合**

**診断**:

```text
# 学習曲線を確認
機械学習 → 評価 → 学習曲線

両方のスコアが低く平坦 → 過小適合
```

**解決策**:

**ステップ1: 前処理を改善**
```
現在の前処理を確認:
- ベースライン補正は適用済みか？
- 正規化は適用済みか？
- ノイズは除去したか？

推奨パイプライン:
1. AsLS (lambda=100000)
2. Savitzky-Golay (window=11)
3. SNV
4. （オプション）一次微分
```

**ステップ2: より複雑なモデルを使用**
```
現在: ロジスティック回帰（線形）
→ Random Forest または XGBoost（非線形）

または
SVM でカーネルを変更:
linear → rbf または poly
```

**ステップ3: 特徴量を増やす**
```
現在: PCAで10成分のみ
→ 50-100成分に増やす

または PCA を使用しない
```

**ステップ4: ハイパーパラメータを調整**
```
機械学習 → 設定 → 
グリッドサーチを有効化

Random Forest:
- n_estimators: 100 → 500
- max_depth: None（無制限）
- min_samples_split: 2 → 5
```

### 問題19: トレーニング精度99%、テスト精度60% → 過学習

**症状**: トレーニングとテストの大きな差

**診断**:

```
過学習の兆候:
✓ 訓練精度 >> テスト精度
✓ 学習曲線: 訓練誤差減少、検証誤差増加
✓ CV スコアの分散が大きい
```

**解決策**:

**ステップ1: データを増やす**
```
最も効果的だが、常に可能とは限らない

可能なら:
- より多くのサンプルを収集
- データ拡張（ノイズ追加など）
```

**ステップ2: 正則化を増やす**
```
Random Forest:
- max_depth を制限: None → 10
- min_samples_leaf を増やす: 1 → 5
- max_features を減らす: 'auto' → 'sqrt'

SVM:
- C を減らす: 1.0 → 0.1
- gamma を減らす: 'auto' → 0.001

XGBoost:
- reg_lambda を増やす: 0 → 1.0
- reg_alpha を増やす: 0 → 0.5
- max_depth を制限: 6 → 3
```

**ステップ3: 特徴量を減らす**
```
PCA で次元削減:
1000波数 → 50主成分

または特徴量選択:
- 分散閾値法
- 相互情報量
- RFE（再帰的特徴削減）
```

**ステップ4: クロスバリデーションを使用**
```
機械学習 → 設定 → 
クロスバリデーション: 5-fold

結果が安定していることを確認
```

**ステップ5: 早期停止（XGBoost/NNの場合）**
```
XGBoost設定:
early_stopping_rounds = 10
eval_metric = 'mlogloss'
eval_set = [(X_val, y_val)]
```

### 問題20: "ValueError: Found input variables with inconsistent numbers of samples"

**症状**:
```
ValueError: Found input variables with inconsistent numbers of samples: [100, 80]
```

**原因**: Xとyのサンプル数が一致しない

**診断**:

```text
print(f"X shape: {X.shape}")  # (100, 1000)
print(f"y shape: {y.shape}")  # (80,)
→ サンプル数が一致しない！
```

**一般的な原因と解決策**:

**原因A: ラベルの欠落**
```
一部のサンプルにラベルがない

解決策:
1. データ → グループ → 
   すべてのサンプルがグループに属していることを確認

2. または欠落サンプルを除去
```

**原因B: フィルタリングの不一致**
```
データをフィルタしたが、ラベルは更新しなかった

解決策:
フィルタ操作後にデータとラベルを同期
```

**原因C: データ分割エラー**
```text
# 間違い
X_train, X_test = train_test_split(X, test_size=0.3)
y_train, y_test = train_test_split(y, test_size=0.3)
→ 異なるランダムシードで分割される

# 正しい
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
```

### 問題21: XGBoostが非常に遅い

**症状**: トレーニングに数時間かかる

**原因と解決策**:

**原因A: tree_methodが最適でない**
```text
# 遅い（デフォルト）
tree_method = 'auto'

# 高速（GPUがある場合）
tree_method = 'gpu_hist'

# 高速（CPUの場合）
tree_method = 'hist'

# アプリ内設定
機械学習 → XGBoost → 設定 → 
tree_method: 'hist'
```

**原因B: ハイパーパラメータ探索が広すぎる**
```text
# グリッドサーチの範囲を制限
機械学習 → 設定 → グリッドサーチ → 
パラメータ数を減らす

# 例
n_estimators: [50, 100]  # [50, 100, 200, 500] の代わりに
max_depth: [3, 5]        # [3, 5, 7, 10] の代わりに
```

**原因C: データサイズが大きい**
```
解決策:
1. PCAで特徴量を削減: 1000 → 100
2. サンプリング: 全データの80%でトレーニング
3. 早期停止を使用
```

**原因D: n_estimatorsが大きすぎる**
```text
# 現在
n_estimators = 1000  # 遅い

# 推奨（早期停止と組み合わせ）
n_estimators = 100-300
early_stopping_rounds = 10
```

---

(troubleshooting-performance)=
## パフォーマンス問題

### 問題22: UI がフリーズする

**症状**: 処理中にUIが応答しない

**原因**: メインスレッドでの重い計算

**回避策**:

```
設定 → パフォーマンス → 
「バックグラウンド処理を使用」にチェック
```

**一時的な解決策**:
```
- 処理の進行を待つ（進行状況バーが表示される）
- 強制終了しない（データが失われる可能性）
```

**恒久的な解決策**:
```
1. より小さなデータセットでテスト
2. プレビューを無効化
3. より高速なアルゴリズムを使用
```

### 問題23: メモリ使用量が増え続ける

**症状**: アプリが時間とともに遅くなる

**原因**: メモリリーク

**診断**:

```text
# メモリ使用量を監視
# Linux
watch -n 1 'ps aux | grep python'

# Windows (PowerShell)
while($true) {
    Get-Process python | Select-Object WS; 
    Start-Sleep 1
}
```

**解決策**:

**短期的**:
```
定期的にアプリを再起動
ファイル → 再起動
```

**長期的**:
```
1. 不要なデータをクリア
   データ → クリア → 
   「前処理済みデータをクリア」

2. 結果をエクスポートしてからクリア

3. キャッシュをクリア
   設定 → キャッシュ → クリア

4. ガベージコレクションを強制
   設定 → パフォーマンス → 
   「定期的なガベージコレクション」
```

### 問題24: 大きなファイルのエクスポートが遅い

**症状**: Excelエクスポートに10分以上かかる

**解決策**:

**方法1: より高速なフォーマットを使用**
```
Excel → CSV に変更
（10倍高速）
```

**方法2: エクスポートを分割**
```
1. データを複数のファイルに分割
   - data_part1.xlsx (1-1000行)
   - data_part2.xlsx (1001-2000行)

2. または必要なシートのみエクスポート
```

**方法3: 圧縮を無効化**
```
エクスポート → オプション → 
「圧縮を無効化」にチェック
```

**方法4: HDF5を使用（大規模データ）**
```
エクスポート → HDF5 → 
（Python/MATLABでの再読み込みに最適）
```

---

(troubleshooting-ui)=
## UI/表示問題

### 問題25: 図が表示されない

**症状**: 空白の図エリア

**診断**:

```text
# Matplotlibバックエンドを確認
python -c "import matplotlib; print(matplotlib.get_backend())"

# 期待される出力: Qt5Agg または Qt6Agg
```

**解決策**:

**方法1: バックエンドを変更**
```text
# matplotlibrcファイルを編集
# Linux/macOS: ~/.matplotlib/matplotlibrc
# Windows: %USERPROFILE%\.matplotlib\matplotlibrc

backend: Qt6Agg

# またはコード内
import matplotlib
matplotlib.use('Qt6Agg')
```

**方法2: ソフトウェアレンダリング**
```
設定 → グラフィックス → 
「ソフトウェアレンダリングを使用」
```

**方法3: Matplotlibを再インストール**
```bash
pip uninstall matplotlib
pip install matplotlib
```

### 問題26: フォントが正しく表示されない

**症状**: 文字化けまたは□が表示される

**原因**: 必要なフォントがない

**解決策**:

**Windows**:
```
# 日本語フォントをインストール
設定 → 時刻と言語 → 言語 → 
日本語 → オプション → フォントを追加
```

**macOS**:
```bash
# システムフォントを使用
設定 → フォント → 
「システムフォントを使用」
```

**Linux**:
```bash
# 日本語フォントをインストール
sudo apt-get install fonts-noto-cjk

# フォントキャッシュを更新
fc-cache -fv

# Matplotlibキャッシュをクリア
rm -rf ~/.cache/matplotlib
```

**アプリ内設定**:
```
設定 → 外観 → フォント → 
日本語対応フォントを選択:
- Noto Sans CJK JP
- MS Gothic (Windows)
- Hiragino Sans (macOS)
```

### 問題27: ハイDPI画面で表示が小さい

**症状**: 4Kモニターでテキストやボタンが小さすぎる

**解決策**:

**Windows**:
```
# アプリケーション設定
設定 → 外観 → スケーリング → 
150% または 200%

# またはシステム設定
設定 → システム → ディスプレイ → 
拡大/縮小レイアウト: 150%
```

**macOS**:
```
# Retinaディスプレイで自動的に調整されるはず

# 問題がある場合:
システム環境設定 → ディスプレイ → 
解像度: スケーリング
```

**Linux**:
```bash
# 環境変数を設定
export QT_SCALE_FACTOR=1.5

# または ~/.bashrc に追加
echo 'export QT_SCALE_FACTOR=1.5' >> ~/.bashrc

# アプリ内
設定 → 外観 → スケーリング → 150%
```

---

(troubleshooting-export)=
## エクスポート問題

### 問題28: Excelエクスポートが失敗する

**症状**: "Failed to export to Excel"

**診断**:

```text
# openpyxl がインストールされているか確認
pip list | grep openpyxl

# インストールされていない場合
pip install openpyxl
```

**一般的な問題**:

**問題A: Excelファイルが開いている**
```
エラー: Permission denied

解決策:
1. Excelを閉じる
2. 別の名前で保存
```

**問題B: パスに特殊文字**
```
ファイル名: results (2).xlsx
→ 括弧が問題を引き起こす可能性

解決策:
英数字とアンダースコアのみ使用:
results_2.xlsx
```

**問題C: ディスク容量不足**
```
確認:
# Linux/macOS
df -h

# Windows
wmic logicaldisk get size,freespace,caption

解決策:
不要なファイルを削除、または別のドライブに保存
```

### 問題29: 図のエクスポート品質が悪い

**症状**: 保存したPNG画像がぼやけている

**解決策**:

**方法1: DPIを上げる**
```
図を右クリック → 画像として保存 → 
DPI: 300（印刷用）または 600（高品質）

デフォルト: 100（画面用）
```

**方法2: ベクトル形式を使用**
```
フォーマット: PNG → PDF または SVG
（拡大縮小しても品質維持）
```

**方法3: サイズを大きくする**
```
図の設定 → サイズ → 
幅: 3000 pixels
高さ: 2000 pixels
```

**方法4: アンチエイリアスを有効化**
```
設定 → グラフィックス → 
「高品質レンダリング」にチェック
```

---

(troubleshooting-platform)=
## プラットフォーム固有の問題

### Windows固有

### 問題30: "VCRUNTIME140.dll が見つかりません"

**解決策**:
```
1. Visual C++ Redistributable をダウンロード
   https://aka.ms/vs/17/release/vc_redist.x64.exe

2. インストーラーを実行

3. アプリを再起動
```

### 問題31: Windows Defenderがブロック

**症状**: "WindowsによってPCが保護されました"

**解決策**:
```
1. 「詳細情報」をクリック
2. 「実行」をクリック

または

1. Windows Defender → 除外 → 
2. アプリケーションフォルダを追加
```

### macOS固有

### 問題32: "開発元を確認できないため開けません"

**解決策**:

```bash
# ターミナルで
sudo xattr -cr /Applications/RamanApp.app

# または
sudo xattr -rd com.apple.quarantine /Applications/RamanApp.app
```

その後:
```
システム環境設定 → セキュリティとプライバシー → 
「このまま開く」をクリック
```

### 問題33: macOS Big Sur以降でクラッシュ

**原因**: Rosetta 2の問題（M1/M2 Macの場合）

**解決策**:

```bash
# Rosetta 2をインストール
softwareupdate --install-rosetta

# またはネイティブARMバージョンを使用
pip install raman-app  # 自動的にARM版をインストール
```

### Linux固有

### 問題34: "libGL error"エラー

**症状**:
```
libGL error: No matching fbConfigs or visuals found
libGL error: failed to load driver: swrast
```

**解決策**:

```bash
# Ubunt/Debian
sudo apt-get install libgl1-mesa-glx libgl1-mesa-dri

# Fedora
sudo dnf install mesa-libGL mesa-dri-drivers

# Arch
sudo pacman -S mesa
```

### 問題35: Wayland vs X11の問題

**症状**: アプリが起動しない（Wayland使用時）

**解決策**:

```bash
# X11を強制使用
export QT_QPA_PLATFORM=xcb
python main.py

# または ~/.bashrc に追加
echo 'export QT_QPA_PLATFORM=xcb' >> ~/.bashrc
```

---

## 🆘 それでも解決しない場合

### ログの収集

```bash
# 1. ログファイルを見つける
# Windows
type %APPDATA%\RamanApp\logs\app.log

# macOS
cat ~/Library/Logs/RamanApp/app.log

# Linux
cat ~/.local/share/RamanApp/logs/app.log

# 2. 詳細ログを有効化
python main.py --debug --log-level DEBUG > debug.log 2>&1
```

### バグレポートの作成

GitHub Issueを作成する際に含めるべき情報:

```markdown
**環境**
- OS: [Windows 11 / macOS 13 / Ubuntu 22.04]
- Pythonバージョン: [3.10.8]
- アプリバージョン: [1.0.0]
- インストール方法: [pip / 実行可能ファイル / ソース]

**問題の説明**
何が起こったかを明確に説明

**再現手順**
1. データをロード
2. AsLS を適用
3. PCA を実行
4. → エラー発生

**期待される動作**
何が起こるべきだったか

**実際の動作**
何が実際に起こったか

**エラーメッセージ**
```
完全なエラーメッセージを貼り付け
```

**ログ**
```
関連するログエントリを貼り付け
```

**スクリーンショット**
可能であれば添付
```

### サポートを受ける

- **GitHub Issues**: https://github.com/your-org/raman-app/issues
- **GitHub Discussions**: https://github.com/your-org/raman-app/discussions
- **Email**: support@example.com

---

## 📚 関連リソース

- **[FAQ](faq.md)** - よくある質問
- **[ユーザーガイド](user-guide/index.md)** - 完全な使用ガイド
- **[インストールガイド](installation.md)** - 詳細なインストール手順
- **[開発ガイド](dev-guide/index.md)** - 開発者向け情報

---

**最終更新**: 2026年1月24日 | **バージョン**: 1.0.0
