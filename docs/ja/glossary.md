# 用語集

ラマン分光とデータ解析の用語辞典

---

## A

### AsLS（Asymmetric Least Squares）
非対称最小二乗法によるベースライン補正アルゴリズム。蛍光バックグラウンドの除去に広く使用される。

**主要パラメータ**:
- lambda（λ）: スムーズネス（10²～10⁸）
- p: 非対称性（0.001～0.5）

**関連**: Baseline Correction, AirPLS

---

### AirPLS（Adaptive Iteratively Reweighted Penalized Least Squares）
適応的反復重み付きペナルティ付き最小二乗法。AsLSの改良版で、より複雑なベースラインに対応。

**特徴**: 自動的にベースラインを推定、パラメータ調整が簡単

**関連**: AsLS, Baseline Correction

---

### ANOVA（Analysis of Variance）
分散分析。3つ以上のグループ間の平均値の差を統計的に検定する手法。

**帰無仮説（H₀）**: すべてのグループの平均が等しい  
**対立仮説（H₁）**: 少なくとも1つのグループが異なる

**事後検定**: Tukey HSD、Bonferroni、Dunnett

**関連**: t-test, Post-hoc Test, Multiple Comparison Correction

---

### API（Application Programming Interface）
アプリケーションプログラミングインターフェース。プログラムから本アプリの機能を利用するためのインターフェース。

**用途**: 自動化、バッチ処理、カスタムワークフロー

**関連**: Python API, Batch Processing

---

## B

### Baseline Correction
ベースライン補正。蛍光バックグラウンドや装置由来のオフセットを除去する前処理手法。

**主な手法**:
- AsLS
- AirPLS
- Whittaker
- Rolling Ball

**目的**: ピークの明瞭化、定量分析の精度向上

**関連**: AsLS, AirPLS, Fluorescence

---

### Batch Processing
バッチ処理。複数のファイルに対して同じ処理を自動的に適用すること。

**例**: 100個のスペクトルに同じ前処理パイプラインを適用

**利点**: 効率性、一貫性、再現性

**関連**: Pipeline, Automation

---

### Biplot
バイプロット。PCAのスコアプロットとローディングプロットを重ねて表示したグラフ。

**表示内容**:
- 点: サンプル（スコア）
- 矢印: 変数（ローディング）

**解釈**: サンプルの配置と、それに寄与する変数の関係を同時に可視化

**関連**: PCA, Score Plot, Loading Plot

---

### Bonferroni Correction
ボンフェローニ補正。多重比較における偽陽性を制御する統計手法。

**補正後の有意水準**: α_adjusted = α / n_tests

**例**: 1000回の検定、α=0.05  
→ α_adjusted = 0.05 / 1000 = 0.00005

**特徴**: 保守的（偽陽性は減るが、偽陰性が増える）

**関連**: FDR, Multiple Comparison Correction

---

## C

### CDAE（Convolutional Denoising Autoencoder）
畳み込みデノイジングオートエンコーダー。深層学習を用いたノイズ除去手法。

**特徴**: 非線形ノイズの除去、複雑なパターンの学習

**用途**: 非常にノイズの多いデータ、従来手法で不十分な場合

**関連**: Smoothing, Deep Learning

---

### Clustering
クラスタリング。類似したサンプルをグループ化する教師なし学習手法。

**主な手法**:
- K-means
- Hierarchical Clustering
- DBSCAN

**用途**: データの構造発見、グループ化、外れ値検出

**関連**: K-means, Hierarchical Clustering, DBSCAN

---

### Cohen's d
コーエンのd。効果量の一種で、2グループ間の平均値の差を標準化した指標。

**計算式**: d = (mean₁ - mean₂) / pooled_std

**解釈**:
- |d| < 0.2: 小さい
- 0.2 ≤ |d| < 0.5: 中程度
- 0.5 ≤ |d| < 0.8: 大きい
- |d| ≥ 0.8: 非常に大きい

**関連**: Effect Size, t-test, Eta-squared

---

### Confusion Matrix
混同行列。分類モデルの予測結果を実際のラベルと比較した行列。

**構成要素**:
- TP（True Positive）: 真陽性
- TN（True Negative）: 真陰性
- FP（False Positive）: 偽陽性
- FN（False Negative）: 偽陰性

**派生メトリクス**: Accuracy、Precision、Recall、F1-score

**関連**: Accuracy, Precision, Recall, F1-score

---

### Cross-Validation（CV）
交差検証。データを複数の部分集合に分割し、各部分を順にテストセットとして使用する評価手法。

**k-fold CV**: データをk個に分割

**利点**: より信頼性の高い評価、過学習の検出

**例**: 5-fold CV → 5回の評価の平均

**関連**: Train/Test Split, Overfitting

---

## D

### DBSCAN（Density-Based Spatial Clustering of Applications with Noise）
密度ベースのクラスタリング手法。

**特徴**:
- クラスター数の事前指定不要
- 任意の形状のクラスター検出
- 外れ値の自動検出

**主要パラメータ**:
- eps: 近傍の半径
- min_samples: コアポイントの最小サンプル数

**関連**: K-means, Clustering, Outlier Detection

---

### Dendrogram
デンドログラム（樹形図）。階層的クラスタリングの結果を樹状に表示したグラフ。

**解釈**:
- 縦軸: 距離（類似度）
- 低い位置での結合: 類似度が高い
- 高い位置での結合: 類似度が低い

**活用**: 切断線の設定によりクラスター数を決定

**関連**: Hierarchical Clustering

---

### Dimensionality Reduction
次元削減。高次元データをより低い次元に変換する手法。

**目的**: 可視化、計算効率の向上、ノイズ削減

**主な手法**:
- PCA（線形）
- UMAP（非線形）
- t-SNE（非線形）

**関連**: PCA, UMAP, t-SNE

---

## E

### Effect Size
効果量。統計的検定における差の大きさを表す指標。p値とは独立。

**重要性**: p値は有意性を示すが、実際の差の大きさは示さない

**主な指標**:
- Cohen's d（t検定用）
- Eta-squared（ANOVA用）
- Pearson's r（相関用）

**関連**: Cohen's d, Eta-squared, Statistical Significance

---

### Elbow Method
エルボー法。K-meansクラスタリングにおける最適なクラスター数を決定する手法。

**手順**:
1. kを変えながらクラスタリング
2. 各kでの歪み（WCSS）を計算
3. 歪みの減少が緩やかになる点（エルボー）を探す

**解釈**: エルボーポイント = 最適なk

**関連**: K-means, Silhouette Analysis

---

### EMSC（Extended Multiplicative Signal Correction）
拡張乗法的信号補正。散乱効果を補正する高度な前処理手法。

**特徴**: 物理的光学モデルに基づく補正

**用途**: 粉末試料、不均一試料、散乱の影響が大きい場合

**関連**: MSC, SNV, Scattering

---

### Eta-squared（η²）
エータ二乗。ANOVAにおける効果量。

**計算式**: η² = SS_between / SS_total

**解釈**:
- η² < 0.01: 小さい
- 0.01 ≤ η² < 0.06: 中程度
- 0.06 ≤ η² < 0.14: 大きい
- η² ≥ 0.14: 非常に大きい

**意味**: グループ間の差で説明される分散の割合

**関連**: ANOVA, Effect Size, Cohen's d

---

## F

### F1-score
F1スコア。PrecisionとRecallの調和平均。

**計算式**: F1 = 2 × (Precision × Recall) / (Precision + Recall)

**範囲**: 0～1（1が最良）

**用途**: 不均衡データでの分類性能評価

**関連**: Precision, Recall, Accuracy

---

### FABC（Fast Adaptive Baseline Correction）
高速適応ベースライン補正。計算効率の良いベースライン補正手法。

**特徴**: 高速、自動パラメータ調整

**用途**: リアルタイム処理、大規模データ

**関連**: Baseline Correction, AsLS

---

### False Discovery Rate（FDR）
偽発見率。多重比較において偽陽性の割合を制御する手法。

**Benjamini-Hochberg法**: よく使われるFDR制御法

**q値**: FDR制御後のp値に相当

**特徴**: Bonferroni補正より緩やか（発見力が高い）

**関連**: Bonferroni Correction, Multiple Comparison Correction

---

### Feature Engineering
特徴量エンジニアリング。元データから新しい特徴量を作成する手法。

**ラマン分光での例**:
- ピーク高さ
- ピーク面積
- ピーク比
- 統計量（平均、標準偏差）

**目的**: モデル性能の向上、物理的解釈の追加

**関連**: Feature Selection, Machine Learning

---

### Feature Importance
特徴量の重要度。機械学習モデルにおける各特徴量（波数）の寄与度。

**表示**: 棒グラフ（各波数の重要度）

**用途**:
- 重要な波数の特定
- 化学的解釈
- 特徴量選択

**対応アルゴリズム**: Random Forest、XGBoost

**関連**: Feature Selection, Random Forest, XGBoost

---

### Feature Selection
特徴量選択。重要でない特徴量を除去し、モデルを簡素化する手法。

**主な手法**:
- Variance Threshold（分散閾値）
- RFE（再帰的特徴量削除）
- Feature Importance ベース

**利点**: 過学習の防止、計算時間の短縮、解釈性の向上

**関連**: Feature Importance, RFE

---

### Fluorescence
蛍光。試料が励起光を吸収し、より長波長の光を放出する現象。

**ラマン分光での問題**: 蛍光バックグラウンドがラマン信号を覆い隠す

**対策**: ベースライン補正、長波長励起（1064nm）、時間分解測定

**関連**: Baseline Correction, AsLS

---

## G

### Gap Statistic
ギャップ統計量。クラスター数を決定するための統計指標。

**原理**: 実データのクラスタリングとランダムデータのクラスタリングを比較

**最適なk**: Gap statisticが最大となるk

**利点**: 統計的根拠のあるクラスター数の決定

**関連**: K-means, Elbow Method, Silhouette Analysis

---

### Gaussian Smoothing
ガウシアンスムージング。ガウス関数を用いた平滑化手法。

**特徴**: エッジの保持が良好、パラメータ（sigma）が直感的

**用途**: ノイズ除去、ピーク形状の保持

**関連**: Smoothing, Savitzky-Golay

---

### Grid Search
グリッドサーチ。ハイパーパラメータの最適な組み合わせを網羅的に探索する手法。

**手順**:
1. 各パラメータの候補値を定義
2. すべての組み合わせを評価
3. 最良の組み合わせを選択

**欠点**: 計算コストが高い（組み合わせ数の爆発）

**関連**: Random Search, Hyperparameter Tuning

---

## H

### Hierarchical Clustering
階層的クラスタリング。サンプル間の階層構造を構築するクラスタリング手法。

**手順**:
1. 各サンプルを個別のクラスターとする
2. 最も近い2つのクラスターを結合
3. 繰り返す

**結果**: デンドログラム

**リンケージ法**: Ward、Average、Complete、Single

**関連**: Dendrogram, Clustering

---

### Hyperparameter Tuning
ハイパーパラメータ調整。機械学習モデルのハイパーパラメータ（学習前に設定する必要があるパラメータ）を最適化するプロセス。

**主な手法**:
- Grid Search
- Random Search
- Bayesian Optimization

**関連**: Grid Search, Random Search

---

## K

### K-means
K平均法。最も基本的なクラスタリング手法。

**アルゴリズム**:
1. k個の中心をランダムに初期化
2. 各サンプルを最も近い中心に割り当て
3. 各クラスターの中心を再計算
4. 収束まで繰り返す

**主要パラメータ**: k（クラスター数）

**制約**: 球形のクラスターを仮定

**関連**: Clustering, Elbow Method, Silhouette Analysis

---

## L

### Learning Curve
学習曲線。トレーニングサンプル数に対するモデル性能をプロットしたグラフ。

**表示**:
- X軸: トレーニングサンプル数
- Y軸: スコア
- 2本の曲線: トレーニングスコア、検証スコア

**診断**:
- 過学習: 訓練高、検証低
- 過小適合: 両方低

**関連**: Overfitting, Underfitting, Validation Curve

---

### Loading Plot
ローディングプロット。PCAにおける主成分の各変数（波数）への寄与を示すグラフ。

**表示**:
- X軸: 波数
- Y軸: ローディング値

**解釈**:
- 正の値: その波数がサンプルのスコアを正の方向に動かす
- 負の値: 負の方向に動かす
- ゼロ: 寄与しない

**関連**: PCA, Score Plot

---

### Logistic Regression
ロジスティック回帰。分類問題のための線形モデル。

**出力**: 各クラスの確率（0～1）

**特徴**: 高速、解釈しやすい、線形分離

**主要パラメータ**: C（正則化の強さの逆数）

**関連**: Linear Regression, Machine Learning

---

## M

### Machine Learning
機械学習。データからパターンを学習し、予測を行う手法の総称。

**種類**:
- 教師あり学習（分類、回帰）
- 教師なし学習（クラスタリング、次元削減）
- 強化学習

**ラマン分光での応用**:
- 化合物の識別
- 濃度予測
- 品質管理
- 異常検出

**関連**: Supervised Learning, Unsupervised Learning

---

### Min-Max Normalization
最小-最大正規化。データを指定した範囲（通常0～1）にスケーリングする手法。

**計算式**: X_norm = (X - X_min) / (X_max - X_min)

**特徴**: すべての値が[0, 1]の範囲に収まる

**用途**: 機械学習の前処理、可視化

**関連**: Normalization, Vector Norm

---

### MSC（Multiplicative Scatter Correction）
乗法的散乱補正。光散乱の影響を補正する前処理手法。

**原理**: 各スペクトルを参照スペクトルに対してスケーリング

**用途**: 粉末試料、不均一試料

**関連**: SNV, EMSC, Scattering

---

### Multiple Comparison Correction
多重比較補正。複数の統計検定を行う際の偽陽性を制御する手法。

**問題**: 検定回数が増えると偽陽性の確率が増加

**主な手法**:
- Bonferroni補正（保守的）
- FDR（Benjamini-Hochberg法）

**関連**: Bonferroni Correction, FDR

---

## N

### Neural Network
ニューラルネットワーク。脳の神経回路を模したモデル。

**構成要素**: ニューロン（ノード）、重み、活性化関数

**学習**: バックプロパゲーション

**種類**: CNN、RNN、Transformer（多層パーセプトロンMLPは将来のリリースで予定）

**関連**: Deep Learning

---

### Normalization
正規化。スペクトルの強度スケールを統一する前処理手法。

**主な手法**:
- Vector Norm（L2）
- Min-Max
- Max Normalization
- SNV

**目的**: 強度のばらつきの影響を除去、スペクトル間の比較を容易に

**関連**: Vector Norm, SNV, Min-Max Normalization

---

## O

### Outlier Detection
外れ値検出。他のサンプルと大きく異なるサンプルを特定する手法。

**主な手法**:
- 統計的手法（Z-score、IQR）
- 距離ベース（Isolation Forest）
- PCAベース
- DBSCANによるノイズ検出

**用途**: データクリーニング、品質管理、異常検出

**関連**: PCA, DBSCAN, Quality Control

---

### Overfitting
過学習（過適合）。モデルがトレーニングデータに過度に適合し、新しいデータへの汎化性能が低下する現象。

**兆候**:
- トレーニング精度が非常に高い（>95%）
- テスト精度が低い（<80%）
- 大きなギャップ

**対策**:
- より多くのデータ
- 正則化
- クロスバリデーション
- 早期停止

**関連**: Underfitting, Regularization, Cross-Validation

---

## P

### PCA（Principal Component Analysis）
主成分分析。データの分散が最大となる方向（主成分）を見つける次元削減手法。

**数式**: X = TPᵀ + E

**出力**:
- スコア（T）: サンプルの新しい座標
- ローディング（P）: 主成分の方向
- 説明分散: 各主成分が説明する分散の割合

**用途**: 可視化、ノイズ削減、特徴量削減

**関連**: Score Plot, Loading Plot, Scree Plot

---

### Peak Detection
ピーク検出。スペクトル中のピーク（極大値）を自動的に検出する手法。

**主要パラメータ**:
- height: 最小ピーク高さ
- distance: ピーク間の最小距離
- prominence: 卓立度
- width: 最小ピーク幅

**用途**: ピーク位置の特定、ピーク数のカウント、ピークの比較

**関連**: Peak Fitting, Peak Identification

---

### Peak Fitting
ピークフィッティング。検出されたピークに数学的関数を当てはめる手法。

**関数**:
- Gaussian（ガウス）
- Lorentzian（ローレンツ）
- Voigt（フォークト）
- Pseudo-Voigt

**推定パラメータ**: 位置、高さ、幅、面積

**用途**: 重なったピークの分離、定量分析

**関連**: Peak Detection

---

### Pipeline
パイプライン。複数の前処理ステップを順番に適用するワークフロー。

**例**:
1. ベースライン補正（AsLS）
2. スムージング（Savitzky-Golay）
3. 正規化（Vector Norm）

**利点**: 再現性、一貫性、効率性

**保存**: .pipeline.json形式

**関連**: Preprocessing, Workflow

---

### Post-hoc Test
事後検定。ANOVAで有意差が検出された後、どのグループ間に差があるかを特定する検定。

**主な手法**:
- Tukey HSD（すべてのペア）
- Bonferroni（保守的）
- Dunnett（対照群との比較）

**関連**: ANOVA, Multiple Comparison Correction

---

### Precision
適合率（精度）。陽性と予測したもののうち、実際に陽性だった割合。

**計算式**: Precision = TP / (TP + FP)

**解釈**: 高い精度 = 偽陽性が少ない

**用途**: 偽陽性のコストが高い場合に重視

**関連**: Recall, F1-score, Confusion Matrix

---

### Preprocessing
前処理。分析前にデータをクリーニング・変換するプロセス。

**主なステップ**:
1. ベースライン補正
2. スムージング
3. 正規化
4. 微分（オプション）

**目的**: ノイズ削減、比較可能性の向上、特徴の強調

**関連**: Pipeline, Baseline Correction, Normalization

---

## R

### Random Forest
ランダムフォレスト。複数の決定木を組み合わせたアンサンブル学習手法。

**利点**:
- 高精度
- 過学習に強い
- 特徴量の重要度を提供

**主要パラメータ**:
- n_estimators: 木の数
- max_depth: 木の深さ

**関連**: Machine Learning, Feature Importance

---

### Random Search
ランダムサーチ。ハイパーパラメータをランダムにサンプリングして探索する手法。

**特徴**: Grid Searchより効率的、広い範囲を探索可能

**パラメータ**: n_iter（試行回数）

**関連**: Grid Search, Hyperparameter Tuning

---

### Recall
再現率（感度）。実際の陽性のうち、正しく検出できた割合。

**計算式**: Recall = TP / (TP + FN)

**解釈**: 高い再現率 = 偽陰性が少ない

**用途**: 偽陰性のコストが高い場合に重視（例: 疾病診断）

**関連**: Precision, F1-score

---

### Regularization
正則化。モデルの複雑さにペナルティを与え、過学習を防ぐ手法。

**種類**:
- L1正則化（Lasso）: スパース解
- L2正則化（Ridge）: 係数を小さく

**パラメータ**: alpha（正則化の強さ）

**関連**: Overfitting, Logistic Regression

---

### RFE（Recursive Feature Elimination）
再帰的特徴量削除。重要度の低い特徴量を順次削除する特徴量選択手法。

**手順**:
1. すべての特徴量でモデルを訓練
2. 最も重要度の低い特徴量を削除
3. 指定した数まで繰り返す

**関連**: Feature Selection, Feature Importance

---

### ROC Curve（Receiver Operating Characteristic Curve）
ROC曲線。分類器の性能を評価するグラフ。

**軸**:
- X軸: 偽陽性率（FPR）
- Y軸: 真陽性率（TPR = Recall）

**AUC（Area Under Curve）**: ROC曲線下の面積（0.5～1.0）

**解釈**:
- AUC = 0.5: ランダム
- AUC = 0.7-0.8: 良好
- AUC = 0.8-0.9: 優秀
- AUC > 0.9: 非常に優秀

**関連**: Precision, Recall, AUC

---

## S

### Savitzky-Golay
サビツキー・ゴレイフィルター。局所的な多項式フィッティングによるスムージング手法。

**主要パラメータ**:
- window_length: ウィンドウサイズ（奇数、5～31）
- polyorder: 多項式の次数（1～4）

**特徴**: ピーク形状の保持が良好

**関連**: Smoothing, Gaussian Smoothing

---

### Scattering
散乱。光が試料中の粒子や不均一性によって進行方向を変える現象。

**ラマン分光での影響**: ベースラインの歪み、強度の変化

**対策**: MSC、SNV、EMSC

**関連**: MSC, SNV, EMSC

---

### Score Plot
スコアプロット。PCAで得られたサンプルの座標（スコア）をプロットしたグラフ。

**表示**:
- 2Dプロット: PC1 vs PC2
- 3Dプロット: PC1 vs PC2 vs PC3

**解釈**:
- 近いサンプル: 類似
- 遠いサンプル: 異なる
- クラスターの形成

**関連**: PCA, Loading Plot

---

### Scree Plot
スクリープロット。PCAの各主成分が説明する分散の割合を示すグラフ。

**表示**:
- X軸: 主成分番号
- Y軸: 説明分散比率

**用途**: 適切な主成分数の決定（エルボーポイント）

**関連**: PCA, Elbow Method

---

### Silhouette Analysis
シルエット分析。クラスタリングの品質を評価する手法。

**シルエット係数**: -1～1の値

**解釈**:
- s > 0.5: 良好なクラスタリング
- 0.3 < s < 0.5: 弱いクラスタリング
- s < 0.3: 不適切なクラスタリング

**用途**: 最適なクラスター数の決定

**関連**: K-means, Elbow Method

---

### Smoothing
スムージング（平滑化）。ノイズを除去するための前処理手法。

**主な手法**:
- Savitzky-Golay
- Gaussian
- Moving Average

**目的**: ノイズ削減、ピークの明瞭化

**注意**: 過度なスムージングはピークを消失させる

**関連**: Savitzky-Golay, Gaussian Smoothing, Noise

---

### SNV（Standard Normal Variate）
標準正規変量。各スペクトルを平均0、標準偏差1に標準化する正規化手法。

**計算式**: X_snv = (X - mean(X)) / std(X)

**用途**: 散乱補正、強度変動の除去

**関連**: Normalization, MSC

---

### Supervised Learning
教師あり学習。ラベル（正解）付きデータからモデルを学習する機械学習の一種。

**種類**:
- 分類（クラスラベルの予測）
- 回帰（連続値の予測）

**主なアルゴリズム**: Random Forest、SVM、XGBoost、ロジスティック回帰

**関連**: Machine Learning, Unsupervised Learning

---

### SVM（Support Vector Machine）
サポートベクターマシン。高次元空間でクラスを分離する超平面を見つける機械学習手法。

**特徴**: 高次元データに強い、カーネルトリックで非線形分離

**主要パラメータ**:
- C: 正則化パラメータ
- kernel: 'linear', 'rbf', 'poly'
- gamma: RBFカーネルのパラメータ

**関連**: Machine Learning, Kernel Trick

---

## T

### t-SNE（t-Distributed Stochastic Neighbor Embedding）
t分布型確率的近傍埋め込み。非線形次元削減手法。

**特徴**: 優れた可視化能力、局所構造の保持

**主要パラメータ**:
- perplexity: 近傍の数（5～50）
- learning_rate: 学習率
- n_iter: 反復回数

**注意**: 距離やクラスターサイズに意味がない、再現性の問題

**関連**: PCA, UMAP, Dimensionality Reduction

---

### t-test
t検定。2つのグループ間の平均値の差を統計的に検定する手法。

**種類**:
- 独立2標本t検定
- 対応のあるt検定

**出力**:
- t統計量
- p値
- 自由度
- 信頼区間

**解釈**: p < 0.05で有意差あり（有意水準5%）

**関連**: ANOVA, Effect Size, Statistical Significance

---

### Train/Test Split
訓練/テスト分割。データを訓練用とテスト用に分割すること。

**推奨比率**:
- 70:30
- 80:20
- 60:20:20（訓練:検証:テスト）

**目的**: モデルの汎化性能を評価

**層化分割**: 各クラスの比率を保つ分割

**関連**: Cross-Validation, Overfitting

---

## U

### UMAP（Uniform Manifold Approximation and Projection）
一様多様体近似射影。非線形次元削減手法。

**特徴**: PCAより優れた非線形構造の保持、t-SNEより高速

**主要パラメータ**:
- n_neighbors: 局所構造のサイズ（5～50）
- min_dist: 埋め込み空間での最小距離（0～1）

**用途**: 複雑な非線形構造の可視化

**関連**: PCA, t-SNE, Dimensionality Reduction

---

### Underfitting
過小適合。モデルがデータのパターンを十分に学習できていない状態。

**兆候**:
- トレーニング精度が低い
- テスト精度も低い
- 両者のギャップが小さい

**対策**:
- より複雑なモデル
- 特徴量を増やす
- 正則化を弱める

**関連**: Overfitting, Model Complexity

---

### Unsupervised Learning
教師なし学習。ラベルなしデータから構造を発見する機械学習の一種。

**主な手法**:
- クラスタリング（K-means、階層的）
- 次元削減（PCA、UMAP）
- 異常検出

**関連**: Supervised Learning, Clustering, PCA

---

## V

### Validation Curve
検証曲線。ハイパーパラメータの値に対するモデル性能をプロットしたグラフ。

**表示**:
- X軸: パラメータ値
- Y軸: スコア
- 2本の曲線: トレーニングスコア、検証スコア

**用途**: 最適なパラメータ値の特定

**関連**: Learning Curve, Hyperparameter Tuning

---

### Vector Norm
ベクトルノルム正規化（L2正規化）。各スペクトルのL2ノルム（ユークリッドノルム）を1にする正規化手法。

**計算式**: X_norm = X / ||X||₂

**特徴**: スペクトルの形状を保持、強度の影響を除去

**用途**: 最も一般的な正規化手法

**関連**: Normalization, SNV

---

## W

### Wavenumber
波数。光の波長の逆数で、ラマンシフトを表す単位（cm⁻¹）。

**計算**: wavenumber = 1 / wavelength

**ラマン分光**: 励起波長からのエネルギーシフトを表す

**典型的な範囲**: 200～4000 cm⁻¹

**関連**: Raman Shift, Raman Spectroscopy

---

### Whittaker Smoothing
ウィッテカー平滑化。ペナルティ付き最小二乗法によるスムージング手法。

**特徴**: ベースライン補正とスムージングの両方に使用可能

**主要パラメータ**: lambda（スムーズネス）

**関連**: Smoothing, Baseline Correction

---

### Workflow
ワークフロー。データのロードから結果の出力までの一連の処理手順。

**典型的なワークフロー**:
1. データロード
2. 品質チェック
3. 前処理
4. 探索的分析
5. 統計解析または機械学習
6. 結果の解釈
7. レポート生成

**関連**: Pipeline, Best Practices

---

## X

### XGBoost（eXtreme Gradient Boosting）
勾配ブースティング決定木の高効率実装。

**特徴**: 最高レベルの精度、正則化機能、早期停止

**主要パラメータ**:
- n_estimators: 木の数
- max_depth: 木の深さ
- learning_rate: 学習率
- subsample: サンプルのサブサンプリング比率

**用途**: 分類、回帰、ランキング

**関連**: Random Forest, Machine Learning, Gradient Boosting

---

## Z

### Z-score
Zスコア。標準偏差を単位とした偏差値。

**計算式**: z = (x - mean) / std

**用途**: 外れ値の検出

**基準**: |z| > 3で外れ値

**関連**: Outlier Detection, Standardization

---

## 🔗 関連ドキュメント

- **[ユーザーガイド](index.md)** - アプリの使い方
- **[分析手法](analysis-methods/index.md)** - 各手法の詳細
- **[API リファレンス](api/index.md)** - プログラミングインターフェース
- **[FAQ](faq.md)** - よくある質問

---

**最終更新**: 2026年1月24日 | **バージョン**: 1.0.0
