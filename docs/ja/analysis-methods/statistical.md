# 統計分析（日本語版）

このページでは、アプリ内で利用できる統計手法の概要を説明します。

## 目次

- [ANOVA（分散分析）](#anova分散分析)
- [補足: 多重検定（Multiple testing）](#補足-多重検定multiple-testing)

---

## ANOVA（分散分析）

**目的**: 3 つ以上のグループ間で平均の差を検定します。

**このアプリでの ANOVA**:

- ANOVA は **各波数（wavenumber）ごと** に実行されます（= 波数点の数だけ検定が発生します）。
- そのため、アプリでは **多重検定補正**（例: FDR）を選択できます。

### モード（Simple / Grouped）

- **Simple モード**: 選択した各 **データセット** を 1 グループとして扱います。
- **Grouped モード**: データセットをユーザー定義のグループへ割り当て、**グループラベル**単位で ANOVA を行います（ML ページと同じグループ設定を共有します）。

### パラメータ

| パラメータ                |   既定値 | 説明                                         |
| ------------------------- | -------: | -------------------------------------------- |
| `alpha`                   |     0.05 | 有意水準（閾値）                             |
| `p_adjust`                | `fdr_bh` | 多重検定補正: `none`, `fdr_bh`, `bonferroni` |
| `post_hoc`                |   `none` | 追加検定（任意）: `none`, `tukey`            |
| `max_posthoc_wavenumbers` |       20 | 追加検定の波数点数上限（0 で無効）           |
| `show_mean_overlay`       |     True | グループ平均スペクトルの重ね描き表示         |
| `highlight_significant`   |     True | 有意な波数領域のハイライト表示               |

### 使い分けの目安

- **2 グループのみ**の場合: 「Pairwise Statistical Tests（ペア検定）」の利用が推奨です。
- **3 グループ以上**の場合: ANOVA を利用し、多重検定補正（例: FDR）を併用すると安全です。

---

## 補足: 多重検定（Multiple testing）

波数ごとに検定を行うと、検定回数が多くなり、偶然の有意（偽陽性）が増えやすくなります。

- `p_adjust = fdr_bh`（FDR / Benjamini–Hochberg）: 偽発見率を制御しつつ感度を保ちやすい（推奨）
- `p_adjust = bonferroni`: 非常に保守的（有意になりにくい）
- `p_adjust = none`: 補正なし（探索目的以外では注意）

---

英語版（詳細・最新）: https://raman-spectroscopy-analysis-application.readthedocs.io/en/latest/analysis-methods/statistical.html
