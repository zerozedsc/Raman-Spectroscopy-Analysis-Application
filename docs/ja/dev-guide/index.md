# 開発者ガイド（日本語版）

このページは準備中です。

日本語の開発ドキュメントは順次整備しています。現時点で「確実に言える範囲」のみを記載します。

---

## ✅ 前提条件（現状）

- Python **3.12**（`pyproject.toml` の `requires-python` に準拠）
- Git

---

## 🚀 最小クイックスタート（from source）

1) リポジトリを取得

```bash
git clone https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application.git
cd Raman-Spectroscopy-Analysis-Application
```

2) 仮想環境を作成して有効化

```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

3) 依存関係をインストール（開発用オプションは任意）

```bash
pip install -e .
# 開発用（pytest / black / watchdog など）
pip install -e .[dev]
```

4) 起動

```bash
python main.py
```

---

## 🧪 簡易テスト（スモークテスト）

このリポジトリには `smoke_tests.py` があり、主要な描画・可視化経路が import 可能でクラッシュしないことを確認できます。

```bash
python smoke_tests.py
```

---

## 📌 関連ページ（日本語）

- [アーキテクチャ](architecture.md)
- [貢献ガイド](contributing.md)
- [ビルドシステム](build-system.md)
- [テストガイド](testing.md)

---

## 🔗 外部リンク

- 英語版（最新）: https://raman-spectroscopy-analysis-application.readthedocs.io/en/latest/dev-guide/index.html

---

## 🆘 サポート

- GitHub Discussions: https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/discussions
- GitHub Issues: https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/issues

---

**最終更新**: 2026年1月24日
