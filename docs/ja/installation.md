# インストールガイド

本アプリは現在、**インストーラー配布を準備中**です（Coming soon）。
現時点では以下のいずれかの方法をご利用ください。

---

## 1) Windows（ポータブル版）

配布物として `RamanApp.exe`（または同等の実行ファイル）が提供されている場合：

1. ZIPを解凍
2. `RamanApp.exe` を実行

※ インストーラー版（`.exe` セットアップ形式）は準備中です。

---

## 2) ソースから実行（開発者向け）

### 前提

- Python 3.12（3.12.x）

### 手順（例）

```bash
# 1) このリポジトリを取得して作業ディレクトリへ移動

# 2) 仮想環境
python -m venv .venv

# 3) 有効化
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 4) 依存関係をインストール（pyproject.toml を利用）
pip install -e .

# 5) 起動
python main.py
```

---

## うまくいかない場合

- [FAQ](faq.md)
- [トラブルシューティング](troubleshooting.md)

---

**最終更新**: 2026年1月24日
