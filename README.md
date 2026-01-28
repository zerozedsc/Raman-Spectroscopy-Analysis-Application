# Raman Spectroscopy Analysis Application

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PySide6](https://img.shields.io/badge/GUI-PySide6-green.svg)](https://www.qt.io/qt-for-python)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application)
[![Documentation](https://readthedocs.org/projects/raman-spectroscopy-analysis-application/badge/?version=latest)](https://raman-spectroscopy-analysis-application.readthedocs.io/en/latest/)

<div align="center">
  <img src="readme/images/app-main-interface.png" alt="Main application interface" width="800"/>
  
  *A comprehensive desktop application for real-time Raman spectroscopy classification and disease detection*
</div>

---

## 🌐 Language / 言語

📖 **[Read in English](readme/README_EN.md)** | **[日本語で読む](readme/README_JA.md)**

📚 **[Full Documentation](https://raman-spectroscopy-analysis-application.readthedocs.io/en/latest/)** | **[完全なドキュメント](https://raman-spectroscopy-analysis-application.readthedocs.io/ja/latest/)**

---

## 🎯 Overview

An **open-source**, **cross-platform** desktop application designed for **real-time Raman spectroscopy classification** with focus on **disease detection** in clinical and research settings. Developed at the **University of Toyama**, under the **Laboratory for Clinical Photonics and Information Engineering** (臨床光情報工学研究室).

### ✨ Key Features

| Feature                         | Description                                                                                                    |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| 🔬 **40+ Preprocessing Methods** | Research-validated algorithms including baseline correction, smoothing, normalization, and feature engineering |
| 📊 **Real-Time Analysis**        | PCA, UMAP, t-SNE, hierarchical clustering, K-means with interactive visualization                              |
| 🤖 **Machine Learning**          | Complete ML pipeline with SVM, Random Forest, XGBoost, Logistic Regression, and SHAP interpretability          |
| 🎨 **Modern GUI**                | Intuitive PySide6/Qt6 interface with multi-language support (English/Japanese)                                 |
| 🧪 **Research-Grade**            | Validated parameter constraints from peer-reviewed literature                                                  |
| 📦 **Distribution (Windows)**    | Portable executable available via Releases; installer workflow is still being finalized                        |
| 🌍 **Open Source**               | MIT License, contributions welcome                                                                             |

### 🎓 Academic Context

This software was developed as a **final year project** for the Bachelor of Science degree at the **University of Toyama**, focusing on applying Raman spectroscopy for early disease detection and classification.

---

## 🚀 Quick Start

### Option 1: From Source (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application.git
cd Raman-Spectroscopy-Analysis-Application

# Install using UV (recommended)
pip install uv
uv venv
uv pip install -e .
uv run python main.py

# Or use a traditional venv + pip
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -e .
python main.py
```

### Option 2: Portable Executable (Windows Only)

Download pre-built executable from [Releases](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/releases) — no installation required.

### Option 3: Installer (Windows Only)

The Windows installer workflow is still being finalized. For now, please use the **Portable Executable** option or run from source.

---

## 📚 Documentation

### 📖 User Documentation

| Document                                                                                          | Description                             |
| ------------------------------------------------------------------------------------------------- | --------------------------------------- |
| [📘 Full Documentation](https://raman-spectroscopy-analysis-application.readthedocs.io/en/latest/) | Complete online documentation (English) |
| [📗 Complete English Guide](readme/README_EN.md)                                                   | Comprehensive user guide with tutorials |
| [📕 完全な日本語ガイド](readme/README_JA.md)                                                       | 日本語の完全なユーザーガイド            |
| [📄 Development Guidelines](readme/DEVELOPMENT_GUIDELINES.md)                                      | For contributors and developers         |

### 🔗 Quick Links

- [Installation Guide](https://raman-spectroscopy-analysis-application.readthedocs.io/en/latest/installation.html)
- [User Guide](https://raman-spectroscopy-analysis-application.readthedocs.io/en/latest/user-guide/index.html)
- [Analysis Methods Reference](https://raman-spectroscopy-analysis-application.readthedocs.io/en/latest/analysis-methods/index.html)
- [API Documentation](https://raman-spectroscopy-analysis-application.readthedocs.io/en/latest/api/index.html)
- [Changelog](CHANGELOG.md)
- [License](LICENSE)

---

## 🔬 Research Context

**Project Title:** Real-Time Raman Spectroscopy Classification Software for Disease Detection  
**Institution:** University of Toyama (富山大学)  
**Laboratory:** [Clinical Photonics and Information Engineering](http://www3.u-toyama.ac.jp/medphoto/)  
**Research Focus:** Pre-disease detection (未病), multiple myeloma (MM), MGUS classification

**Student:** Muhammad Helmi bin Rozain (12270294)  
**Supervisors:** 大嶋 佑介 (Oshima Yusuke), 竹谷 皓規 (Taketani Akinori)

### Research Applications

This software has been designed for:
- Early cancer detection via Raman spectroscopy
- Multiple myeloma (MM) and MGUS differentiation
- Cell, blood, and tissue sample analysis
- Exploratory biomarker discovery
- Quality control and method validation

---

## 🤝 Contributing

We welcome contributions from the research community! Please see our [Contributing Guidelines](readme/DEVELOPMENT_GUIDELINES.md) for:

- Reporting bugs and requesting features
- Submitting pull requests
- Code style and documentation standards
- Testing requirements

```bash
# Contribution workflow
git checkout -b feature/your-feature
git commit -m "feat: add new preprocessing method"
git push origin feature/your-feature
# Then open a Pull Request on GitHub
```

---

## 📞 Support & Contact

- 🐛 **Bug Reports:** [GitHub Issues](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/issues)
- 💬 **Questions & Discussions:** [GitHub Discussions](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/discussions)
- 📧 **Direct Contact:** [@zerozedsc](https://github.com/zerozedsc)
- 📖 **Documentation:** [ReadTheDocs](https://raman-spectroscopy-analysis-application.readthedocs.io/en/latest/)

---

## 🌟 Citation

If you use this software in your research, please cite:

```bibtex
@software{muhammadhelmirozain2025raman,
  author = {Rozain, Muhammad Helmi bin},
  title = {Raman Spectroscopy Analysis Application: A Comprehensive Platform for Real-Time Spectral Classification},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application},
  institution = {University of Toyama, Laboratory for Clinical Photonics and Information Engineering}
}
```

---

## ⚖️ License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

> Note: This software is intended for **research use only** and is **not approved** for clinical diagnostic purposes.

---

<div align="center">
  <p><strong>Developed for the Advancement of Raman Spectroscopy Research</strong></p>
  <p>
    <a href="http://www3.u-toyama.ac.jp/medphoto/">Laboratory for Clinical Photonics and Information Engineering</a> • 
    <a href="https://www.u-toyama.ac.jp/">University of Toyama</a>
  </p>
  <p><strong>富山大学 臨床光情報工学研究室</strong></p>
  <p><em>Empowering biomedical research through open-source software</em></p>
</div>
