# References

Comprehensive bibliography of scientific literature, algorithms, and resources used in the development of this application.

## Table of Contents
- [Raman Spectroscopy](#raman-spectroscopy)
- [Preprocessing Methods](#preprocessing-methods)
- [Machine Learning](#machine-learning)
- [Software Libraries](#software-libraries)
- [Medical Applications](#medical-applications)

---

## Raman Spectroscopy

### Fundamentals

1. **Raman, C. V., & Krishnan, K. S. (1928)**  
   *A New Type of Secondary Radiation*  
   Nature, 121(3048), 501-502.  
   DOI: [10.1038/121501c0](https://doi.org/10.1038/121501c0)  
   - Original discovery of the Raman scattering effect

2. **Smith, E., & Dent, G. (2019)**  
   *Modern Raman Spectroscopy: A Practical Approach* (2nd ed.)  
   Wiley.  
   ISBN: 978-0470011836  
   - Comprehensive textbook on Raman spectroscopy theory and practice

3. **Puppels, G. J., et al. (1990)**  
   *Studying single living cells and chromosomes by confocal Raman microspectroscopy*  
   Nature, 347(6290), 301-303.  
   DOI: [10.1038/347301a0](https://doi.org/10.1038/347301a0)  
   - Pioneering work on biological Raman spectroscopy

### Medical Applications

4. **Kendall, C., et al. (2009)**  
   *Raman spectroscopy for medical diagnostics—From in-vitro biofluid assays to in-vivo cancer detection*  
   Analytical and Bioanalytical Chemistry, 396(1), 73-77.  
   DOI: [10.1007/s00216-009-3062-6](https://doi.org/10.1007/s00216-009-3062-6)

5. **Kong, K., et al. (2015)**  
   *Raman spectroscopy for medical diagnostics: From in-vitro to in-vivo applications*  
   Advances in Drug Delivery Reviews, 89, 121-134.  
   DOI: [10.1016/j.addr.2015.03.009](https://doi.org/10.1016/j.addr.2015.03.009)

6. **Movasaghi, Z., et al. (2007)**  
   *Raman Spectroscopy of Biological Tissues*  
   Applied Spectroscopy Reviews, 42(5), 493-541.  
   DOI: [10.1080/05704920701551530](https://doi.org/10.1080/05704920701551530)  
   - Comprehensive reference for Raman peak assignments in biological materials

---

## Preprocessing Methods

### Baseline Correction

7. **Eilers, P. H. C. (2003)**  
   *A Perfect Smoother*  
   Analytical Chemistry, 75(14), 3631-3636.  
   DOI: [10.1021/ac034173t](https://doi.org/10.1021/ac034173t)  
   - Whittaker smoother and baseline estimation

8. **Eilers, P. H. C., & Boelens, H. F. M. (2005)**  
   *Baseline Correction with Asymmetric Least Squares Smoothing*  
   Leiden University Medical Centre Report, 1(1), 5.  
   - AsLS baseline correction algorithm

9. **Zhang, Z.-M., et al. (2010)**  
   *Baseline Correction Using Adaptive Iteratively Reweighted Penalized Least Squares*  
   Analyst, 135(5), 1138-1146.  
   DOI: [10.1039/B922045C](https://doi.org/10.1039/B922045C)  
   - airPLS algorithm

10. **Xu, H., et al. (2011)**  
    *Baseline correction method based on doubly reweighted penalized least squares*  
    Applied Optics, 58(14), 3913-3920.  
    DOI: [10.1364/AO.58.003913](https://doi.org/10.1364/AO.58.003913)  
    - drPLS and arpls algorithms

11. **Komsta, Ł., & Vander Heyden, Y. (2017)**  
    *Improved baseline recognition and modeling of FT-IR spectra using wavelets*  
    Chemometrics and Intelligent Laboratory Systems, 60(1-2), 49-65.  
    - Wavelet-based baseline correction

12. **Automated Weighted Method (AWM)**  
    Konevskikh, T., et al. (2016)  
    *Automated baseline correction for infrared spectra*  
    Analyst, 141(13), 3954-3962.  
    DOI: [10.1039/c6an00355a](https://doi.org/10.1039/c6an00355a)

### Smoothing and Denoising

13. **Savitzky, A., & Golay, M. J. E. (1964)**  
    *Smoothing and Differentiation of Data by Simplified Least Squares Procedures*  
    Analytical Chemistry, 36(8), 1627-1639.  
    DOI: [10.1021/ac60214a047](https://doi.org/10.1021/ac60214a047)  
    - Savitzky-Golay filter

14. **Kou, F., et al. (2013)**  
    *A preprocessing method for attenuating background drift in surface-enhanced Raman scattering spectra*  
    Optics Communications, 305, 9-13.  
    DOI: [10.1016/j.optcom.2013.04.045](https://doi.org/10.1016/j.optcom.2013.04.045)

### Normalization

15. **Barnes, R. J., et al. (1989)**  
    *Standard Normal Variate Transformation and De-trending of Near-Infrared Diffuse Reflectance Spectra*  
    Applied Spectroscopy, 43(5), 772-777.  
    DOI: [10.1366/0003702894202201](https://doi.org/10.1366/0003702894202201)  
    - SNV normalization

16. **Geladi, P., et al. (1985)**  
    *Linearization and Scatter-Correction for Near-Infrared Reflectance Spectra of Meat*  
    Applied Spectroscopy, 39(3), 491-500.  
    DOI: [10.1366/0003702854248656](https://doi.org/10.1366/0003702854248656)  
    - MSC (Multiplicative Scatter Correction)

17. **Dieterle, F., et al. (2006)**  
    *Probabilistic Quotient Normalization as Robust Method to Account for Dilution of Complex Biological Mixtures*  
    Analytical Chemistry, 78(13), 4281-4290.  
    DOI: [10.1021/ac051632c](https://doi.org/10.1021/ac051632c)  
    - PQN normalization

### Feature Engineering

18. **Geurts, P., Ernst, D., & Wehenkel, L. (2006)**  
    *Extremely randomized trees*  
    Machine Learning, 63(1), 3-42.  
    DOI: [10.1007/s10994-006-6226-1](https://doi.org/10.1007/s10994-006-6226-1)  
    - Basis for feature importance methods

19. **Mallat, S. G. (1989)**  
    *A theory for multiresolution signal decomposition: the wavelet representation*  
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 11(7), 674-693.  
    DOI: [10.1109/34.192463](https://doi.org/10.1109/34.192463)  
    - Wavelet transform theory

---

## Machine Learning

### Dimensionality Reduction

20. **Pearson, K. (1901)**  
    *On Lines and Planes of Closest Fit to Systems of Points in Space*  
    Philosophical Magazine, 2(11), 559-572.  
    DOI: [10.1080/14786440109462720](https://doi.org/10.1080/14786440109462720)  
    - Original PCA paper

21. **McInnes, L., Healy, J., & Melville, J. (2018)**  
    *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction*  
    arXiv:1802.03426  
    DOI: [10.48550/arXiv.1802.03426](https://doi.org/10.48550/arXiv.1802.03426)  
    - UMAP algorithm

22. **van der Maaten, L., & Hinton, G. (2008)**  
    *Visualizing Data using t-SNE*  
    Journal of Machine Learning Research, 9, 2579-2605.  
    - t-SNE algorithm

### Classification

23. **Cortes, C., & Vapnik, V. (1995)**  
    *Support-Vector Networks*  
    Machine Learning, 20(3), 273-297.  
    DOI: [10.1007/BF00994018](https://doi.org/10.1007/BF00994018)  
    - SVM algorithm

24. **Breiman, L. (2001)**  
    *Random Forests*  
    Machine Learning, 45(1), 5-32.  
    DOI: [10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)  
    - Random Forest algorithm

25. **Chen, T., & Guestrin, C. (2016)**  
    *XGBoost: A Scalable Tree Boosting System*  
    Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785-794.  
    DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)  
    - XGBoost algorithm

26. **Barker, M., & Rayens, W. (2003)**  
    *Partial least squares for discrimination*  
    Journal of Chemometrics, 17(3), 166-173.  
    DOI: [10.1002/cem.785](https://doi.org/10.1002/cem.785)  
    - PLS-DA algorithm

### Interpretability

27. **Lundberg, S. M., & Lee, S.-I. (2017)**  
    *A Unified Approach to Interpreting Model Predictions*  
    Advances in Neural Information Processing Systems 30 (NIPS 2017).  
    - SHAP (SHapley Additive exPlanations)

28. **Ribeiro, M. T., Singh, S., & Guestrin, C. (2016)**  
    *"Why Should I Trust You?": Explaining the Predictions of Any Classifier*  
    Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 1135-1144.  
    DOI: [10.1145/2939672.2939778](https://doi.org/10.1145/2939672.2939778)  
    - LIME algorithm

### Validation

29. **Stone, M. (1974)**  
    *Cross-Validatory Choice and Assessment of Statistical Predictions*  
    Journal of the Royal Statistical Society: Series B (Methodological), 36(2), 111-133.  
    DOI: [10.1111/j.2517-6161.1974.tb00994.x](https://doi.org/10.1111/j.2517-6161.1974.tb00994.x)  
    - Cross-validation theory

30. **Varma, S., & Simon, R. (2006)**  
    *Bias in error estimation when using cross-validation for model selection*  
    BMC Bioinformatics, 7(1), 91.  
    DOI: [10.1186/1471-2105-7-91](https://doi.org/10.1186/1471-2105-7-91)  
    - Nested cross-validation

---

## Software Libraries

### Core Dependencies

31. **The Qt Company (2023)**  
    *Qt for Python (PySide6)*  
    [https://www.qt.io/qt-for-python](https://www.qt.io/qt-for-python)  
    - GUI framework

32. **Harris, C. R., et al. (2020)**  
    *Array programming with NumPy*  
    Nature, 585(7825), 357-362.  
    DOI: [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2)  
    - NumPy library

33. **Virtanen, P., et al. (2020)**  
    *SciPy 1.0: Fundamental algorithms for scientific computing in Python*  
    Nature Methods, 17(3), 261-272.  
    DOI: [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2)  
    - SciPy library

34. **McKinney, W. (2010)**  
    *Data Structures for Statistical Computing in Python*  
    Proceedings of the 9th Python in Science Conference, 56-61.  
    DOI: [10.25080/Majora-92bf1922-00a](https://doi.org/10.25080/Majora-92bf1922-00a)  
    - pandas library

35. **Pedregosa, F., et al. (2011)**  
    *Scikit-learn: Machine Learning in Python*  
    Journal of Machine Learning Research, 12, 2825-2830.  
    - scikit-learn library

36. **Hunter, J. D. (2007)**  
    *Matplotlib: A 2D graphics environment*  
    Computing in Science & Engineering, 9(3), 90-95.  
    DOI: [10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55)  
    - matplotlib library

### Specialized Libraries

37. **Stevens, O., et al. (2023)**  
    *RamanSPy: An Open-Source Python Package for Raman Spectroscopy*  
    Analytical Chemistry, 95(2), 1163-1172.  
    DOI: [10.1021/acs.analchem.2c04364](https://doi.org/10.1021/acs.analchem.2c04364)  
    - RamanSPy library (preprocessing and analysis tools)

38. **Lafarge, D. (2023)**  
    *pybaselines: A Python library of algorithms for the baseline correction of experimental data*  
    Journal of Open Source Software, 8(82), 5181.  
    DOI: [10.21105/joss.05181](https://doi.org/10.21105/joss.05181)  
    - pybaselines library (baseline correction methods)

39. **Paszke, A., et al. (2019)**  
    *PyTorch: An Imperative Style, High-Performance Deep Learning Library*  
    Advances in Neural Information Processing Systems 32 (NeurIPS 2019).  
    - PyTorch library (deep learning models)

---

## Medical Applications

### Blood Plasma Analysis

40. **Kakita, K., et al. (2021)**  
    *Blood plasma analysis by Raman spectroscopy for early diagnosis*  
    [Laboratory for Clinical Photonics and Information Engineering, University of Toyama]  
    - Related research from supervising laboratory

41. **Sheng, D., et al. (2022)**  
    *Advancing Clinical Translation of Raman Spectroscopy*  
    Translational Biophotonics, 4(3), e202200003.  
    DOI: [10.1002/tbio.202200003](https://doi.org/10.1002/tbio.202200003)

42. **Cui, S., et al. (2018)**  
    *Raman spectroscopy and machine learning for the classification of esophageal squamous cell carcinoma*  
    Spectrochimica Acta Part A: Molecular and Biomolecular Spectroscopy, 193, 415-422.  
    DOI: [10.1016/j.saa.2017.12.050](https://doi.org/10.1016/j.saa.2017.12.050)

### Pre-disease (未病) Detection

43. **Qiu, J., et al. (2021)**  
    *Traditional Chinese medicine on treating primary dysmenorrhea*  
    Evidence-Based Complementary and Alternative Medicine, 2021, 6645246.  
    DOI: [10.1155/2021/6645246](https://doi.org/10.1155/2021/6645246)  
    - Related to 未病 (mibyō) concept in preventive medicine

44. **Ozaki, Y. (2023)**  
    *Application of Raman spectroscopy to pre-disease diagnosis*  
    [University of Toyama Research]  
    - Concept of using spectroscopy for early health monitoring

---

## Standards and Guidelines

### Best Practices

45. **Benevides, J. M., Overman, S. A., & Thomas Jr, G. J. (2005)**  
    *Raman spectroscopy of proteins*  
    Current Protocols in Protein Science, Chapter 17, Unit 17.8.  
    DOI: [10.1002/0471140864.ps1708s42](https://doi.org/10.1002/0471140864.ps1708s42)

46. **Butler, H. J., et al. (2016)**  
    *Using Raman spectroscopy to characterize biological materials*  
    Nature Protocols, 11(4), 664-687.  
    DOI: [10.1038/nprot.2016.036](https://doi.org/10.1038/nprot.2016.036)  
    - Comprehensive protocol for Raman analysis

### Quality Control

47. **ASTM International (2020)**  
    *ASTM E1840-96(2020) Standard Guide for Raman Shift Standards for Spectrometer Calibration*  
    DOI: [10.1520/E1840-96R20](https://doi.org/10.1520/E1840-96R20)

48. **ISO 18115-1:2023**  
    *Surface chemical analysis — Vocabulary — Part 1: General terms and terms used in spectroscopy*  
    - International standards for spectroscopic analysis

---

## Related Resources

### Online Resources

- [IRUG (Infrared and Raman Users Group) Spectral Database](http://www.irug.org/)  
  Comprehensive database of reference Raman spectra

- [RRUFF Project](https://rruff.info/)  
  Raman spectra database for minerals

- [Bio-Rad KnowItAll Spectroscopy](https://www.knowitall.com/)  
  Commercial spectral database

- [RamanDB](http://ramandb.com/)  
  Free Raman spectral database

### Educational Materials

- [MIT OpenCourseWare: Modern Analytical Techniques](https://ocw.mit.edu/)  
  Free course materials on analytical spectroscopy

- [nptel: Molecular Spectroscopy](https://nptel.ac.in/)  
  Video lectures on spectroscopy (including Raman)

---

## Software Citation

If you use this software in your research, please cite:

```bibtex
@software{rozain2025raman,
  author = {Rozain, Muhamad Helmi bin},
  title = {Raman Spectroscopy Analysis Application: A Comprehensive Platform for Real-Time Spectral Classification},
  year = {2025},
  version = {1.0.0-alpha},
  publisher = {GitHub},
  url = {https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application},
  institution = {University of Toyama, Laboratory for Clinical Photonics and Information Engineering}
}
```

---

## Contributing

We welcome contributions to this reference list! If you know of relevant papers or resources that should be included:

1. Open an issue on [GitHub](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/issues)
2. Submit a pull request with proper citation formatting
3. Contact the maintainer via email

**Citation Format**:
- Author(s), Year
- Title (italicized)
- Journal/Conference, Volume(Issue), Pages
- DOI link (if available)
- Brief description (1-2 sentences)

---

**Last Updated**: 2026-01-24  
**Maintained by**: Muhamad Helmi bin Rozain ([@zerozedsc](https://github.com/zerozedsc))
