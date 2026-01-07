# Quick Reference: PCA UI Improvements (2025-01-21)

## ✅ What Changed

### 1. Visual Feedback for Checkboxes
**All checkboxes now highlight when checked:**
- Light blue background (#e3f2fd)
- Rounded corners
- Applied to: Dataset selection, Component selection (Loading & Distributions)

### 2. Select All Validation
**Select All now respects maximum limits:**
- **Loading Plot**: Max 8 components
- **Distributions**: Max 6 components
- Disabled with tooltip when limit exceeded
- Smart logic prevents over-selection

### 3. Full Localization
**All UI strings now use localization:**
- Display Options button
- Show Components buttons
- Select All checkboxes
- Mode dropdown options
- Info labels

**Languages supported**: English (EN) + Japanese (JA)

### 4. X-Axis Labels
**Only bottom row shows x-axis labels:**
- Clearer subplot layouts
- Dynamic title sizing (smaller when >4 plots)

### 5. Clean Console
**No more tight_layout warnings:**
- Improved error handling in matplotlib_widget.py
- Silent fallbacks for incompatible layouts

---

## 📖 User Guide

### How to Use Display Options (Spectrum Preview)

1. Click **"⚙️ Display Options"** button to open sidebar
2. Choose **Display Mode**:
   - "Mean Spectra" - shows average of each dataset
   - "All Spectra" - shows all individual spectra
3. **Select Datasets**: Check which datasets to show
4. Use **"Select All"** to quickly select/deselect all

**Visual Feedback**: Checked items have blue background

---

### How to Use Component Selectors (Loading Plot & Distributions)

#### Loading Plot (Max 8)
1. Click **"🔧 Show Components"** to open sidebar
2. Check which PC components to display (up to 8)
3. Use **"Select All"** if ≤8 components available
4. Plot updates automatically in 2-column grid

**Note**: If >8 components exist, Select All is disabled (shows tooltip)

#### Distributions (Max 6)
1. Click **"📊 Show Components"** to open sidebar
2. Check which PC components to show distributions (up to 6)
3. Use **"Select All"** if ≤6 components available
4. Plot updates automatically

**Note**: If >6 components exist, Select All is disabled (shows tooltip)

---

## 🎨 Visual Changes

### Before
```
☐ PC1 (45.2%)     <- No visual feedback
☐ PC2 (23.1%)
☑ PC3 (12.5%)
```

### After
```
☐ PC1 (45.2%)
☐ PC2 (23.1%)
☑ PC3 (12.5%)  <- Blue background when checked
```

---

## 🌐 Localization

### English
- Display Options
- Show Components
- Select All
- Display Mode: Mean Spectra / All Spectra
- Select Datasets to Show:

### Japanese
- 表示オプション
- 成分を表示
- すべて選択
- 表示モード: 平均スペクトル / 全スペクトル
- 表示するデータセットを選択:

**How to Switch Language**: Configure in app settings (existing localization system)

---

## 🛠️ Developer Notes

### New CSS Pattern
```python
cb.setStyleSheet("""
    QCheckBox { padding: 4px; }
    QCheckBox:checked {
        background-color: #e3f2fd;
        border-radius: 3px;
    }
""")
```

### Localization Pattern
```python
button = QPushButton(localize_func("ANALYSIS_PAGE.display_options_button"))
```

### Select All Pattern
```python
select_all_cb.setEnabled(len(items) <= max_limit)
if len(items) > max_limit:
    select_all_cb.setToolTip("Explanation...")
```

---

## 📊 Technical Specs

### Max Limits
| Component        | Max Items | Reason          |
| ---------------- | --------- | --------------- |
| Loading Plot     | 8         | 4×2 grid layout |
| Distributions    | 6         | 3×2 grid layout |
| Spectrum Preview | Unlimited | Dynamic layout  |

### Grid Layouts
| Items | Grid | Bottom Row Indices |
| ----- | ---- | ------------------ |
| 2     | 1×2  | 0, 1               |
| 4     | 2×2  | 2, 3               |
| 6     | 3×2  | 4, 5               |
| 8     | 4×2  | 6, 7               |

### Title Sizing
- **≤4 plots**: fontsize=11
- **>4 plots**: fontsize=9

---

## ❓ FAQ

**Q: Why is Select All disabled?**  
A: Total items exceeds maximum limit. Hover for tooltip with details.

**Q: Why don't all plots show x-axis labels?**  
A: Only bottom row shows labels to reduce clutter. This is intentional.

**Q: Can I create multiple groups at once?**  
A: Not yet - this is Task 7 (deferred). Use "Add Group" for now.

**Q: Are there performance issues with many components?**  
A: Max limits (8 for Loading, 6 for Distributions) prevent performance issues.

**Q: Can I switch languages?**  
A: Yes, use app's existing localization settings (EN/JA supported).

---

## 🐛 Known Issues

**None** - All improvements tested and working as expected.

---

## 📞 Support

For issues or questions:
1. Check this guide
2. Review full documentation: `.docs/summary/2025-01-21_pca-ui-improvements-localization.md`
3. Check TODO for future enhancements: `.docs/todos/2025-01-21_task7-multi-group-creation.md`

---

*Last Updated: 2025-01-21*  
*Version: 1.0*
