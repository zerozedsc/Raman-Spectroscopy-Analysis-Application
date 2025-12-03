# Splash Screen Image Requirements

## 📏 Image Specifications

| Property | Requirement | Notes |
|----------|-------------|-------|
| **File name** | `splash.png` | Exact name required |
| **Location** | `assets/splash.png` | Relative to project root |
| **Format** | PNG | Supports transparency |
| **Dimensions** | 600 x 400 pixels | 3:2 aspect ratio |
| **File size** | < 500 KB | Will be embedded in executable |
| **Color mode** | RGB or RGBA | Alpha channel optional |

## 🎨 Design Guidelines

### Layout Zones

```
┌─────────────────────────────────────────────────────────┐
│                                                         │ 50px
│                    TOP ZONE (Logo)                      │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│                                                         │
│                  MIDDLE ZONE                            │ 250px
│                  (App Name & Tagline)                   │
│                                                         │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │ 100px
│               BOTTOM ZONE (Reserved)                    │
│            Progress bar + Loading text                  │
│           ⚠️ DO NOT PUT CONTENT HERE                     │
│                                                         │
└─────────────────────────────────────────────────────────┘
    600px wide

⚠️ Bottom 100px is reserved for:
   - Progress bar (auto-drawn by code)
   - Loading status text
   - Progress percentage
```

### Content Requirements

#### ✅ MUST Include:
- App name: "Raman Spectroscopy Analysis" (or localized)
- Clear branding/identity
- Professional appearance

#### ✅ SHOULD Include:
- App logo or icon (if available)
- Tagline: "Real-Time Spectral Analysis"
- Version number (e.g., "v1.0.0")
- Organization/lab name

#### ❌ DO NOT Include:
- Content in bottom 100px (reserved for progress bar)
- Busy/distracting backgrounds
- Small text (< 12pt equivalent)
- Low contrast text

## 🎨 Color Recommendations

### Option 1: Dark Theme (Current auto-generated)
```
Background: #2D3446 (dark blue-gray)
Text:       #FFFFFF (white)
Accent:     #4CAF50 (green) for progress bar
```

### Option 2: Light Theme
```
Background: #F5F5F5 (light gray)
Text:       #333333 (dark gray)
Accent:     #2196F3 (blue) for progress bar
```

### Option 3: Gradient
```
Top:        #1E3A5F (dark blue)
Bottom:     #2D3446 (darker blue-gray)
Text:       #FFFFFF (white)
Accent:     #4CAF50 (green)
```

## 📐 Example Layouts

### Layout 1: Centered Logo + Text

```
┌───────────────────────────────────────────┐
│                                           │
│              [LOGO IMAGE]                 │
│                                           │
│     Raman Spectroscopy Analysis           │
│     Real-Time Spectral Analysis           │
│                v1.0.0                     │
│                                           │
│                                           │
│                                           │
│     [Progress bar drawn by code]          │
│     Loading utilities... 10%              │
└───────────────────────────────────────────┘
```

### Layout 2: Side Logo + Text

```
┌───────────────────────────────────────────┐
│                                           │
│   [LOGO]  Raman Spectroscopy Analysis     │
│           Real-Time Spectral Analysis     │
│           v1.0.0                          │
│                                           │
│           Your Lab/Organization           │
│                                           │
│                                           │
│     [Progress bar drawn by code]          │
│     Loading utilities... 10%              │
└───────────────────────────────────────────┘
```

### Layout 3: Full Background

```
┌───────────────────────────────────────────┐
│    [Full background gradient/image]       │
│                                           │
│         Raman Spectroscopy                │
│         Analysis Application              │
│                                           │
│         Real-Time Spectral Analysis       │
│         Version 1.0.0                     │
│                                           │
│     [Progress bar drawn by code]          │
│     Loading utilities... 10%              │
└───────────────────────────────────────────┘
```

## 🛠️ Tools & Resources

### Quick Creation Tools

#### 1. PowerPoint (Easiest)
```
1. Create new slide (600x400 px)
2. Add app logo/text
3. Export as PNG
4. Save to assets/splash.png
```

#### 2. Canva (Free Online)
```
1. Create custom size: 600x400 px
2. Choose "Presentation" template
3. Customize with your branding
4. Download as PNG
```

#### 3. GIMP (Free Desktop)
```
1. New image: 600x400 px, RGB
2. Add layers: background, logo, text
3. Export as PNG
4. Save to assets/splash.png
```

#### 4. Photoshop (Professional)
```
1. New document: 600x400 px, 72 DPI
2. Design splash screen
3. Export as PNG-24
4. Save to assets/splash.png
```

### Free Logo Resources

- **Icons8**: https://icons8.com/icons (free with attribution)
- **Flaticon**: https://www.flaticon.com/ (free with attribution)
- **Unsplash**: https://unsplash.com/ (backgrounds)
- **Pixabay**: https://pixabay.com/ (backgrounds)

### Font Recommendations

- **Modern**: Segoe UI, Roboto, Open Sans
- **Professional**: Arial, Helvetica, Calibri
- **Scientific**: Cambria, Georgia, Times New Roman

## 📝 Text Content Suggestions

### App Name Variations

**English**:
- "Raman Spectroscopy Analysis"
- "Raman Spectral Analyzer"
- "Raman Analysis Suite"

**Japanese** (if localized):
- "ラマン分光分析"
- "ラマンスペクトル解析"

### Tagline Options

- "Real-Time Spectral Analysis"
- "Advanced Spectroscopy Tools"
- "Scientific Data Analysis"
- "Spectral Analysis Made Easy"
- "Professional Raman Analysis"

### Version Display

- "Version 1.0.0"
- "v1.0.0"
- "Build 2025.11"

## ⚠️ Important Notes

### Bottom 100px Reserved Zone

The bottom 100px of the splash screen is **automatically drawn by the code**:

```python
# This is drawn by splash_screen.py, NOT by your image
┌────────────────────────────────────────┐
│ Background bar (gray):                 │
│ ████████████████████████████████████   │
│                                        │
│ Progress fill (green):                 │
│ ████████████░░░░░░░░░░░░░░░░░░░░░░░░   │
│                                        │
│ Status text:                           │
│ "Loading utilities... 10%"             │
└────────────────────────────────────────┘
```

**DO NOT** include:
- ❌ Your own progress bar in the image
- ❌ "Loading..." text in the image
- ❌ Percentage indicators in bottom area

**The code will overlay**:
- ✓ Animated progress bar (green)
- ✓ Dynamic status messages
- ✓ Current percentage

## ✅ Current Status

**File**: `assets/splash.png` is currently a **placeholder** text file.

**Action Required**:
1. Delete the placeholder file
2. Create actual PNG image (600x400 px)
3. Save as `assets/splash.png`
4. Rebuild with `.\build_scripts\build_optimized.ps1`

**If you don't replace it**:
- App will auto-generate gradient background
- Shows "Raman Spectroscopy Analysis Application" text
- Still works, but less professional appearance

## 🧪 Testing Your Splash Screen

### Quick Test (No Build Required)

```powershell
# Test splash screen directly
cd J:\Coding\研究\raman-app
python splash_screen.py

# Should show:
#   - Your splash image (if PNG exists)
#   - Or auto-generated gradient (if PNG missing)
#   - Progress bar at bottom
```

### Full Test (After Build)

```powershell
# Build with new splash
.\build_scripts\build_optimized.ps1 -Clean

# Run executable
.\dist\raman_app\raman_app.exe

# Verify:
#   ✓ Splash appears in 1-2 seconds
#   ✓ Your custom image displays
#   ✓ Progress bar animates smoothly
#   ✓ Text is readable
#   ✓ No visual artifacts
```

## 📚 Reference Images

See example splash screens in:
- `.docs/building/examples/` (if available)
- Similar PySide6/Qt applications
- Professional scientific software

## 🎯 Quick Checklist

Before building:
- [ ] PNG image created (600x400 px)
- [ ] Saved as `assets/splash.png`
- [ ] Bottom 100px kept clear
- [ ] App name visible and readable
- [ ] Colors match app theme
- [ ] File size < 500 KB
- [ ] Text is legible
- [ ] Tested with `python splash_screen.py`

---

**Need Help?**

See comprehensive guide: `.docs/building/STARTUP_OPTIMIZATION_GUIDE.md`

**Current Status**: ⚠️ Placeholder - replace with actual image for professional appearance

**Last Updated**: November 21, 2025
