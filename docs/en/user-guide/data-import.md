# Data Import Guide

Complete guide to importing, organizing, and managing spectral data in the application.

## Table of Contents
- {ref}`Supported File Formats <supported-file-formats>`
- {ref}`Import Workflow <import-workflow>`
- {ref}`Data Organization <data-organization>`
- {ref}`Group Management <group-management>`
- {ref}`Data Validation <data-validation>`
- {ref}`Advanced Features <advanced-features>`

---

(supported-file-formats)=
## Supported File Formats

### Primary Formats

#### CSV Files (Recommended)
**Format**: Comma-Separated Values

**Structure**:
```text
Wavenumber,Sample1,Sample2,Sample3
400.0,100.5,98.3,102.1
401.0,101.2,99.1,103.5
402.0,102.8,100.4,104.2
...
```

**Requirements**:
- First column: Wavenumbers (numeric, ascending)
- Subsequent columns: Intensity values for each spectrum
- Header row: Sample identifiers (optional but recommended)
- Decimal separator: Period (`.`)
- No missing values (use `0` or interpolate)

**Example Import**:
```python
# File: blood_plasma_data.csv
# Columns: wavenumber, patient_001, patient_002, patient_003
# Rows: 1000+ wavenumber points
```

#### TXT Files (Text Format)
**Format**: Tab or space-delimited

**Structure**:
```
400.0    100.5    98.3     102.1
401.0    101.2    99.1     103.5
402.0    102.8    100.4    104.2
...
```

**Requirements**:
- Similar to CSV but using tabs or spaces
- Optional header row
- Consistent delimiter throughout file

#### ASC/ASCII Files

**Format**: Text format containing two columns: wavenumber and intensity

**Supported extensions**: `.asc`, `.ascii`

#### PKL Files

**Format**: Pickled pandas DataFrame

**Supported extension**: `.pkl`

### Future Import Support (Planned)

- **SPC**: Galactic SPC binary format
- **WDF**: Renishaw WiRE format

---

(import-workflow)=
## Import Workflow

### Step 1: Navigate to Data Package Page

1. Open the application
2. Click **📦 Data Package** tab
3. Ensure you're in an active project (create new if needed)

### Step 2: Select Files for Import

**Method A: File Browser**

1. Click **[Import Data]** button
2. File dialog opens
3. Navigate to your data directory
4. Select one or multiple files (CSV/TXT/ASC/PKL)
5. Click **[Open]**

**Method B: Drag and Drop**

1. Open file explorer (Windows Explorer, Finder)
2. Navigate to your data files
3. Drag files directly into the import area
4. Release to drop

**Method C: Paste File Paths**

1. Copy file path(s) from explorer
2. Click **[Import from Path]**
3. Paste paths (one per line for multiple files)
4. Click **[Import]**

### Step 3: Data Validation

Application automatically checks:

During import, you will see a validation status panel/toast listing items like:

- File format (CSV/TXT/ASC/PKL)
- Wavenumber column detected
- Number of spectra (samples)
- Wavenumber range (e.g., 400–1800 cm⁻¹)
- Data integrity checks
- Missing values handling (if enabled)

> **Visual reference**: See the Data Package Page screenshot in `interface-overview.md`.

**Validation Checks**:
- File format compatibility
- Wavenumber column detection
- Consistent wavenumber spacing
- No duplicate wavenumbers
- Numeric data types
- Missing value handling
- Outlier detection (optional)

### Step 4: Preview and Confirm

**Preview window**:

The preview dialog shows:

- Selected file name
- Sample count
- Wavenumber range
- A preview plot (typically the first few spectra)
- Import options (auto-detect wavenumber column, interpolate missing values, etc.)
- **Cancel** / **Import** actions

**Options**:
- **Auto-detect wavenumber column**: Automatically identify x-axis
- **Interpolate missing values**: Fill gaps with linear interpolation
- **Apply baseline correction**: Pre-process during import (optional)

### Step 5: Confirmation

After import completes, a success notification confirms the number of spectra imported and the source file.

---

(data-organization)=
## Data Organization

### Project Structure

Data is organized hierarchically:

```
Project: blood_plasma_study/
├── Data Packages/
│   ├── batch1_healthy/
│   │   ├── healthy_001.csv
│   │   ├── healthy_002.csv
│   │   └── metadata.json
│   ├── batch2_disease/
│   │   ├── disease_001.csv
│   │   ├── disease_002.csv
│   │   └── metadata.json
│   └── batch3_validation/
│       └── validation_set.csv
├── Preprocessing Pipelines/
│   └── standard_pipeline.json
└── Results/
    ├── analysis/
    └── ml_models/
```

### Creating Data Packages

**Data Package** = Collection of related spectra

**Create New Package**:
1. Click **[+ New Package]** in Data Package page
2. Enter package name: `batch1_healthy`
3. Add description (optional): "Healthy controls, batch 1"
4. Import files into this package

**Benefits**:
- Organize by experimental batch
- Group by sample type
- Separate training/validation/test sets
- Apply batch-specific processing

### Metadata Management

Each data package can have metadata:

```json
{
  "package_name": "batch1_healthy",
  "description": "Healthy control samples from first batch",
  "acquisition_date": "2025-12-15",
  "laser_power": 50,
  "integration_time": 10,
  "spectrometer": "RamanSpecPro 5000",
  "notes": "Room temperature, 785nm laser"
}
```

**Edit Metadata**:
1. Right-click on data package
2. Select **Edit Metadata**
3. Fill in fields
4. Click **Save**

---

(group-management)=
## Group Management

### Creating Sample Groups

Groups are used for:
- Classification labels
- Statistical comparisons
- Visualization colors
- Cross-validation splits

**Create Group**:
1. Click **[Manage Groups]** button
2. Click **[+ New Group]**
3. Enter group details:
   - **Name**: `Healthy Control`
   - **Label**: `0` (numeric for ML)
   - **Color**: 🟢 Green
   - **Description**: "Healthy patients without disease"
4. Click **Create**

**Common Group Naming**:
```
For Classification:
- Healthy Control (label: 0)
- Disease Group A (label: 1)
- Disease Group B (label: 2)

For Regression:
- Low Concentration (value: 0-5)
- Medium Concentration (value: 5-10)
- High Concentration (value: 10-20)
```

### Assigning Samples to Groups

**Method A: Manual Selection**

1. Select samples in data list (Ctrl+Click for multiple)
2. Right-click → **Assign to Group**
3. Select group from dropdown
4. Click **Assign**

**Method B: Bulk Assignment**

1. Click **[Bulk Assign]** button
2. Use pattern matching:
   - **Pattern**: `healthy_*` → Group: Healthy Control
   - **Pattern**: `disease_*` → Group: Disease
3. Preview assignments
4. Click **Apply**

**Method C: CSV Mapping**

Create a CSV file with sample-to-group mapping:

```text
sample_name,group_label
healthy_001,Healthy Control
healthy_002,Healthy Control
disease_001,Disease
disease_002,Disease
```

**Import**:
1. Click **[Import Group Mapping]**
2. Select CSV file
3. Verify mappings
4. Click **Apply**

### Multi-Group Assignment

Some samples may belong to multiple groups:

**Example**: Clinical study with multiple factors
- Group 1: Disease Status (Healthy, Disease A, Disease B)
- Group 2: Gender (Male, Female)
- Group 3: Age Range (<30, 30-50, >50)

**Enable**:
1. `Settings → Data Management → Allow Multiple Groups`
2. Assign samples to multiple group hierarchies
3. Select active grouping for analysis

---

(data-validation)=
## Data Validation

### Automatic Checks

Application performs validation on import:

#### 1. Wavenumber Consistency

**Check**: All spectra must have identical wavenumber axis

```
✓ All spectra: 400-1800 cm⁻¹, 1000 points
✗ Mismatch detected:
  - File 1: 400-1800 cm⁻¹
  - File 2: 500-1700 cm⁻¹ (different range)
```

**Solution**: 
- Interpolate to common grid
- Crop to common range
- Use "Align Wavenumbers" tool

#### 2. Missing Values

**Check**: No NaN or infinite values

```
⚠ Missing values detected:
  - Spectrum 15: 3 NaN values at 1200-1202 cm⁻¹
  - Spectrum 47: 1 NaN value at 850 cm⁻¹
```

**Solutions**:
- **Linear interpolation** (default)
- **Polynomial interpolation**
- **Remove affected spectra**
- **Manual correction**

#### 3. Outlier Detection

**Check**: Identify spectra with unusual intensity values

```
⚠ Potential outliers:
  - Spectrum 32: Intensity >10σ from mean
  - Spectrum 88: Negative intensity values
```

**Solutions**:
- **Flag for review** (don't remove yet)
- **Visual inspection** (plot spectrum)
- **Remove if confirmed** (after manual check)
- **Note in metadata** (keep but annotate)

#### 4. Duplicate Spectra

**Check**: Detect identical or near-identical spectra

```
⚠ Duplicates detected:
  - Spectra 15 and 47: 99.8% correlation
  - Spectra 22 and 23: Identical (100%)
```

**Solutions**:
- **Remove exact duplicates** (keep one copy)
- **Flag near-duplicates** (may be technical replicates)
- **Keep all** (if intentional replicates)

### Manual Validation Tools

#### Spectrum Viewer

**Inspect individual spectra**:

1. Click on spectrum in list
2. Viewer shows:
   - Full spectrum plot
   - Statistics (mean, std, min, max)
   - Peak detection
   - Quality metrics

**Actions**:
- **Accept**: Mark as validated
- **Reject**: Remove from dataset
- **Edit**: Manually correct issues
- **Notes**: Add comments

#### Batch Validation

**Review multiple spectra**:

1. Click **[Batch Validation]**
2. Spectra displayed in grid (e.g., 3x3)
3. Navigate: Next/Previous pages
4. Actions: Accept, Reject, Flag

Use the on-screen controls for review actions.

---

(advanced-features)=
## Advanced Features

### Wavenumber Calibration

**Purpose**: Correct systematic shifts in wavenumber axis

**Calibration Methods**:

1. **Reference Peak Calibration**
   - Select known peak (e.g., 1001 cm⁻¹ for benzene)
   - Specify expected position
   - Apply linear shift correction

2. **Multi-Peak Calibration**
   - Use multiple reference peaks
   - Fit polynomial correction curve
   - Apply non-linear calibration

**Workflow**:
```text
# Example: Calibrate using 1001 cm⁻¹ benzene peak
1. Click [Calibration] in Data Package page
2. Select calibration standard spectrum
3. Mark expected peak position: 1001 cm⁻¹
4. Detected peak: 1003.5 cm⁻¹
5. Shift: -2.5 cm⁻¹
6. Apply to all spectra in package
```

### Data Merging

**Combine multiple datasets**:

1. Select data packages to merge
2. Click **[Merge Packages]**
3. Choose merge strategy:
   - **Concatenate**: Stack spectra (keep all)
   - **Average**: Mean of all spectra per group
   - **Interleave**: Alternate between datasets
4. Handle wavenumber mismatches:
   - **Interpolate**: Resample to common grid
   - **Crop**: Use common wavenumber range only
5. Click **Merge**

**Use Cases**:
- Combine multiple experimental batches
- Create larger training sets
- Merge technical replicates

### Data Splitting

**Split dataset into train/validation/test**:

1. Select data package
2. Click **[Split Dataset]**
3. Configure split ratios:
   - Training: 70%
   - Validation: 15%
   - Test: 15%
4. Choose split strategy:
   - **Random**: Random assignment
   - **Stratified**: Maintain group proportions
   - **Patient-level**: Keep all spectra from one patient together
5. Click **Split**

**Result**: Three new data packages created automatically

### Export Data

**Export for external use**:

In the current application:

- The **Data Package** page can export **metadata as JSON**.
- The **Analysis** page can export:
   - Plots: **PNG**, **SVG**
   - Data tables: **CSV**, **XLSX**, **JSON**, **TXT**, **PKL**

**Options**:

Available export options depend on the selected analysis method and output type.

### Batch Import

**Import multiple files at once**:

1. Click **[Batch Import]**
2. Select folder containing CSV files
3. Options:
   - **Recursive**: Include subfolders
   - **Pattern**: Filter by filename (e.g., `*.csv`)
   - **Auto-group**: Assign groups by folder name
4. Preview file list
5. Click **Import All**

**Progress**:

During batch import, the application shows a progress indicator with:

- Overall percent complete
- Processed/total file count
- Current file name
- Estimated time remaining

---

## Best Practices

### File Organization

**Recommended folder structure**:
```
data/
├── raw/
│   ├── batch1/
│   │   ├── healthy/
│   │   │   ├── patient_001.csv
│   │   │   └── patient_002.csv
│   │   └── disease/
│   │       ├── patient_101.csv
│   │       └── patient_102.csv
│   └── batch2/
│       └── ...
└── processed/
    └── ...
```

**Benefits**:
- Clear organization by batch and condition
- Easy batch import
- Automatic group assignment
- Simplified version control

### Naming Conventions

**Files**:
```
Good: patient_001_healthy.csv
Bad:  p1.csv

Good: disease_group_a_replicate_1.csv
Bad:  data.csv
```

**Groups**:
```
Good: Healthy_Control, Disease_GroupA, Disease_GroupB
Bad:  Group1, Group2, G3
```

### Quality Control

**Before analysis**:
1. ✓ Visual inspection of spectra
2. ✓ Check for outliers
3. ✓ Verify group assignments
4. ✓ Validate wavenumber calibration
5. ✓ Document any issues in metadata

**During project**:
- Keep raw data unchanged
- Version processed datasets
- Document preprocessing steps
- Backup regularly

---

## Troubleshooting

### Import Fails

**Error**: "Could not parse CSV file"

**Solutions**:
- Check delimiter (comma vs tab vs semicolon)
- Verify decimal separator (period vs comma)
- Check for non-numeric characters
- Use UTF-8 encoding

### Wavenumber Mismatch

**Error**: "Spectra have different wavenumber axes"

**Solutions**:
1. Use **Align Wavenumbers** tool
2. Interpolate to common grid
3. Crop to common range
4. Import separately and merge later

### Memory Issues

**Error**: "Out of memory during import"

**Solutions**:
- Import in smaller batches
- Close other applications
- Enable "Chunked Loading" in settings
- Use 64-bit version of application

### Missing Groups

**Error**: "No groups defined for classification"

**Solutions**:
1. Create groups first
2. Assign samples to groups
3. Verify group labels are correct
4. Check for unassigned samples

---

## See Also

- [Interface Overview](interface-overview.md) - Navigate the Data Package page
- [Preprocessing Guide](preprocessing.md) - Next step after import
- [FAQ - Data Import](../faq.md#data-questions) - Common questions
- [Troubleshooting](../troubleshooting.md#data-import-issues) - Detailed error solutions

---

**Next**: [Preprocessing Guide](preprocessing.md) →
