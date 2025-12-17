---
Title: Analysis Page Dataset Selector Card Refresh
Date: 2025-11-23
Author: agent/auto
Tags: analysis-ui, dataset-selection, localization
RelatedFiles:
  - pages/analysis_page_utils/method_view.py
  - assets/locales/en.json
  - assets/locales/ja.json
---

## Summary
- Replaced the Select Dataset QGroupBox with a modern card-style QFrame that better matches the design reference, including iconographic title, badges, and descriptive subtitle.
- Added dynamic badges that reflect the required selection mode (single vs multi) and minimum dataset count based on method metadata.
- Introduced new localization strings for the card subtitle, badge labels, and minimum dataset indicator in both English and Japanese resource files.

## Implementation Notes
- New layout maintains existing dataset toggle and stacked widget content while encapsulating them in the refreshed card container for improved spacing and readability.
- Styling uses rounded borders and updated typography consistent with other Analysis page improvements.
- Localization keys: `ANALYSIS_PAGE.dataset_selection_subtitle`, `ANALYSIS_PAGE.dataset_mode_multi`, `ANALYSIS_PAGE.dataset_mode_single`, `ANALYSIS_PAGE.min_datasets`.

## Testing
- Manual UI inspection within the Analysis page to ensure layout integrity and correct badge text for both single and multi dataset methods.
- Verified localization fallbacks in English and Japanese JSON files.
