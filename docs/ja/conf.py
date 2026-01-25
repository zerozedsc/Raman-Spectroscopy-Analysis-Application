# Configuration file for Japanese documentation (ja)
# This file extends the English configuration with Japanese-specific settings

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath('../..'))

# Import base configuration from English docs
import importlib.util
spec = importlib.util.spec_from_file_location("base_conf", "../en/conf.py")
base_conf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_conf)

# Import all settings from English configuration
for attr in dir(base_conf):
    if not attr.startswith('_'):
        globals()[attr] = getattr(base_conf, attr)

# Override language-specific settings
language = 'ja'

# Override project metadata for Japanese
project = 'ラマン分光分析アプリケーション'
html_title = 'ラマン分光分析アプリケーション ドキュメント'

# Japanese-specific search
html_search_language = 'ja'

# Japanese-specific theme options
html_theme_options.update({
    'navigation_depth': 4,
    'titles_only': False,
})

# Locale directory (only used if you manage translations via gettext catalogs)
locale_dirs = ['../locale/']
gettext_compact = False
gettext_uuid = True
