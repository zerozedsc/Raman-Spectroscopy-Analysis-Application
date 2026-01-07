"""
Verify Create Group Fix
Check if the code is correctly configured for the multi-group dialog
"""

import os
import re

def check_file(filepath, patterns):
    """Check if file contains expected patterns"""
    print(f"\n{'='*60}")
    print(f"Checking: {filepath}")
    print(f"{'='*60}")
    
    if not os.path.exists(filepath):
        print(f"❌ FILE NOT FOUND: {filepath}")
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    all_passed = True
    for pattern_name, pattern in patterns.items():
        if re.search(pattern, content, re.MULTILINE | re.DOTALL):
            print(f"✅ {pattern_name}")
        else:
            print(f"❌ {pattern_name}")
            all_passed = False
    
    return all_passed

# Check group_assignment_table.py
group_table_patterns = {
    "Direct connection (no lambda)": r'multi_group_action\.triggered\.connect\(self\._on_create_groups_clicked\)',
    "Method exists": r'def _on_create_groups_clicked\(self\):',
    "Opens multi-group dialog": r'def _open_multi_group_dialog\(self\):',
    "Imports MultiGroupCreationDialog": r'from \.multi_group_dialog import MultiGroupCreationDialog',
    "Creates dialog instance": r'dialog = MultiGroupCreationDialog\(',
    "Debug statements present": r'\[DEBUG\]',
}

# Check multi_group_dialog.py exists
multi_dialog_file = r"pages\analysis_page_utils\multi_group_dialog.py"

print("\n" + "="*60)
print("CREATE GROUP FIX VERIFICATION")
print("="*60)

# Check main file
result1 = check_file(
    r"pages\analysis_page_utils\group_assignment_table.py",
    group_table_patterns
)

# Check dialog exists
print(f"\n{'='*60}")
print(f"Checking: {multi_dialog_file}")
print(f"{'='*60}")
dialog_exists = os.path.exists(multi_dialog_file)
if dialog_exists:
    print(f"✅ MultiGroupCreationDialog file exists")
else:
    print(f"❌ MultiGroupCreationDialog file NOT FOUND")

# Check localization
print(f"\n{'='*60}")
print("Checking: Localization files")
print(f"{'='*60}")

ja_loc = r"assets\locales\ja.json"
if os.path.exists(ja_loc):
    with open(ja_loc, 'r', encoding='utf-8') as f:
        content = f.read()
    if '複数グループを一度に作成' in content:
        print("✅ Japanese localization: '複数グループを一度に作成' (NEW dialog title)")
    else:
        print("❌ Japanese localization missing")
else:
    print("❌ ja.json not found")

# Final summary
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")

if result1 and dialog_exists:
    print("✅ All checks PASSED - Create Group should use NEW multi-group dialog")
    print("\n⚠️  IMPORTANT: You MUST completely restart the application!")
    print("    1. Close the running application completely")
    print("    2. Clear Python cache (already done)")
    print("    3. Run: uv run main.py")
    print("    4. Click 'Create Groups' button")
    print("    5. You should see: '複数グループを一度に作成' (Create Multiple Groups at Once)")
else:
    print("❌ Some checks FAILED - Fix may not work correctly")
    print("\nPlease check the errors above.")

print(f"\n{'='*60}\n")
