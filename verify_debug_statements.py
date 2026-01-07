"""
Final verification that all debug statements are in place
"""

import os

filepath = r"pages\analysis_page_utils\group_assignment_table.py"

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Check for all critical DEBUG statements
checks = [
    ("Button initialization", "INITIALIZING CREATE GROUPS BUTTON"),
    ("Button clicked", "CREATE GROUPS BUTTON CLICKED"),
    ("Opening dialog", "OPENING MULTI-GROUP DIALOG"),
    ("Step 1", "Step 1: Attempting to import"),
    ("Step 2", "Step 2: Creating dialog instance"),
    ("Step 3", "Step 3: Calling dialog.exec()"),
    ("Dialog title", "Dialog window title"),
    ("Timestamp", "Timestamp:"),
]

print("\n" + "="*70)
print("VERIFICATION: DEBUG STATEMENTS IN group_assignment_table.py")
print("="*70)

all_found = True
for check_name, check_string in checks:
    if check_string in content:
        print(f"✅ {check_name}: '{check_string[:40]}...'")
    else:
        print(f"❌ {check_name}: NOT FOUND")
        all_found = False

print("="*70)

if all_found:
    print("\n✅ ALL DEBUG STATEMENTS ARE IN PLACE!")
    print("\n📋 NEXT STEPS:")
    print("1. Close the application completely")
    print("2. Restart: uv run main.py")
    print("3. Go to Analysis page → Classification mode")
    print("4. Click '➕ Create Groups' button")
    print("5. COPY ALL CONSOLE OUTPUT and send to me")
    print("\nThe console will show EXACTLY where the problem is.")
else:
    print("\n❌ Some DEBUG statements missing!")

print("\n" + "="*70 + "\n")
