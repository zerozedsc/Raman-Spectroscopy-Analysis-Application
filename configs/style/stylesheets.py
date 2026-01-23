"""
Stylesheet definitions for the Raman Spectral Classifier application.
Organized by page/component for better maintainability.
"""

# Base styles used across components
BASE_STYLES = {
    "group_box": """
        QGroupBox {
            font-size: 14px;
            font-weight: 600;
            color: #2c3e50;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            margin-top: 8px;
            padding-top: 8px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 8px;
            background-color: #f8f9fa;
        }
    """,
    "input_field": """
        QLineEdit, QDoubleSpinBox, QSpinBox {
            padding: 8px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            font-size: 13px;
            background-color: white;
        }
        QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus {
            border-color: #0078d4;
            outline: none;
        }
    """,
    "combo_box": """
        QComboBox {
            padding: 8px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            background-color: white;
            font-size: 13px;
        }
        QComboBox::drop-down {
            border: none;
            width: 20px;
        }
        QComboBox::down-arrow {
            width: 12px;
            height: 12px;
        }
        /* Popup list (fixes black dropdown on some themes/platforms) */
        QComboBox QAbstractItemView {
            background-color: #ffffff;
            color: #212529;
            border: 1px solid #ced4da;
            selection-background-color: #e3f2fd;
            selection-color: #2c3e50;
            outline: 0;
        }
        QComboBox QAbstractItemView::item {
            padding: 6px 10px;
        }
    """,
    "primary_button": """
        QPushButton {
            background-color: #0078d4;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: 500;
            font-size: 13px;
        }
        QPushButton:hover {
            background-color: #106ebe;
        }
        QPushButton:pressed {
            background-color: #005a9e;
        }
    """,
    "secondary_button": """
        QPushButton {
            background-color: #ffffff;
            color: #6c757d;
            border: 1px solid #ced4da;
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 12px;
        }
        QPushButton:hover {
            background-color: #f8f9fa;
            border-color: #adb5bd;
        }
    """,
    "success_button": """
        QPushButton {
            background-color: #28a745;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 14px;
        }
        QPushButton:hover {
            background-color: #218838;
        }
        QPushButton:pressed {
            background-color: #1e7e34;
        }
        QPushButton:disabled {
            background-color: #6c757d;
            cursor: not-allowed;
        }
    """,
    "danger_button": """
        QPushButton {
            background-color: #dc3545;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: 500;
            font-size: 12px;
        }
        QPushButton:hover {
            background-color: #c82333;
        }
        QPushButton:pressed {
            background-color: #bd2130;
        }
    """,
    "list_widget": """
        QListWidget {
            border: 1px solid #ced4da;
            border-radius: 4px;
            background-color: white;
            padding: 4px;
        }
        QListWidget::item {
            padding: 6px;
            border-bottom: 1px solid #f1f3f4;
        }
        QListWidget::item:selected {
            background-color: #e3f2fd;
            color: #1976d2;
        }
    """,
    "scroll_area": """
        QScrollArea {
            border: none;
            background-color: white;
        }
        QScrollBar:vertical {
            background-color: #f1f1f1;
            width: 8px;
            border-radius: 4px;
        }
        QScrollBar::handle:vertical {
            background-color: #c1c1c1;
            border-radius: 4px;
            min-height: 20px;
        }
        QScrollBar::handle:vertical:hover {
            background-color: #a8a8a8;
        }
    """,
    "progress_bar": """
        QProgressBar {
            border: 1px solid #ced4da;
            border-radius: 4px;
            text-align: center;
            background-color: #f8f9fa;
            font-size: 12px;
        }
        QProgressBar::chunk {
            background-color: #28a745;
            border-radius: 3px;
        }
    """,
    "info_label": """
        QLabel {
            color: #666;
            font-size: 12px;
            padding: 8px;
            background: #f8f9fa;
            border-radius: 4px;
        }
    """,
    "parameter_label": """
        QLabel {
            font-weight: 500;
            color: #2c3e50;
        }
    """,
    "status_label": """
        QLabel {
            color: #666;
            font-size: 12px;
        }
    """,

    # Theme-consistent table styling for dialogs and views
    "table_widget": """
        QTableWidget {
            border: 1px solid #dee2e6;
            border-radius: 8px;
            background-color: white;
            alternate-background-color: #f8f9fa;
            selection-background-color: #e3f2fd;
            selection-color: #2c3e50;
            font-size: 13px;
        }
        QTableWidget::item {
            padding: 8px;
            border: none;
        }
        QTableWidget::item:selected {
            background-color: #e3f2fd;
            color: #2c3e50;
        }
        QHeaderView::section {
            background-color: #f8f9fa;
            padding: 10px 8px;
            border: none;
            border-bottom: 2px solid #dee2e6;
            font-weight: 600;
            color: #495057;
            font-size: 13px;
        }
    """,

    "tree_widget": """
        QTreeWidget {
            border: 1px solid #dee2e6;
            border-radius: 8px;
            background-color: white;
            font-size: 13px;
        }
        QTreeWidget::item {
            padding: 8px 6px;
            border-bottom: 1px solid #f1f3f5;
        }
        QTreeWidget::item:hover {
            background-color: #f8f9fa;
        }
        QTreeWidget::item:selected {
            background-color: #e3f2fd;
            color: #2c3e50;
        }
        QHeaderView::section {
            background-color: #f8f9fa;
            padding: 10px 8px;
            border: none;
            border-bottom: 2px solid #dee2e6;
            font-weight: 600;
            color: #495057;
            font-size: 13px;
        }
    """,
}

# Preprocessing Page specific styles
PREPROCESS_PAGE_STYLES = {
    "main_splitter": """
        QSplitter::handle {
            background-color: #ddd;
            border: 1px solid #bbb;
        }
        QSplitter::handle:hover {
            background-color: #0078d4;
        }
    """,
    "left_panel": """
        #leftPanel {
            background-color: #f8f9fa;
            border-right: 1px solid #dee2e6;
        }
    """,
    "pipeline_list": """
        QListWidget {
            border: 1px solid #ced4da;
            border-radius: 4px;
            background-color: white;
            padding: 4px;
        }
        QListWidget::item {
            padding: 8px;
            border-bottom: 1px solid #f1f3f4;
            background-color: #f8f9fa;
            margin-bottom: 2px;
            border-radius: 3px;
        }
        QListWidget::item:selected {
            background-color: #e3f2fd;
            color: #1976d2;
            border: 1px solid #1976d2;
        }
    """,
    "dataset_list": """
        QListWidget {
            border: 1px solid #ced4da;
            border-radius: 4px;
            background-color: white;
            padding: 4px;
        }
        QListWidget::item {
            padding: 10px;
            border-bottom: 1px solid #f1f3f4;
            background-color: #ffffff;
            margin-bottom: 2px;
            border-radius: 3px;
        }
        QListWidget::item:hover {
            background-color: #f5f5f5;
        }
        QListWidget::item:selected {
            background-color: #1565c0;
            color: white;
            border: 2px solid #0d47a1;
            font-weight: 500;
        }
        QListWidget::item:selected:hover {
            background-color: #1976d2;
        }
    """,
    "visualization_widget": """
        MatplotlibWidget {
            border: 1px solid #dee2e6;
            border-radius: 4px;
            background-color: white;
        }
    """,
    "parameter_input": """
        QLineEdit, QDoubleSpinBox, QSpinBox {
            padding: 6px;
        }
        QComboBox {
            padding: 6px;
        }
    """,
}

# Workspace Page specific styles
WORKSPACE_PAGE_STYLES = {
    "main_container": """
        QWidget {
            background-color: #f8f9fa;
        }
    """,
    "project_card": """
        QFrame {
            background-color: white;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 16px;
        }
        QFrame:hover {
            border-color: #0078d4;
            background-color: #f8fcff;
        }
    """,
}

# Data Package Page specific styles
DATA_PACKAGE_PAGE_STYLES = {
    "dataset_item": """
        QFrame {
            background-color: white;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 12px;
        }
        QFrame:hover {
            border-color: #0078d4;
        }
    """,
    "import_button": """
        QPushButton {
            background-color: #17a2b8;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            font-weight: 600;
        }
        QPushButton:hover {
            background-color: #138496;
        }
    """,
}

# ML Training Page specific styles
ML_TRAINING_PAGE_STYLES = {
    "model_card": """
        QFrame {
            background-color: white;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 16px;
            margin: 8px;
        }
        QFrame:hover {
            border-color: #28a745;
        }
    """,
    "training_progress": """
        QProgressBar {
            border: 2px solid #28a745;
            border-radius: 6px;
            text-align: center;
            background-color: #f8f9fa;
            font-size: 14px;
            font-weight: 500;
        }
        QProgressBar::chunk {
            background-color: #28a745;
            border-radius: 4px;
        }
    """,
}

# Real-time Analysis Page specific styles
REALTIME_PAGE_STYLES = {
    "control_panel": """
        QFrame {
            background-color: #343a40;
            border-radius: 8px;
            padding: 16px;
        }
    """,
    "start_button": """
        QPushButton {
            background-color: #28a745;
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 16px;
        }
        QPushButton:hover {
            background-color: #218838;
        }
        QPushButton:pressed {
            background-color: #1e7e34;
        }
    """,
    "stop_button": """
        QPushButton {
            background-color: #dc3545;
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 16px;
        }
        QPushButton:hover {
            background-color: #c82333;
        }
        QPushButton:pressed {
            background-color: #bd2130;
        }
    """,
    "status_indicator": """
        QLabel {
            background-color: #28a745;
            color: white;
            border-radius: 12px;
            padding: 6px 12px;
            font-weight: 500;
        }
    """,
    "selection_card": """
        #selectionCard {
            background-color: white;
            border: 1px solid #e1e4e8;
            border-radius: 8px;
            padding: 12px;
        }
    """,
    "modern_pipeline_group": """
        QGroupBox#modernPipelineGroup {
            background-color: white;
            border: 1px solid #dee2e6;
            border-radius: 10px;
            margin-top: 10px;
            padding-top: 10px;
            font-weight: 600;
            font-size: 14px;
        }
        QGroupBox#modernPipelineGroup::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0px 8px;
            color: #2c3e50;
        }
    """,
}

PREPROCESS_PAGE_STYLES_2 = """
            QPushButton#iconButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ffffff, stop:1 #f0f0f0);
                border: 2px solid #d0d0d0;
                border-radius: 20px;
                font-size: 18px;
                font-weight: 600;
                color: #555555;
                min-width: 36px;
                max-width: 36px;
                min-height: 36px;
                max-height: 36px;
                padding: 0px;
                text-align: center;
            }
            
            QPushButton#iconButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8f8f8, stop:1 #e8e8e8);
                border: 2px solid #0078d4;
                color: #0078d4;
            }
            
            QPushButton#iconButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #e0e0e0, stop:1 #d0d0d0);
                border: 2px solid #005a9e;
                color: #005a9e;
            }
            
            QPushButton#iconButton:disabled {
                background: #f8f8f8;
                border: 2px solid #e0e0e0;
                color: #c0c0c0;
            }
            
            QPushButton#refreshButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #e3f2fd, stop:1 #bbdefb);
                border: 2px solid #90caf9;
                border-radius: 8px;
                color: #1565c0;
                font-weight: 600;
                font-size: 14px;
                padding: 10px 16px;
                min-height: 36px;
            }
            
            QPushButton#refreshButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #bbdefb, stop:1 #90caf9);
                border-color: #42a5f5;
            }
            
            QPushButton#refreshButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #90caf9, stop:1 #64b5f6);
            }
            
            QPushButton#ctaButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0078d4, stop:1 #005a9e);
                border: 2px solid #0078d4;
                border-radius: 8px;
                color: white;
                font-weight: 600;
                font-size: 14px;
                padding: 12px 20px;
                min-height: 40px;
            }
            
            QPushButton#ctaButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #106ebe, stop:1 #004578);
                border-color: #106ebe;
            }
            
            QPushButton#ctaButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #005a9e, stop:1 #004578);
            }
            
            QPushButton#ctaButton:disabled {
                background: #cccccc;
                border-color: #cccccc;
                color: #666666;
            }
            
            QPushButton#cancelButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ffebee, stop:1 #ffcdd2);
                border: 2px solid #f44336;
                border-radius: 8px;
                color: #c62828;
                font-weight: 600;
                font-size: 14px;
                padding: 10px 16px;
                min-height: 36px;
            }
            
            QPushButton#cancelButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ffcdd2, stop:1 #ef9a9a);
                border-color: #d32f2f;
            }
            
            QPushButton#cancelButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ef9a9a, stop:1 #e57373);
            }
        """


def get_main_stylesheet(font_family: str) -> str:
    """
    Generates the main application stylesheet with a dynamically provided font family.
    Combines all base styles and page-specific styles into a single comprehensive stylesheet.
    """

    # Combine all base styles
    base_styles = "\n".join(BASE_STYLES.values())

    # Combine all page-specific styles
    page_styles = "\n".join(
        [
            "\n".join(PREPROCESS_PAGE_STYLES.values()),
            PREPROCESS_PAGE_STYLES_2,
            "\n".join(WORKSPACE_PAGE_STYLES.values()),
            "\n".join(DATA_PACKAGE_PAGE_STYLES.values()),
            "\n".join(ML_TRAINING_PAGE_STYLES.values()),
            "\n".join(REALTIME_PAGE_STYLES.values()),
        ]
    )

    # Main application stylesheet
    main_stylesheet = f"""
        QWidget {{
            font-family: {font_family};
            font-size: 14px;
            background-color: #f8f9fa;
            color: #2c3e50;
        }}
        
        /* --- Home Page Layout --- */
        #homePage {{
            background-color: #f8f9fa;
        }}
        
        #homeSidebar {{
            background-color: #ffffff;
            border-right: 1px solid #e9ecef;
        }}
        
        #homeContentArea {{
            background-color: #f8f9fa;
        }}
        
        /* --- Home Page Typography --- */
        #homeTitle {{
            font-size: 28px;
            font-weight: 700;
            color: #1a1a1a;
            margin-bottom: 4px;
        }}
        
        #homeSubtitle {{
            font-size: 14px;
            font-weight: 400;
            color: #6c757d;
            margin-bottom: 0px;
        }}
        
        #recentProjectsHeader {{
            font-size: 18px;
            font-weight: 600;
            color: #1a1a1a;
        }}
        
        /* --- Action Cards --- */
        #actionCard {{
            background-color: #ffffff;
            border: 1px solid #e9ecef;
            border-radius: 12px;
            padding: 0px;
        }}
        
        #actionCard[hover="true"] {{
            border-color: #0078d4;
            background-color: #f8fcff;
        }}
        
        #actionCard #cardTitle {{
            font-size: 16px;
            font-weight: 600;
            color: #1a1a1a;
            margin-bottom: 4px;
        }}
        
        #actionCard #cardDescription {{
            font-size: 13px;
            color: #6c757d;
            line-height: 1.4;
        }}
        
        /* --- Recent Projects List --- */
        #recentProjectsList {{
            background-color: transparent;
            border: none;
            outline: none;
        }}
        
        #recentProjectsList::item {{
            border: none;
            background-color: transparent;
            padding: 0px;
            margin-bottom: 8px;
        }}
        
        #recentProjectsList::item:selected {{
            background-color: transparent;
        }}
        
        #recentProjectItem {{
            background-color: #ffffff;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            margin: 0px;
        }}
        
        #recentProjectItem:hover {{
            border-color: #0078d4;
            background-color: #f8fcff;
        }}
        
        #recentProjectItem #projectName {{
            font-size: 15px;
            font-weight: 600;
            color: #1a1a1a;
            background-color: transparent;
        }}
        
        #recentProjectItem #projectTime {{
            font-size: 12px;
            color: #6c757d;
            background-color: transparent;
        }}
        
        /* --- App Tab Bar Styles --- */
        #appTabBar {{
            background-color: #ffffff;
            border-bottom: 1px solid #e9ecef;
        }}
        
        #appTabBar QPushButton {{
            font-size: 14px;
            font-weight: 500;
            color: #6c757d;
            padding: 8px 24px;
            border: none;
            background-color: transparent;
            border-bottom: 2px solid transparent;
        }}
        
        #appTabBar QPushButton:hover {{
            color: #1a1a1a;
            background-color: #f8f9fa;
        }}
        
        /* --- COLOR CHANGE --- */
        /* This changes the style for the selected tab button. */
        #appTabBar QPushButton:checked {{
            color: #2c3e50;
            border-bottom: 2px solid #adb5bd;
            background-color: #e9ecef; /* This is the gray background for the tab */
        }}
        
        /* --- General Widget Styles --- */
        QGroupBox {{
            font-size: 16px;
            font-weight: 600;
            color: #1a1a1a;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            margin-top: 12px;
            background-color: #ffffff;
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 8px;
            background-color: #ffffff;
        }}
        
        QLineEdit, QTextEdit {{
            border: 1px solid #ced4da;
            border-radius: 6px;
            padding: 8px 12px;
            background-color: #ffffff;
            font-size: 14px;
        }}
        
        QLineEdit:focus, QTextEdit:focus {{
            border-color: #0078d4;
            outline: none;
        }}
        
        QPushButton {{
            background-color: #ffffff;
            color: #1a1a1a;
            border: 1px solid #ced4da;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: 500;
        }}
        
        QPushButton:hover {{
            background-color: #f8f9fa;
            border-color: #adb5bd;
        }}
        
        QPushButton:pressed {{
            background-color: #e9ecef;
        }}
        
        #ctaButton {{
            background-color: #0078d4;
            color: #ffffff;
            border: 1px solid #0078d4;
            font-weight: 600;
        }}
        
        #ctaButton:hover {{
            background-color: #106ebe;
            border-color: #106ebe;
        }}
        
        /* --- Scrollbar Styles --- */
        QScrollBar:vertical {{
            background-color: #f8f9fa;
            width: 8px;
            border-radius: 4px;
        }}
        
        QScrollBar::handle:vertical {{
            background-color: #ced4da;
            border-radius: 4px;
            min-height: 20px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background-color: #adb5bd;
        }}
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            border: none;
            background: none;
        }}
        
        /* --- Matplotlib Toolbar --- */
        /* This targets the QToolBar inside our custom matplotlib widget */
        #matplotlibWidget QToolBar {{
            background-color: #e9ecef; /* A light gray background for the toolbar */
            border-bottom: 1px solid #ced4da;
        }}

        #matplotlibWidget QToolBar QToolButton {{
            background-color: transparent;
            border-radius: 4px;
            padding: 4px;
            margin: 1px;
        }}

        #matplotlibWidget QToolBar QToolButton:hover {{
            background-color: #ced4da; /* Slightly darker on hover */
        }}

        #matplotlibWidget QToolBar QToolButton:checked {{
            background-color: #adb5bd; /* Even darker when a tool is active (like zoom) */
        }}
        
    """

    return main_stylesheet


def get_page_style(page_name, style_name):
    """Get a page-specific style by page and style name."""
    page_styles = {
        "preprocess": PREPROCESS_PAGE_STYLES,
        "workspace": WORKSPACE_PAGE_STYLES,
        "data_package": DATA_PACKAGE_PAGE_STYLES,
        "ml_training": ML_TRAINING_PAGE_STYLES,
        "realtime": REALTIME_PAGE_STYLES,
    }
    return page_styles.get(page_name, {}).get(style_name, "")


def combine_styles(*styles):
    """Combine multiple styles into one."""
    return "\n".join(filter(None, styles))


# Utility functions to get styles
def get_base_style(style_name):
    """Get a base style by name."""
    return BASE_STYLES.get(style_name, "")
