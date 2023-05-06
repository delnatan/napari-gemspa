from ._gemspa_locate_widget import GEMspaLocateWidget
from ._gemspa_widget import GEMspaLogWidget
from qtpy.QtWidgets import (QWidget, QLabel, QPushButton, QVBoxLayout, QTabWidget)
from qtpy import QtCore


"""Defines: GEMspaPlugin"""


class GEMspaPlugin(QWidget):

    """Definition of a GEMspa napari plugin

    Parameters
    ----------
    napari_viewer: Viewer
        Napari viewer

    """

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.title = 'GEMspa plugin'
        self.main_tab = None
        self.selected_widget = None

        # Run button will start the currently active widget
        self.run_btn = None

        # Log widget displays all outputs
        self.log_widget = GEMspaLogWidget()

    def init_ui(self):
        """Initialize the plugin graphical interface
        Add a run button and a log widget to the plugin
        """

        layout = QVBoxLayout()

        # Title of widget
        title_label = QLabel(self.title)
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title_label)

        # Tab widget for switching between subwidgets
        self.main_tab = QTabWidget()
        layout.addWidget(self.main_tab)

        # Subwidget 1 : Localization
        widget = GEMspaLocateWidget(self.viewer)
        widget.init_ui()
        widget.worker.log.connect(self.log_widget.add_log)
        self.main_tab.addTab(widget, "Locate features")

        # Subwidget 2 : Link features
        #widget = GEMspaLinkWidget(self.viewer)
        #widget.init_ui()
        #widget.worker.log.connect(self.log_widget.add_log)
        #self.main_tab.addTab(widget, "Link features")

        # Subwidget 3 : Analysis
        # widget = GEMspaAnalyzeWidget(self.viewer)
        # widget.init_ui()
        # widget.worker.log.connect(self.log_widget.add_log)
        # self.main_tab.addTab(widget, "Analyze tracks")

        # Connect the signal to a slot method
        self.main_tab.currentChanged.connect(self.on_current_tab_changed)

        # Run button
        self.run_btn = QPushButton('Run')
        self.run_btn.released.connect(self.run)
        layout.addWidget(self.run_btn)

        # Log widget
        layout.addWidget(self.log_widget)

        self.setLayout(layout)
        self.selected_widget = self.main_tab.currentWidget()

    def on_current_tab_changed(self, index):
        self.selected_widget = self.main_tab.currentWidget()

    def run(self):
        """Check inputs and start thread"""
        if self.selected_widget.check_inputs():
            self.selected_widget.start_task()
