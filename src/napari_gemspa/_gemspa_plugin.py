from ._gemspa_locate_widget import GEMspaLocateWidget, GEMspaLocateWorker
from ._gemspa_widget import GEMspaLogWidget
from qtpy.QtWidgets import (QWidget, QLabel, QPushButton, QVBoxLayout, QTextEdit, QMessageBox, QTabWidget)
from qtpy import QtCore
from qtpy.QtCore import Signal, QThread, QObject


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

        # thread
        self.title = 'GEMspa plugin'

        # Subwidget 1 : Localization
        self.locate_widget = GEMspaLocateWidget(self.viewer)
        self.locate_worker = GEMspaLocateWorker(self.viewer, self.locate_widget)
        self.locate_thread = QThread()

        self.link_worker = None
        self.link_widget = None
        self.link_thread = QThread()

        self.analyze_worker = None
        self.analyze_widget = None
        self.analyze_thread = QThread()

        # Run button will execute functionality for the currently active tab
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
        main_tab = QTabWidget()
        layout.addWidget(main_tab)

        # Subwidget 1 : Localization
        main_tab.addTab(self.locate_widget, "Locate features")

        # Subwidget 2 : Link features
        main_tab.addTab(self.link_widget, "Link features")

        # Subwidget 3 : Analysis
        main_tab.addTab(self.analyze_widget, "Analysis")

        # Run button
        self.run_btn = QPushButton('Run')
        self.run_btn.released.connect(self.run)
        layout.addWidget(self.run_btn)

        # Log widget
        layout.addWidget(self.log_widget)

        self.setLayout(layout)

        # Connects:

        # move QObject to thread (a QThread() object)
        # when thread sends started signal, worker.run is executed
        self.locate_worker.moveToThread(self.locate_thread)
        self.locate_thread.started.connect(self.locate_worker.run)

        self.link_worker.moveToThread(self.link_thread)
        self.link_thread.started.connect(self.link_worker.run)

        self.analyze_worker.moveToThread(self.analyze_thread)
        self.analyze_thread.started.connect(self.analyze_worker.run)

        # when any worker sends log signal (str), log_widget.add_log is executed
        self.locate_worker.log.connect(self.log_widget.add_log)
        self.link_worker.log.connect(self.log_widget.add_log)
        self.analyze_worker.log.connect(self.log_widget.add_log)

        # when worker sends finished signal, thread.quit is executed
        self.locate_worker.finished.connect(self.locate_thread.quit)
        self.link_worker.finished.connect(self.link_thread.quit)
        self.analyze_worker.finished.connect(self.analyze_thread.quit)

        # TODO: move this functionality to the worker
        # self.locate_worker.finished.connect(self.set_outputs)
        # self.link_worker.finished.connect(self.set_outputs)
        # self.analyze_worker.finished.connect(self.set_outputs)

    def run(self):
        """Start the worker in a new thread"""

        # TODO: check what tab is currently selected, then, check inputs and start thread
        if self.widget.check_inputs():
            self.thread.start()

    def set_enable(self, mode: bool):
        """Callback called to disable the run button when the inputs layers are not available"""

        # TODO: check what tab is currently selected, then, check needed layers are available and if not, disable 
        self.run_btn.setEnabled(mode)

    # def set_outputs(self):
    #     """Call the worker set_outputs method to set the plugin outputs to napari layers"""
    #     self.worker.set_outputs()


