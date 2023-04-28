from qtpy.QtWidgets import (QWidget, QLabel, QPushButton, QVBoxLayout, QTextEdit, QMessageBox)
from qtpy import QtCore
from qtpy.QtCore import Signal, QThread, QObject


class GEMspaWidget(QWidget):
    """Definition of a GEMspa napari widget

    This interface implements three methods
    - show_error: to display a user input error
    - check_inputs (abstract): to check all the user input from the plugin widget
    - state (abstract): to get the plugin widget state: the user inputs values set in the widget

    """

    enable = Signal(bool)

    def __init__(self):
        super().__init__()

    @staticmethod
    def show_error(message):
        """Display an error message in a QMessage box

        Parameters
        ----------
        message: str
            Error message

        """
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)

        msg.setText(message)
        msg.setWindowTitle("GEMspa error")
        msg.exec_()

    def check_inputs(self):
        """Check the user input in this widget

        Returns:
            True if no error, False if at least one input contains an error.

        """
        raise NotImplementedError()

    def state(self):
        """Return the current state of the widget

        The state in input values displayed in the widget.

        Returns:
            dict: a dictionary containing the widget inputs

        """
        raise NotImplementedError()


class GEMspaWorker(QObject):
    """Definition of a GEMspaWorker

    The worker runs the calculation (using the run method) using the user inputs
    from the plugin widget interface (GEMspaWidget)

    """

    # Worker can send these signals
    finished = Signal()
    log = Signal(str)

    def __init__(self, napari_viewer, widget):
        super().__init__()
        self.viewer = napari_viewer
        self.widget = widget

    def state(self):
        """Get the states from the GEMSpaWidget"""
        self.widget.state()

    def run(self):
        """Exec the data processing"""
        raise NotImplementedError()


class GEMspaLogWidget(QWidget):
    """Widget to log the GEMspa plugin messages in the graphical interface"""
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.log_area = QTextEdit()
        layout.addWidget(self.log_area)
        self.setLayout(layout)

    def add_log(self, value: str):
        """Callback to add a new message in the log area"""
        self.log_area.append(value)

    def clear_log(self):
        """Callback to clear all the log area"""
        self.log_area.clear()


class GEMspaPlugin(QWidget):
    """Definition of a GEMspa napari plugin

    Parameters
    ----------
    napari_viewer: Viewer
        Napari viewer

    Attributes
    ----------
    worker: GEMspaWorker
        Instance of the plugin worker

    widget: GEMspaInputWidget
        Instance of the plugin widget
    """

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        # thread
        self.title = 'Default plugin'
        self.worker = None
        self.widget = None
        self.fill_widget_resize = 1
        self.thread = QThread()

        # GUI
        self.run_btn = None
        self.log_widget = GEMspaLogWidget()

    def init_gui(self):
        """Initialize the plugin graphical interface
        Add a run button and a log widget to the plugin
        """

        layout = QVBoxLayout()

        # Title of widget
        title_label = QLabel(self.title)
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title_label)

        # The main widget
        layout.addWidget(self.widget)

        # Run button
        self.run_btn = QPushButton('Run')
        self.run_btn.released.connect(self.run)
        layout.addWidget(self.run_btn)

        # Log widget
        layout.addWidget(self.log_widget)

        # not sure what this is
        layout.addWidget(QWidget(), self.fill_widget_resize, QtCore.Qt.AlignTop)

        self.setLayout(layout)

        # Connects:

        # move QObject to thread (a QThread() object)
        self.worker.moveToThread(self.thread)

        # when thread sends started signal, worker.run is executed
        self.thread.started.connect(self.worker.run)

        # when worker sends log signal (str), log_widget.add_log is executed
        self.worker.log.connect(self.log_widget.add_log)

        # when worker sends finished signal, thread.quit is executed and set_outputs is executed
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.set_outputs)

    def run(self):
        """Start the worker in a new thread"""
        if self.widget.check_inputs():
            self.thread.start()

    def set_enable(self, mode: bool):
        """Callback called to disable the run button when the inputs layers are not available"""
        self.run_btn.setEnabled(mode)

    def set_outputs(self):
        """Call the worker set_outputs method to set the plugin outputs to napari layers"""
        self.worker.set_outputs()


