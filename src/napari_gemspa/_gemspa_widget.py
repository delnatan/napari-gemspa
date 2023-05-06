from qtpy.QtWidgets import (QWidget, QVBoxLayout, QTextEdit, QMessageBox)
from qtpy.QtCore import Signal, Slot, QObject, QThread


"""Defines: GEMspaWidget, GEMspaWorker, GEMspaLogWidget"""


class GEMspaWorker(QObject):
    """Definition of a GEMspaWorker

    Receives and Sends input/output as a dictionary

    """

    # Worker can send these signals
    finished = Signal()
    update_data = Signal(dict)
    log = Signal(str)

    def __init__(self):
        super().__init__()

    @Slot(dict)
    def run(self, state):
        """Exec the data processing"""

        self.finished.emit()


class GEMspaWidget(QWidget):
    """Definition of a GEMspa napari widget

    """

    def __init__(self, napari_viewer):
        super().__init__()

        self.viewer = napari_viewer
        self.thread = QThread()

        self.worker = GEMspaWorker()
        self.state = dict()

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

    def start_task(self):

        self.worker.moveToThread(self.thread)

        # when thread is started, worker is run with the current state of the widget as input
        self.thread.started.connect(lambda: self.worker.run(self.state()))

        # when the worker sends update_data signal, update_data of the widget will execute to update the GUI
        self.worker.update_data.connect(self.update_data)

        # Cleanup (as chatGPT suggested)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def check_inputs(self):
        """Check the user input in this widget

        Returns:
            True if no error, False if at least one input contains an error.

        """

        raise NotImplementedError

    def state(self):
        """Return the current state of the widget

        """

        return NotImplementedError

    def update_data(self, out_data):
        """Update the data output from the worker to the GUI

        """

        return NotImplementedError


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



