from qtpy.QtWidgets import (QWidget, QVBoxLayout, QTextEdit, QMessageBox)
from qtpy.QtCore import Signal, QObject


"""Defines: GEMspaWidget, GEMspaWorker, GEMspaLogWidget"""


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



