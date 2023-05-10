from qtpy.QtWidgets import (QWidget, QVBoxLayout, QTextEdit, QMessageBox)
from qtpy.QtCore import Signal, QObject, QThread
from ._gemspa_data_views import GEMspaTableWindow, GEMspaPlottingWindow


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

    def run(self):
        """Exec the data processing"""

        self.finished.emit()


class GEMspaWidget(QWidget):
    """Definition of a GEMspa napari widget

    """

    name = 'GEMspaWidget'

    def __init__(self, napari_viewer):
        super().__init__()

        self.viewer = napari_viewer
        self.thread = None
        self.worker = None

        # viewers for feature properties
        self.properties_viewers = []

        # viewers for the graphical output
        self.plots_viewers = []

        self._input_values = {}
        self._required_inputs = []

    def closeEvent(self, event):
        self._delete_viewers
        event.accept()  # let the window close

    def _delete_viewers(self):
        while self.plots_viewers:
            viewer = self.plots_viewers.pop()
            viewer.close()
            viewer.deleteLater()

        while self.properties_viewers:
            viewer = self.properties_viewers.pop()
            viewer.close()
            viewer.deleteLater()

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

    @staticmethod
    def _convert_to_float(value):
        if value:
            return float(value)
        else:
            return None

    @staticmethod
    def _convert_to_int(value):
        if value:
            return int(value)
        else:
            return None

    def _new_plots_viewer(self, title='Plot view'):
        i = len(self.plots_viewers)
        self.plots_viewers.append(GEMspaPlottingWindow(self.viewer))
        self.plots_viewers[i].setWindowTitle(title)
        return self.plots_viewers[i]

    def _new_properties_viewer(self, title='Table view'):
        i = len(self.properties_viewers)
        self.properties_viewers.append(GEMspaTableWindow(self.viewer))
        self.properties_viewers[i].setWindowTitle(title)
        return self.properties_viewers[i]

    def start_task(self, layer_name, log_widget):

        # Perform startup tasks and start thread: worker must be initialized before this function is called

        # connect worker log signal to the GEMspaLogWidget all_log method
        self.worker.log.connect(log_widget.add_log)

        # Thread for this worker
        self.thread = QThread()
        self.worker.moveToThread(self.thread)

        # when thread is started, worker is run with the current state of the widget as input
        self.thread.started.connect(lambda: self.worker.run(self.state(layer_name)))

        # when the worker sends update_data signal, update_data of the widget will execute to update the GUI
        self.worker.update_data.connect(self.update_data)

        # Cleanup (as chatGPT suggested)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def check_inputs(self):
        valid = True
        for key in self._input_values.keys():
            text = self._input_values[key].text()
            if key in self._required_inputs or text:
                # if input is not blank, check it is a number
                # except for Diameter, it cannot be blank
                try:
                    _ = float(text)
                except ValueError:
                    self.show_error(f"{key} input must be a number")
                    valid = False
        return valid


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



