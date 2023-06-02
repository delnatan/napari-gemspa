"""Defines: GEMspaWidget, GEMspaWorker, GEMspaLogWidget"""

import numpy as np
import pandas as pd
from qtpy.QtCore import QObject, QThread, Signal
from qtpy.QtWidgets import QGridLayout, QLabel, QTextEdit, QVBoxLayout, QWidget

from ._gemspa_data_views import GEMspaPlottingWindow, GEMspaTableWindow
from ._utils import show_error


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

    @staticmethod
    def _make_trackpy_table(layer_type, data, props):
        req_cols = ["mass", "size", "ecc", "signal", "raw_mass", "ep"]
        for col in req_cols:
            if col not in props.keys():
                return None
        if layer_type == "points":
            df = pd.DataFrame()
            if data.shape[1] == 2:
                i = 0
            elif data.shape[1] == 3:
                df["frame"] = data[:, 0]
                i = 1
            else:  # data.shape[1] >= 4
                df["frame"] = data[:, 0]
                df["z"] = data[:, 1]
                i = 2
            df["y"] = data[:, i]
            df["x"] = data[:, i + 1]
            for col in props.keys():
                df[col] = props[col]
        elif layer_type == "tracks":
            df = pd.DataFrame()
            df["particle"] = data[:, 0]
            df["frame"] = data[:, 1]
            if data.shape[1] == 4:
                i = 2
            else:  # data.shape[1] >= 5
                df["z"] = data[:, 2]
                i = 3
            df["y"] = data[:, i]
            df["x"] = data[:, i + 1]
            for col in props.keys():
                if col != "track_id":
                    df[col] = props[col]
        else:
            raise ValueError(f"Invalid layer type: {layer_type}")

        return df

    def run(self):
        """Exec the data processing"""

        self.finished.emit()


class GEMspaWidget(QWidget):
    """Definition of a GEMspa napari widget"""

    name = "GEMspaWidget"

    def __init__(self, napari_viewer, title=None):
        super().__init__()

        self.viewer = napari_viewer
        self.title = title
        self.thread = None
        self.worker = None

        # viewers for feature properties
        self.properties_viewers = []

        # viewers for the graphical output
        self.plots_viewers = []

        self.display_table_view = False

        self._input_values = {}
        self._required_inputs = []

    def init_ui(self):
        layout = QVBoxLayout()

        # Set up the input GUI items
        grid_layout = QGridLayout()
        grid_layout.setContentsMargins(0, 0, 0, 0)
        i = 0

        if self.title is not None:
            grid_layout.addWidget(QLabel(self.title), i, 0)
            i += 1

        for key in self._input_values.keys():
            grid_layout.addWidget(QLabel(key), i, 0)
            grid_layout.addWidget(self._input_values[key], i, 1)
            i += 1

        layout.addLayout(grid_layout)
        layout.addStretch()
        self.setLayout(layout)

    def closeEvent(self, event):
        # TODO: this isn't being called when viewer is closed or when plugin is closed
        # assert (False)
        self._delete_viewers()
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

    def _new_plots_viewer(self, title="Plot view", figsize=(8, 3), close_last=True):
        if close_last and len(self.plots_viewers) >= 1:
            viewer = self.plots_viewers.pop()
            viewer.close()
            viewer.deleteLater()

        i = len(self.plots_viewers)
        self.plots_viewers.append(GEMspaPlottingWindow(self.viewer, figsize=figsize))
        self.plots_viewers[i].setWindowTitle(title)
        return self.plots_viewers[i]

    def _new_properties_viewer(self, title="Table view", close_last=True):
        if close_last and len(self.properties_viewers) >= 1:
            viewer = self.properties_viewers.pop()
            viewer.close()
            viewer.deleteLater()

        i = len(self.properties_viewers)
        self.properties_viewers.append(GEMspaTableWindow(self.viewer))
        self.properties_viewers[i].setWindowTitle(title)
        return self.properties_viewers[i]

    def start_task(self, layer_names, log_widget):
        # Perform startup tasks and start thread: worker must be initialized before this function is called

        # Thread for this worker
        self.thread = QThread()
        self.worker.moveToThread(self.thread)

        # connect worker log signal to the GEMspaLogWidget all_log method
        self.worker.log.connect(log_widget.add_log)

        # when thread is started, worker is run with the current state of the widget as input
        self.thread.started.connect(lambda: self.worker.run(self.state(layer_names)))

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
                    show_error(f"{key} input must be a number")
                    valid = False
        return valid

    def _add_napari_layer(self, layer_type, df, **kwargs):
        if layer_type == "points":
            if "z" in df.columns:
                data_cols = ["frame", "z", "y", "x"]
            else:
                data_cols = ["frame", "y", "x"]
        elif layer_type == "tracks":
            if "z" in df.columns:
                data_cols = ["track_id", "frame", "z", "y", "x"]
            else:
                data_cols = ["track_id", "frame", "y", "x"]
        else:
            raise ValueError(
                f"Unrecognized layer type: {layer_type}.  Expected points or tracks."
            )

        data = df[data_cols].to_numpy()
        props = {}
        for col in df.columns:
            if col not in data_cols:
                # nan_to_num because napari properties does not handle nan data
                props[col] = np.nan_to_num(df[col].to_numpy())

        add_layer = getattr(self.viewer, f"add_{layer_type}")
        return add_layer(data, properties=props, **kwargs)


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
