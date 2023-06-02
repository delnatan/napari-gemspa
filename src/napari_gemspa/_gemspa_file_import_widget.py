"""Defines: GEMspaFileImport, GEMspaFileImportWidget"""

import os

import napari
import nd2
import numpy as np
import pandas as pd
from gemspa_spt import ParticleTracks
from qtpy.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)
from skimage import io

from ._utils import show_error


class GEMspaFileImport:
    def __init__(self, path, data_format):
        if data_format not in ParticleTracks.file_columns.keys():
            raise ValueError(
                f"Unexpected file format: {data_format} when importing file."
            )

        self.file_path = path
        self.file_format = data_format
        file_ext = os.path.splitext(self.file_path)[1].lower()
        if file_ext == ".csv":
            self.file_sep = ","
        elif file_ext == ".txt" or file_ext == ".tsv":
            self.file_sep = "\t"
        else:
            print(
                f"Unknown extension found for '{self.file_path}'. Attempting to open as a tab-delimited text file."
            )
            self.file_sep = "\t"

        self.file_df = pd.read_csv(
            self.file_path,
            sep=self.file_sep,
            header=0,
            skiprows=ParticleTracks.skip_rows[data_format],
        )

    def find_col(self, target_string):
        lower_target = target_string.lower()
        for col in self.file_df.columns:
            if col.lower() == lower_target:
                return col
        return None

    def get_layer_data(self):
        all_cols = ParticleTracks.file_columns[self.file_format]
        orig_cols = []

        # 'x' is mandatory
        col = self.find_col(all_cols[4])
        if col is None:
            raise Exception(
                f"Error in importing file: required column {all_cols[4]} is missing."
            )
        orig_cols.insert(0, col)

        # 'y' is mandatory
        col = self.find_col(all_cols[3])
        if col is None:
            raise Exception(
                f"Error in importing file: required column {all_cols[3]} is missing."
            )
        orig_cols.insert(0, col)

        # 'z' is optional
        col = self.find_col(all_cols[2])
        if col:
            orig_cols.insert(0, col)

        # 'frame' is optional
        frame_col = self.find_col(all_cols[1])
        if frame_col:
            orig_cols.insert(0, frame_col)

        # 'track_id' is optional; if it exists this is tracks data (not points)
        track_id_col = self.find_col(all_cols[0])
        if track_id_col:
            if frame_col is None:
                raise Exception(
                    "Error in importing file data: data appears to be tracks layer but frame column is missing."
                )
            orig_cols.insert(0, track_id_col)
            layer_type = "tracks"
        else:
            layer_type = "points"

        # data and properties for the layer data tuple
        data = self.file_df[orig_cols].to_numpy()
        props = {}
        for col in self.file_df.columns:
            if col not in orig_cols:
                # nan_to_num because napari properties does not handle nan data
                props[col] = np.nan_to_num(self.file_df[col].to_numpy())

        # add properties and other keyword args
        add_kwargs = {"properties": props, "name": os.path.split(self.file_path)[1]}
        if layer_type == "points":
            add_kwargs["face_color"] = "transparent"
            add_kwargs["edge_color"] = "red"
        elif layer_type == "tracks":
            add_kwargs["blending"] = "translucent"
            add_kwargs["tail_length"] = int(data[:, 1].max())

        return data, add_kwargs, layer_type


class GEMspaFileImportWidget(QWidget):
    """Widget for Import file plugin"""

    name = "GEMspaFileImportWidget"

    def __init__(self, napari_viewer):
        super().__init__()

        self.viewer = napari_viewer

        self._track_file_format_rbs = [
            QRadioButton("GEMspa", self),
            QRadioButton("Mosaic", self),
            QRadioButton("Trackmate", self),
            QRadioButton("Trackpy", self),
        ]

        self._open_tracks_file_btn = QPushButton("Open file...", self)
        self._open_image_file_btn = QPushButton("Open file...", self)
        self._open_labels_file_btn = QPushButton("Open file...", self)
        self._new_labels_layer_btn = QPushButton("Add layer", self)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        group = QGroupBox("Add image layer")
        group_layout = QVBoxLayout()
        group_layout.addWidget(QLabel("Open image file (nd2/tif time-lapse movie):"))
        group_layout.addWidget(self._open_image_file_btn)
        group.setLayout(group_layout)
        layout.addWidget(group)

        group = QGroupBox("Add labels layer")
        group_layout = QGridLayout()
        group_layout.addWidget(
            QLabel("Create blank 2D labels layer for selected image:"), 0, 0
        )
        group_layout.addWidget(self._new_labels_layer_btn, 0, 1)
        group_layout.addWidget(QLabel("Open labels file:"), 1, 0)
        group_layout.addWidget(self._open_labels_file_btn, 1, 1)
        group.setLayout(group_layout)
        layout.addWidget(group)

        group = QGroupBox("Add tracks layer")
        group_layout = QVBoxLayout()
        for rb in self._track_file_format_rbs:
            group_layout.addWidget(rb)
        group_layout.addWidget(self._open_tracks_file_btn)
        group.setLayout(group_layout)
        layout.addWidget(group)

        layout.addStretch()

        self._track_file_format_rbs[0].setChecked(True)
        self._open_tracks_file_btn.clicked.connect(self._load_tracks_file)
        self._open_image_file_btn.clicked.connect(self._load_image_file)
        self._open_labels_file_btn.clicked.connect(self._load_labels_file)
        self._new_labels_layer_btn.clicked.connect(self._new_labels_layer)

        self.setLayout(layout)

    def _new_labels_layer(self):
        # create blank image same x/y dimensions as selected image layer
        # if image layer has other dimensions, ignore
        # add as a labels layer
        selected_layers = self.viewer.layers.selection

        found_image = False
        if len(selected_layers) > 0:
            for layer in selected_layers:
                if isinstance(layer, napari.layers.image.image.Image):
                    # get final 2 image dimensions and create mask
                    dims = layer.data.shape[-2:]
                    new_mask = np.zeros(dims, dtype="int64")
                    self.viewer.add_labels(new_mask)
                    found_image = True
                    break

        if not found_image:
            show_error("Cannot create labels layer: no selected image layer found.")

    def _load_image_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "time-lapse movies (*.tif *.tiff *.nd2)"
        )
        if filename:
            ext = os.path.splitext(filename)[1]
            if ext == ".nd2":
                f = nd2.ND2File(filename)
                images = f.asarray()
                f.close()
            elif ext == ".tif" or ext == ".tiff":
                images = io.imread(filename)
            else:
                raise ValueError(f"Unrecognized file extension for image file {ext}")

            self.viewer.add_image(images, name=os.path.split(filename)[1])

    def _load_labels_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "images with integer labels (*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff *.jp2)",
        )
        if filename:
            labeled_image = io.imread(filename)
            self.viewer.add_labels(labeled_image, name=os.path.split(filename)[1])

    def _load_tracks_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "tab or comma delimited text files (*.txt *.tsv *.csv)"
        )
        if filename:
            # Get selected format for imported file
            data_format = self._track_file_format_rbs[0].text()
            for rb in self._track_file_format_rbs:
                if rb.isChecked():
                    data_format = rb.text()
                    break
            data_format = data_format.lower()

            file_import = GEMspaFileImport(filename, data_format)
            layer_data = file_import.get_layer_data()

            # Add layer (points or tracks)
            if layer_data[2] == "points":
                self.viewer.add_points(layer_data[0], **layer_data[1])
            elif layer_data[2] == "tracks":
                self.viewer.add_tracks(layer_data[0], **layer_data[1])
            else:
                pass
