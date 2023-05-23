from qtpy.QtWidgets import (QGridLayout, QVBoxLayout, QFileDialog, QRadioButton, QPushButton, QWidget, QLabel)
import os
import pandas as pd

"""Defines: GEMspaFileImport, GEMspaFileImportWidget"""


class GEMspaFileImport:
    file_columns = {'mosaic': ['trajectory', 'frame', 'z', 'y', 'x'],
                    'trackmate': ['track_id', 'frame', 'position_z', 'position_y', 'position_x'],
                    'trackpy': ['particle', 'frame', 'z', 'y', 'x'],
                    'gemspa': ['track_id', 'frame', 'z', 'y', 'x']}
    skip_rows = {'mosaic': None, 'trackmate': [1, 2, 3], 'trackpy': None, 'gemspa': None}

    def __init__(self, path, data_format):

        if data_format not in GEMspaFileImport.file_columns.keys():
            raise ValueError(f"Unexpected file format: {data_format} when importing file.")

        self.file_path = path
        self.file_format = data_format
        file_ext = os.path.splitext(self.file_path)[1].lower()
        if file_ext == '.csv':
            self.file_sep = ','
        elif file_ext == '.txt' or file_ext == '.tsv':
            self.file_sep = '\t'
        else:
            print(f"Unknown extension found for file {self.file_path}.  Attempting to open as a tab-delimited text file.")
            self.file_sep = '\t'

        self.file_df = pd.read_csv(self.file_path, sep=self.file_sep, header=0, skiprows=self.skip_rows[data_format])
        self.file_df.columns = [item.lower() for item in self.file_df.columns]

    def get_layer_data(self):

        all_cols = GEMspaFileImport.file_columns[self.file_format]
        cols = all_cols[3:5]  # ['y','x']: mandatory
        for col in cols:
            if col not in self.file_df.columns:
                raise Exception(f"Error in importing file: required column {col} is missing.")

        if all_cols[2] in self.file_df.columns:  # 'z': optional, add if it exists
            cols.insert(0, all_cols[2])

        if all_cols[1] in self.file_df.columns:  # 'frame': optional, add if it exists
            cols.insert(0, all_cols[1])

        if all_cols[0] in self.file_df.columns:  # 'track_id': optional, if it exists this is tracks data (not points)
            if cols[0] != all_cols[1]:  # must have 'frame' column if there is a 'track_id' column
                raise Exception(
                    f"Error in importing file data: data appears to be tracks layer but frame column is missing.")
            cols.insert(0, all_cols[0])
            layer_type = "tracks"
        else:
            layer_type = "points"

        # data and properties for the layer data tuple
        data = self.file_df[cols].to_numpy()
        props = {}
        for col in self.file_df.columns:
            if col not in cols:
                props[col] = self.file_df[col].to_numpy()

        # add properties and other keyword args
        add_kwargs = {'properties': props,
                      'name': os.path.split(self.file_path)[1]}
        if layer_type == 'points':
            add_kwargs['face_color'] = 'transparent'
            add_kwargs['edge_color'] = 'red'
        elif layer_type == 'tracks':
            add_kwargs['blending'] = 'translucent'
            add_kwargs['tail_length'] = data[:, 1].max()

        return data, add_kwargs, layer_type


class GEMspaFileImportWidget(QWidget):
    """Widget for Import file plugin"""

    name = 'GEMspaFileImportWidget'

    def __init__(self, napari_viewer):
        super().__init__()

        self.viewer = napari_viewer

        self._format_rbs = [QRadioButton("GEMspa", self),
                            QRadioButton("Mosaic", self),
                            QRadioButton("Trackmate", self),
                            QRadioButton("Trackpy", self)
                            ]

        self._open_file_btn = QPushButton("Open file...", self)

        self.init_ui()

    def init_ui(self):

        layout = QVBoxLayout()

        grid_layout = QGridLayout()
        grid_layout.setContentsMargins(0, 0, 0, 0)
        i = 0

        grid_layout.addWidget(QLabel("Select file type:"), i, 0)
        i += 1

        for rb in self._format_rbs:
            grid_layout.addWidget(rb, i, 0)
            i += 1

        grid_layout.addWidget(self._open_file_btn, i, 0)
        i += 1

        layout.addLayout(grid_layout)
        layout.addStretch()

        self._open_file_btn.clicked.connect(self._load_file)
        self._format_rbs[0].setChecked(True)

        self.setLayout(layout)

    def _load_file(self):

        fname = QFileDialog.getOpenFileName(self,
                                            "Tab-delimited Text Files (*.txt);;Tab-delimited Text Files (*.tsv);;CSV Files (*.csv);;All Files (*)")
        if fname[0]:

            # Get selected format for imported file
            data_format = self._format_rbs[0].text()
            for rb in self._format_rbs:
                if rb.isChecked():
                    data_format = rb.text()
                    break
            data_format = data_format.lower()

            file_import = GEMspaFileImport(fname[0], data_format)
            layer_data = file_import.get_layer_data()

            # Add layer (points or tracks)
            if layer_data[2] == 'points':
                self.viewer.add_points(layer_data[0], **layer_data[1])
            elif layer_data[2] == 'tracks':
                self.viewer.add_tracks(layer_data[0], **layer_data[1])
            else:
                pass
