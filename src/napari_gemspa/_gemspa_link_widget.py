from ._gemspa_widget import GEMspaWidget, GEMspaWorker
import pandas as pd
import trackpy as tp
from qtpy.QtWidgets import (QWidget, QGridLayout, QLabel, QLineEdit, QCheckBox, QComboBox, QVBoxLayout)
from qtpy.QtCore import Slot
from qtpy import QtCore


"""Defines: GEMspaLinkWidget, GEMspaLinkWorker"""


class GEMspaLinkWorker(GEMspaWorker):
    """Worker for the Link Features plugin

    """

    def __init__(self):
        super().__init__()

    @Slot(dict)
    def run(self, state):
        """Execute the processing"""

        input_params = state['inputs']
        state_params = state['parameters']

        search_range = state_params['search_range']

        memory = state_params['memory']
        if memory is None:
            memory = 0

        min_frames = state_params['min_frames']

        points_layer_data = input_params['points_layer_data']
        points_layer_props = input_params['points_layer_props']

        out_data = dict()
        if len(points_layer_data.shape) > 1 and points_layer_data.shape[1] >= 3:

            # Make trackpy table from layer data
            f = pd.DataFrame()
            f['y'] = points_layer_data[:, 1]
            f['x'] = points_layer_data[:, 2]
            for col in points_layer_props.keys():
                f[col] = points_layer_props[col]
            f['frame'] = points_layer_data[:, 0]

            tp.quiet()  # trackpy to quiet mode

            # perform linking
            t = tp.link(f, search_range=search_range, memory=memory)
            self.log.emit(f"Number of particles: {t['particle'].nunique()}")

            # Filter spurious trajectories
            if min_frames is not None and min_frames > 1:
                t = tp.filter_stubs(t, threshold=min_frames)
                self.log.emit(f"After filter for Min frames, number of particles: {t['particle'].nunique()}")

            # emit the output data after sorting by track_id (particle) and frame (needed for tracks layer)
            t.index.name = 'index'  # pandas complains when index name and column name are the same
            t = t.sort_values(by=['particle', 'frame'], axis=0, ascending=True)
            if 'z' not in t.columns:
                t['z'] = 0

            out_data = {'df': t}
        else:
            self.log.emit(f"The data does not have a dimension for linking.")

        self.update_data.emit(out_data)
        super().run()


class GEMspaLinkWidget(GEMspaWidget):
    """Widget for Locate Features plugin"""

    name = "GEMspaLinkWidget"

    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)

        self._input_values = {'Link range': QLineEdit('5'),
                              'Memory': QLineEdit('0'),
                              'Min frames': QLineEdit('3')
                              }

        # Link range does not have a default value in trackpy and must be input by the user
        self._required_inputs = ['Link range', ]

        self.init_ui()

    def init_ui(self):

        layout = QVBoxLayout()

        # Set up the input GUI items
        grid_layout = QGridLayout()
        grid_layout.setContentsMargins(0, 0, 0, 0)

        i = 0
        for key in self._input_values.keys():
            grid_layout.addWidget(QLabel(key), i, 0)
            grid_layout.addWidget(self._input_values[key], i, 1)
            i += 1

        layout.addLayout(grid_layout)
        layout.addStretch()
        self.setLayout(layout)

    def start_task(self, layer_name, log_widget):
        # initialize worker and start task
        self.worker = GEMspaLinkWorker()
        super().start_task(layer_name, log_widget)

    def state(self, layer_name) -> dict:
        return {'name': self.name,
                'inputs': {'points_layer_name': layer_name,
                           'points_layer_data': self.viewer.layers[layer_name].data,
                           'points_layer_props': self.viewer.layers[layer_name].properties
                           },
                'parameters': {'search_range': self._convert_to_float(self._input_values['Link range'].text()),
                               'memory': self._convert_to_int(self._input_values['Memory'].text()),
                               'min_frames': self._convert_to_int(self._input_values['Min frames'].text()),
                               },
                }

    def update_data(self, out_dict):
        """Set the worker outputs to napari layers"""

        if 'df' in out_dict:
            df = out_dict['df']
            data = df[['particle', 'frame', 'z', 'y', 'x']].to_numpy()

            props = {}
            for col in df.columns:
                if col not in ['particle', 'frame', 'z', 'y', 'x']:
                    props[col] = df[col].to_numpy()

            layer = self.viewer.add_tracks(data, properties=props, name='Linked features')

            plots_viewer = self._new_plots_viewer(layer.name)
            properties_viewer = self._new_properties_viewer(layer.name)

            plots_viewer.plot_link_results(df)
            plots_viewer.show()

            df.insert(0, 'z', df.pop('z'))
            df.insert(0, 't', df.pop('frame'))
            df.insert(0, 'track_id', df.pop('particle'))
            properties_viewer.reload_from_pandas(df)
            properties_viewer.show()
