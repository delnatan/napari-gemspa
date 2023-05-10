from ._gemspa_widget import GEMspaWidget, GEMspaWorker
import pandas as pd
import trackpy as tp
from qtpy.QtWidgets import (QGridLayout, QLabel, QLineEdit, QCheckBox, QComboBox)
from qtpy.QtCore import Slot


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

        link_range = state_params['link_range']
        memory = state_params['memory']
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
            t = tp.link(f, search_range=link_range, memory=memory)
            self.log.emit(f"Number of particles: {t['particle'].nunique()}")

            # Filter spurious trajectories
            t = tp.filter_stubs(t, threshold=min_frames)
            self.log.emit(f"After filter_stubs, number of particles: {t['particle'].nunique()}")

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

        self._link_range_value = None
        self._memory_value = None
        self._min_frames_value = None

        self.init_ui()

    def init_ui(self):

        # Set up the input GUI items
        self._link_range_value = QLineEdit('5')
        self._memory_value = QLineEdit('0')
        self._min_frames_value = QLineEdit('3')

        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel('Link range'), 0, 0)
        layout.addWidget(self._link_range_value, 0, 1)
        layout.addWidget(QLabel('Memory'), 1, 0)
        layout.addWidget(self._memory_value, 1, 1)
        layout.addWidget(QLabel('Min frames'), 2, 0)
        layout.addWidget(self._min_frames_value, 2, 1)

        self.setLayout(layout)

    def start_task(self, layer_name, log_widget):
        # initialize worker and start task
        self.worker = GEMspaLinkWorker()
        super().start_task(layer_name, log_widget)

    def check_inputs(self):

        # Link range
        try:
            _ = float(self._link_range_value.text())
        except ValueError as err:
            self.show_error(f"Link range input must be a number")
            return False

        # Memory
        try:
            _ = int(self._memory_value.text())
        except ValueError as err:
            self.show_error(f"Memory input must be an integer")
            return False

        # Min frames
        try:
            _ = int(self._min_frames_value.text())
        except ValueError as err:
            self.show_error(f"Min frames input must be an integer")
            return False

        return True

    def state(self, layer_name) -> dict:
        return {'name': self.name,
                'inputs': {'points_layer_name': layer_name,
                           'points_layer_data': self.viewer.layers[layer_name].data,
                           'points_layer_props': self.viewer.layers[layer_name].properties
                           },
                'parameters': {'link_range': float(self._link_range_value.text()),
                               'memory': int(self._memory_value.text()),
                               'min_frames': int(self._min_frames_value.text()),
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
