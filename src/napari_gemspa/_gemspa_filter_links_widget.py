from ._gemspa_widget import GEMspaWidget, GEMspaWorker
import pandas as pd
import trackpy as tp
from qtpy.QtWidgets import (QWidget, QGridLayout, QLabel, QLineEdit, QCheckBox, QComboBox, QVBoxLayout)
from qtpy.QtCore import Slot
from qtpy import QtCore


"""Defines: GEMspaFilterLinksWidget, GEMspaFilterLinksWorker"""


class GEMspaFilterLinksWorker(GEMspaWorker):
    """Worker for the Link Features plugin

    """

    def __init__(self):
        super().__init__()

    @Slot(dict)
    def run(self, state):
        """Execute the processing"""

        input_params = state['inputs']
        state_params = state['parameters']

        min_frames = state_params['min_frames']
        min_mass = state_params['min_mass']
        max_mass = state_params['max_mass']
        min_size = state_params['min_size']
        max_size = state_params['max_size']
        min_ecc = state_params['min_ecc']
        max_ecc = state_params['max_ecc']

        tracks_layer_data = input_params['tracks_layer_data']
        tracks_layer_props = input_params['tracks_layer_props']

        # Make trackpy table from layer data
        t = self._make_trackpy_table("tracks", tracks_layer_data, tracks_layer_props)
        self.log.emit(f"Number of particles: {t['particle'].nunique()}")

        tp.quiet()

        # filter by min frames
        if min_frames is not None and min_frames > 1:
            t = tp.filter_stubs(t, threshold=min_frames)
            self.log.emit(f"After filter for Min frames, number of particles: {t['particle'].nunique()}")

        # filter by mass, size, eccentricity
        mean_t = t.groupby('particle').mean()

        if min_mass is not None and min_mass > 0:
            mean_t = mean_t[mean_t['mass'] >= min_mass]
        if max_mass is not None:
            mean_t = mean_t[mean_t['mass'] <= max_mass]

        if min_size is not None and min_size > 0:
            mean_t = mean_t[mean_t['size'] >= min_size]
        if max_size is not None:
            mean_t = mean_t[mean_t['size'] <= max_size]

        if min_ecc is not None and min_ecc > 0:
            mean_t = mean_t[mean_t['ecc'] >= min_ecc]
        if max_ecc is not None and max_ecc < 1:
            mean_t = mean_t[mean_t['ecc'] <= max_ecc]

        t = t[t['particle'].isin(mean_t.index)]
        self.log.emit(f"After filter for mass/size/eccentricity, number of particles: {t['particle'].nunique()}")

        # emit the output data after sorting by track_id (particle) and frame (needed for tracks layer)
        t.index.name = 'index'  # pandas complains when index name and column name are the same
        t = t.sort_values(by=['particle', 'frame'], axis=0, ascending=True)
        if 'z' not in t.columns:
            t['z'] = 0

        out_data = {'df': t}

        self.update_data.emit(out_data)
        super().run()


class GEMspaFilterLinksWidget(GEMspaWidget):
    """Widget for Filter links plugin"""

    name = "GEMspaFilterLinksWidget"

    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)

        self._input_values = {'Min frames': QLineEdit('3'),
                              'Min mass': QLineEdit('0'),
                              'Max mass': QLineEdit(''),
                              'Min size': QLineEdit('0'),
                              'Max size': QLineEdit(''),
                              'Min eccentricity': QLineEdit('0'),
                              'Max eccentricity': QLineEdit('1'),
                              }

        self._required_inputs = []

        self.init_ui()

    def start_task(self, layer_name, log_widget):
        # initialize worker and start task
        self.worker = GEMspaFilterLinksWorker()
        super().start_task(layer_name, log_widget)

    def state(self, layer_name) -> dict:
        return {'name': self.name,
                'inputs': {'tracks_layer_name': layer_name,
                           'tracks_layer_data': self.viewer.layers[layer_name].data,
                           'tracks_layer_props': self.viewer.layers[layer_name].properties
                           },
                'parameters': {'min_frames': self._convert_to_int(self._input_values['Min frames'].text()),
                               'min_mass': self._convert_to_float(self._input_values['Min mass'].text()),
                               'max_mass': self._convert_to_float(self._input_values['Max mass'].text()),
                               'min_size': self._convert_to_float(self._input_values['Min size'].text()),
                               'max_size': self._convert_to_float(self._input_values['Max size'].text()),
                               'min_ecc': self._convert_to_float(self._input_values['Min eccentricity'].text()),
                               'max_ecc': self._convert_to_float(self._input_values['Max eccentricity'].text()),
                               },
                }

    def update_data(self, out_dict):
        """Set the worker outputs to napari layers"""

        if 'df' in out_dict:
            kwargs = {'name': 'Linked features (filtered)'}
            df = out_dict['df']
            layer = self._add_napari_layer("tracks", df, **kwargs)

            # TODO
            plots_viewer = self._new_plots_viewer(layer.name)
            properties_viewer = self._new_properties_viewer(layer.name)

            plots_viewer.plot_link_results(df)
            plots_viewer.show()

            df.insert(0, 'z', df.pop('z'))
            df.insert(0, 't', df.pop('frame'))
            df.insert(0, 'track_id', df.pop('particle'))
            properties_viewer.reload_from_pandas(df)
            properties_viewer.show()
