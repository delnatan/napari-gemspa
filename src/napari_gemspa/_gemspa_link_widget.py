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
        scale = input_params['points_layer_scale']
        points_layer_props = input_params['points_layer_props']

        out_data = dict()
        if len(points_layer_data.shape) > 1 and points_layer_data.shape[1] >= 3:

            # Make trackpy table from layer data
            f = self._make_trackpy_table("points", points_layer_data, points_layer_props)
            if f is not None:
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

                # change column name from 'particle' to 'track_id' to identify the track for consistency with napari
                t.rename(columns={'particle': 'track_id'}, inplace=True)

                out_data = {'df': t,
                            'scale': scale}
            else:
                self.log.emit(f"Error: The points layer properties do not contain the required columns for linking.")
        else:
            self.log.emit(f"The data does not have a dimension for linking.")

        self.update_data.emit(out_data)
        super().run()


class GEMspaLinkWidget(GEMspaWidget):
    """Widget for Link features plugin"""

    name = "GEMspaLinkWidget"

    def __init__(self, napari_viewer, title="Link features with trackpy:"):
        super().__init__(napari_viewer, title)

        self._input_values = {'Link range': QLineEdit('5'),
                              'Memory': QLineEdit('0'),
                              'Min frames': QLineEdit('3')
                              }

        # Link range does not have a default value in trackpy and must be input by the user
        self._required_inputs = ['Link range', ]

        self.init_ui()

    def start_task(self, layer_names, log_widget):
        # initialize worker and start task
        self.worker = GEMspaLinkWorker()
        super().start_task(layer_names, log_widget)

    def state(self, layer_names) -> dict:

        inputs_dict = {'points_layer_name': layer_names['points'],
                       'points_layer_data': self.viewer.layers[layer_names['points']].data,
                       'points_layer_scale': self.viewer.layers[layer_names['points']].scale,
                       'points_layer_props': self.viewer.layers[layer_names['points']].properties
                       }
        if 'labels' in layer_names:
            inputs_dict['labels_layer_name'] = layer_names['labels']
            inputs_dict['labels_layer_data'] = self.viewer.layers[layer_names['labels']].data

        return {'name': self.name,
                'inputs': inputs_dict,
                'parameters': {'search_range': self._convert_to_float(self._input_values['Link range'].text()),
                               'memory': self._convert_to_int(self._input_values['Memory'].text()),
                               'min_frames': self._convert_to_int(self._input_values['Min frames'].text()),
                               },
                }

    def update_data(self, out_dict):
        """Set the worker outputs to napari layers"""

        if 'df' in out_dict:
            df = out_dict['df']
            kwargs = {'scale': out_dict['scale'],
                      'blending': 'translucent',
                      'tail_length': df['frame'].max(),
                      'name': 'Linked features'}

            if len(df) == 0:
                self.show_error("No particles were linked!")
            else:
                layer = self._add_napari_layer("tracks", df, **kwargs)

                plots_viewer = self._new_plots_viewer(layer.name)
                properties_viewer = self._new_properties_viewer(layer.name)

                plots_viewer.plot_link_results(df)
                plots_viewer.show()

                # Fixing column ordering for display on table view
                if self.display_table_view:
                    df.insert(0, 'frame', df.pop('frame'))
                    df.insert(0, 'track_id', df.pop('track_id'))
                    properties_viewer.reload_from_pandas(df)
                    properties_viewer.show()
