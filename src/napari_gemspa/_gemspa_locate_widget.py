from ._gemspa_widget import GEMspaWidget, GEMspaWorker
import trackpy as tp
from qtpy.QtWidgets import (QGridLayout, QLabel, QLineEdit, QCheckBox, QVBoxLayout)
from qtpy.QtCore import Slot
from ._gemspa_data_views import GEMspaTableWindow, GEMspaPlottingWindow


"""Defines: GEMspaLocateWidget, GEMspaLocateWorker"""


class GEMspaLocateWorker(GEMspaWorker):
    """Worker for the Locate Features plugin

    """

    def __init__(self):
        super().__init__()

    @Slot(dict)
    def run(self, state):
        """Execute the processing"""

        input_params = state['inputs']
        state_params = state['parameters']

        current_frame = state_params['current_frame']
        del state_params['current_frame']

        diameter = state_params['diameter']
        del state_params['diameter']

        keys = list(state_params.keys())
        for key in keys:
            if state_params[key] is None:
                del state_params[key]

        image = input_params['image_layer_data']
        scale = input_params['image_layer_scale']

        tp.quiet()  # trackpy to quiet mode
        if current_frame:
            # Only process the current frame
            t = input_params['frame']
            f = tp.locate(image[t], diameter, **state_params)
            self.log.emit(f"Processed frame {t}, number of particles: {len(f)}")
        else:
            # process the entire movie - all frames
            f = tp.batch(image, diameter, **state_params)
            self.log.emit(f"Processed {len(image)} frames, number of particles: {len(f)}")

        if 'frame' not in f.columns:
            f['frame'] = t

        out_data = {'df': f,
                    'scale': scale,
                    'diameter': diameter}

        self.update_data.emit(out_data)
        super().run()


class GEMspaLocateWidget(GEMspaWidget):
    """Widget for Locate Features plugin"""

    name = 'GEMspaLocateWidget'

    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)

        self._current_frame_check = QCheckBox('Process only current frame')
        self._invert_check = QCheckBox('Invert')
        self._preprocess_check = QCheckBox('Preprocess')

        self._input_values = {'Diameter': QLineEdit('11'),
                              'Min mass': QLineEdit('100'),
                              'Max size': QLineEdit(''),
                              'Separation': QLineEdit(''),
                              'Noise size': QLineEdit('1'),
                              'Smoothing size': QLineEdit(''),
                              'Threshold': QLineEdit(''),
                              'Percentile': QLineEdit('64'),
                              'Top n': QLineEdit('')
                              }
        # Diameter does not have a default value in trackpy and must be input by the user
        self._required_inputs = ['Diameter', ]

        self.init_ui()

    def init_ui(self):

        layout = QVBoxLayout()

        grid_layout = QGridLayout()
        grid_layout.setContentsMargins(0, 0, 0, 0)
        i = 0

        grid_layout.addWidget(self._current_frame_check, 0, 0, 1, 2)
        i += 1

        for key in self._input_values.keys():
            grid_layout.addWidget(QLabel(key), i, 0)
            grid_layout.addWidget(self._input_values[key], i, 1)
            i += 1

        grid_layout.addWidget(self._invert_check, i, 0, 1, 2)
        i += 1

        self._preprocess_check.setChecked(True)
        grid_layout.addWidget(self._preprocess_check, i, 0, 1, 2)

        layout.addLayout(grid_layout)
        layout.addStretch()
        self.setLayout(layout)

    def start_task(self, layer_name, log_widget):
        # initialize worker and start task
        self.worker = GEMspaLocateWorker()
        super().start_task(layer_name, log_widget)

    def state(self, layer_name) -> dict:

        return {'name': self.name,
                'inputs': {'image_layer_name': layer_name,
                           'image_layer_data': self.viewer.layers[layer_name].data,
                           'image_layer_scale': self.viewer.layers[layer_name].scale,
                           'frame': self.viewer.dims.current_step[0]
                           },
                'parameters': {'diameter': self._convert_to_float(self._input_values['Diameter'].text()),
                               'minmass': self._convert_to_float(self._input_values['Min mass'].text()),
                               'maxsize': self._convert_to_float(self._input_values['Max size'].text()),
                               'separation': self._convert_to_float(self._input_values['Separation'].text()),
                               'noise_size': self._convert_to_float(self._input_values['Noise size'].text()),
                               'smoothing_size': self._convert_to_float(self._input_values['Smoothing size'].text()),
                               'threshold': self._convert_to_float(self._input_values['Threshold'].text()),
                               'percentile': self._convert_to_float(self._input_values['Percentile'].text()),
                               'topn': self._convert_to_float(self._input_values['Top n'].text()),
                               'invert': self._invert_check.isChecked(),
                               'preprocess': self._preprocess_check.isChecked(),
                               'current_frame': self._current_frame_check.isChecked()
                               },
                }

    def update_data(self, out_dict):
        """Set the worker outputs to napari layer"""

        if 'df' in out_dict:
            df = out_dict['df']

            if len(out_dict['scale']) == 4:
                data_cols = ['frame', 'z', 'y', 'x']
            elif len(out_dict['scale']) == 3:
                data_cols = ['frame', 'y', 'x']
            else:
                raise ValueError("Data for points layer is not the expected number of dimensions (3 or 4).")

            data = df[data_cols].to_numpy()
            props = {}
            for col in df.columns:
                if col not in data_cols:
                    props[col] = df[col].to_numpy()

            layer = self.viewer.add_points(data,
                                           properties=props,
                                           scale=out_dict['scale'],
                                           size=out_dict['diameter'],
                                           name='Feature Locations',
                                           face_color='transparent',
                                           edge_color='red')

            plots_viewer = self._new_plots_viewer(layer.name)
            properties_viewer = self._new_properties_viewer(layer.name)

            # viewer for the graphical output
            plots_viewer.plot_locate_results(df)
            plots_viewer.show()

            # viewer for feature properties
            df.insert(0, 't', df.pop('frame'))
            properties_viewer.reload_from_pandas(df)
            properties_viewer.show()
