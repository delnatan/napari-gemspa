from ._gemspa_widget import GEMspaWidget, GEMspaWorker
import trackpy as tp
from qtpy.QtWidgets import (QGridLayout, QLabel, QLineEdit, QCheckBox)
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

        diameter = state_params['diameter']
        min_mass = state_params['min_mass']
        invert = state_params['invert']

        image = input_params['image_layer_data']
        scale = input_params['image_layer_scale']

        tp.quiet()  # trackpy to quiet mode
        if state_params['current_frame']:
            # Only process the current frame
            t = input_params['frame']
            f = tp.locate(raw_image=image[t],
                          diameter=diameter,
                          minmass=min_mass,
                          invert=invert)
            self.log.emit(f"Processed frame {t}, number of particles: {len(f)}")
        else:
            # process the entire movie - all frames
            f = tp.batch(frames=image,
                         diameter=diameter,
                         minmass=min_mass,
                         invert=invert)
            self.log.emit(f"Processed {len(image)} frames, number of particles: {len(f)}")

        # TODO: can trackpy handle 3d data?
        # TODO: fix to check that image has a time dimension?
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

        self._current_frame_check = None
        self._diameter_value = None
        self._min_mass_value = None
        self._invert_check = None

        self.init_ui()

    def init_ui(self):

        # Set up the input GUI items
        self._current_frame_check = QCheckBox('Process only current frame')
        self._diameter_value = QLineEdit('11')
        self._min_mass_value = QLineEdit('0')
        self._invert_check = QCheckBox('Invert')

        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self._current_frame_check, 0, 0, 1, 2)
        layout.addWidget(QLabel('Diameter'), 1, 0)
        layout.addWidget(self._diameter_value, 1, 1)
        layout.addWidget(QLabel('Min mass'), 2, 0)
        layout.addWidget(self._min_mass_value, 2, 1)
        layout.addWidget(self._invert_check, 3, 0, 1, 2)

        self.setLayout(layout)

    def start_task(self, layer_name, log_widget):
        # initialize worker and start task
        self.worker = GEMspaLocateWorker()
        super().start_task(layer_name, log_widget)

    def check_inputs(self):

        # Diameter
        try:
            _ = float(self._diameter_value.text())
        except ValueError as err:
            self.show_error(f"Diameter input must be a number")
            return False

        # Min mass
        try:
            _ = float(self._min_mass_value.text())
        except ValueError as err:
            self.show_error(f"Min mass input must be a number")
            return False

        return True

    def state(self, layer_name) -> dict:
        return {'name': self.name,
                'inputs': {'image_layer_name': layer_name,
                           'image_layer_data': self.viewer.layers[layer_name].data,
                           'image_layer_scale': self.viewer.layers[layer_name].scale,
                           'frame': self.viewer.dims.current_step[0]
                           },
                'parameters': {'diameter': float(self._diameter_value.text()),
                               'min_mass': float(self._min_mass_value.text()),
                               'invert': self._invert_check.isChecked(),
                               'current_frame': self._current_frame_check.isChecked()
                               },
                }

    def update_data(self, out_dict):
        """Set the worker outputs to napari layer"""

        if 'df' in out_dict:
            df = out_dict['df']
            data = df[['frame', 'y', 'x']].to_numpy()

            props = {}
            for col in df.columns:
                if col not in ['frame', 'z', 'y', 'x']:
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
