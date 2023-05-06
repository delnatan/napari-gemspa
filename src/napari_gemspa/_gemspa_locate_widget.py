from ._gemspa_widget import GEMspaWidget, GEMspaWorker
from ._gemspa_data_views import GEMspaTableWindow, GEMspaPlottingWindow
import napari
import trackpy as tp
from qtpy.QtWidgets import (QGridLayout, QLabel, QLineEdit, QCheckBox, QComboBox)
from qtpy.QtCore import Slot, QObject


"""Defines: GEMspaLocateWidget, GEMspaLocateWorker"""


class GEMspaLocateWorker(GEMspaWorker):
    """Worker for the Locate Features plugin

    """

    def __init__(self):
        super().__init__()

    @Slot(dict)
    def run(self, state):
        """Execute the processing"""

        #input_image_layer = state['inputs']['image']
        state_params = state['parameters']

        diameter = state_params['diameter']
        min_mass = state_params['min_mass']
        invert = state_params['invert']

        # TODO
        image = state['image_data']
        #image = self.viewer.layers[input_image_layer].data
        scale = state['image_scale']
        #scale = self.viewer.layers[input_image_layer].scale

        tp.quiet()  # trackpy to quiet mode
        if state_params['current_frame']:
            # Only process the current frame

            # TODO
            #cur_frame = self.viewer.layers[input_image_layer]._data_view
            #t = self.viewer.dims.current_step[0]
            t = state['frame']

            f = tp.locate(raw_image=image[t],
                          diameter=diameter,
                          minmass=min_mass,
                          invert=invert)
        else:
            # process the entire movie - all frames
            f = tp.batch(frames=image,
                         diameter=diameter,
                         minmass=min_mass,
                         invert=invert)

        # TODO: can trackpy handle 3d data?
        # TODO: fix for when the image does not have a time dimension!?
        if 'frame' not in f.columns:
            f['frame'] = t

        data = f[['frame', 'y', 'x']].to_numpy()

        props = {}
        for col in f.columns:
            if col not in ['frame', 'z', 'y', 'x']:
                props[col] = f[col].to_numpy()

        out_data = {'data': data,
                    'kwargs': {'properties': props,
                               'scale': scale,
                               'size': diameter,
                               'name': 'Feature Locations',
                               'face_color': 'transparent',
                               'edge_color': 'red'}}

        self.update_data.emit(out_data)
        super().run()


class GEMspaLocateWidget(GEMspaWidget):
    """Widget for Locate Features plugin"""

    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)

        # Update the widget when the layers are changed
        self.viewer.layers.events.connect(self._on_layer_change)

        # worker to perform tasks for this widget
        self.worker = GEMspaLocateWorker()

        # viewer for feature properties
        self.properties_viewer = GEMspaTableWindow(napari_viewer)
        self.properties_viewer.setVisible(False)

        # viewer for the graphical output
        self.plots_viewer = GEMspaPlottingWindow(napari_viewer)
        self.plots_viewer.setVisible(False)

        self._input_layer_box = None
        self._current_frame_check = None
        self._diameter_value = None
        self._min_mass_value = None
        self._invert_check = None

    def init_ui(self):

        # Set up the input GUI items
        self._input_layer_box = QComboBox()
        self._current_frame_check = QCheckBox('Process only current frame')
        self._diameter_value = QLineEdit('11')
        self._min_mass_value = QLineEdit('0')
        self._invert_check = QCheckBox('Invert')

        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel('Image layer'), 0, 0)
        layout.addWidget(self._input_layer_box, 0, 1)
        layout.addWidget(self._current_frame_check, 1, 0, 1, 2)
        layout.addWidget(QLabel('Diameter'), 2, 0)
        layout.addWidget(self._diameter_value, 2, 1)
        layout.addWidget(QLabel('Min mass'), 3, 0)
        layout.addWidget(self._min_mass_value, 3, 1)
        layout.addWidget(self._invert_check, 4, 0, 1, 2)

        self.setLayout(layout)

        # Initialize the layers list
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.image.image.Image):
                self._input_layer_box.addItem(layer.name)

    def _on_layer_change(self, e):
        """Callback called when a napari layer is updated so the layer list can be updated also

        Parameters
        ----------
        e: QObject
            Qt event

        """

        current_text = self._input_layer_box.currentText()
        self._input_layer_box.clear()

        is_current_item_still_here = False
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.image.image.Image):
                if layer.name == current_text:
                    is_current_item_still_here = True
                self._input_layer_box.addItem(layer.name)
        if is_current_item_still_here:
            self._input_layer_box.setCurrentText(current_text)

    def check_inputs(self):

        if self._input_layer_box.count() < 1:
            self.show_error(f"No image data")
            return False

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

    # TODO: update with additional params needed by Worker since it cannot access napari viwer object
    def state(self) -> dict:
        return {'name': 'GEMspaLocateWidget',
                'inputs': {'image': self._input_layer_box.currentText()},
                'parameters': {'diameter': float(self._diameter_value.text()),
                               'min_mass': float(self._min_mass_value.text()),
                               'invert': self._invert_check.isChecked(),
                               'current_frame': self._current_frame_check.isChecked()
                               },
                'outputs': ['points', 'trackpy detections']
                }

    def show_properties(self, layer_name):
        """reload and display the particles properties in a popup window"""
        self.properties_viewer.layer_name = layer_name
        self.properties_viewer.reload()
        self.properties_viewer.show()

    def show_plots(self, layer_name):
        """reload and display the particles properties in a popup window"""
        self.plots_viewer.layer_name = layer_name
        self.plots_viewer.reload()
        self.plots_viewer.show()

    def update_data(self, out_dict):
        """Set the plugin outputs to napari layers"""

        layer = self.viewer.add_points(out_dict['data'], **out_dict['kwargs'])

        # Show properties from the new layer in a table (pop-up)
        self.show_properties(layer.name)

        # Show plots as a pop-up widget
        self.show_plots(layer.name)
