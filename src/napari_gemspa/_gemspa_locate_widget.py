from ._gemspa_widget import GEMspaWidget, GEMspaWorker
from ._gemspa_data_views import GEMspaTableWindow, GEMspaPlottingWindow
import napari
import trackpy as tp
from qtpy.QtWidgets import (QGridLayout, QLabel, QLineEdit, QCheckBox, QComboBox)


"""Defines: GEMspaLocateWidget, GEMspaLocateWorker"""


class GEMspaLocateWidget(GEMspaWidget):
    """Widget for Locate Features plugin"""

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        napari_viewer.layers.events.connect(self._on_layer_change)

        # viewer for feature properties
        self.properties_viewer = GEMspaTableWindow(napari_viewer)
        self.properties_viewer.setVisible(False)

        # viewer for the graphical output
        self.plots_viewer = GEMspaPlottingWindow(napari_viewer)
        self.plots_viewer.setVisible(False)

        # Set up the input GUI items
        self._input_layer_box = QComboBox()
        self._current_frame_check = QCheckBox('Process only current frame')

        self._diameter_label = QLabel('Diameter')
        self._diameter_value = QLineEdit('11')

        self._min_mass_label = QLabel('Min mass')
        self._min_mass_value = QLineEdit('0')

        self._invert_check = QCheckBox('Invert')

        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel('Image layer'), 0, 0)
        layout.addWidget(self._input_layer_box, 0, 1)

        layout.addWidget(self._current_frame_check, 1, 0, 1, 2)

        layout.addWidget(self._diameter_label, 2, 0)
        layout.addWidget(self._diameter_value, 2, 1)

        layout.addWidget(self._min_mass_label, 3, 0)
        layout.addWidget(self._min_mass_value, 3, 1)

        layout.addWidget(self._invert_check, 4, 0, 1, 2)

        self.setLayout(layout)

    def init_layer_list(self):
        """Initialize the layers lists"""
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.image.image.Image):
                self._input_layer_box.addItem(layer.name)

        # emit a signal indicating whether there are any valid image layers
        if self._input_layer_box.count() < 1:
            self.enable.emit(False)
        else:
            self.enable.emit(True)

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
        if self._input_layer_box.count() < 1:
            self.enable.emit(False)
        else:
            self.enable.emit(True)

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


class GEMspaLocateWorker(GEMspaWorker):
    """Worker for the Locate Features plugin"""

    def __init__(self, napari_viewer, widget):
        super().__init__(napari_viewer, widget)

        self._out_data = None

    def run(self):
        """Execute the processing"""

        state = self.widget.state()
        input_image_layer = state['inputs']['image']
        state_params = state['parameters']

        diameter = state_params['diameter']
        min_mass = state_params['min_mass']
        invert = state_params['invert']

        image = self.viewer.layers[input_image_layer].data
        scale = self.viewer.layers[input_image_layer].scale

        tp.quiet()  # trackpy to quiet mode
        if state_params['current_frame']:
            # Only process the current frame
            #cur_frame = self.viewer.layers[input_image_layer]._data_view
            t = self.viewer.dims.current_step[0]
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

        data = f[['frame', 'y', 'x']].to_numpy() #.astype('int').to_numpy()

        props = {}
        for col in f.columns:
            if col not in ['frame', 'z', 'y', 'x']:
                props[col] = f[col].to_numpy()

        self._out_data = {'data': data,
                          'kwargs': {'properties': props,
                                     'scale': scale,
                                     'size': diameter,
                                     'name': 'Feature Locations',
                                     'face_color': 'transparent',
                                     'edge_color': 'red'}}
        self.finished.emit()

    def set_outputs(self):
        """Set the plugin outputs to napari layers"""
        layer = self.viewer.add_points(self._out_data['data'], **self._out_data['kwargs'])

        # Show properties from the new layer in a table (pop-up)
        self.widget.show_properties(layer.name)

        # Show plots as a pop-up widget
        self.widget.show_plots(layer.name)


class GEMspaLocatePlugin(GEMspaPlugin):

    """Napari plugin for locating features with trackpy

    Parameters
    ----------
    napari_viewer: Viewer
        Napari viewer

    """
    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)
        self.title = 'Locate features with trackpy'
        self.widget = GEMspaLocateWidget(napari_viewer)
        self.worker = GEMspaLocateWorker(napari_viewer, self.widget)

        # connect the enable signal from the widget to the set_enable function (enables/disables run button)
        # enable signal is sent based on whether there are valid layers available (image layers)
        self.widget.enable.connect(self.set_enable)

        self.init_ui()
        self.widget.init_layer_list()