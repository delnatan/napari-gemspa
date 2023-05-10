from ._gemspa_locate_widget import GEMspaLocateWidget
from ._gemspa_link_widget import GEMspaLinkWidget
from ._gemspa_analyze_widget import GEMspaAnalyzeWidget
from ._gemspa_widget import GEMspaLogWidget
from qtpy.QtWidgets import (QWidget, QLabel, QPushButton, QVBoxLayout, QTabWidget, QComboBox, QGridLayout)
from qtpy import QtCore
import napari


"""Defines: GEMspaPlugin"""


class GEMspaLayerInput(QWidget):

    supported_layer_types = ['image', 'points', 'tracks']

    def __init__(self, napari_viewer, layer_type):
        super().__init__()

        if layer_type not in self.supported_layer_types:
            print("Invalid layer type")
            raise ValueError

        self.viewer = napari_viewer
        self.layer_type = layer_type

        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel(f'{self.layer_type} layer'), 0, 0)
        self._input_layer_box = QComboBox()
        layout.addWidget(self._input_layer_box, 0, 1)

        self.setLayout(layout)

        self.viewer.layers.events.connect(self._on_layer_change)

        for layer in self.viewer.layers:
            if self._is_layer_type_instance(layer):
                self._input_layer_box.addItem(layer.name)

    def _is_layer_type_instance(self, layer):
        if ((self.layer_type == 'image' and isinstance(layer, napari.layers.image.image.Image)) or
                (self.layer_type == 'points' and isinstance(layer, napari.layers.points.points.Points)) or
                (self.layer_type == 'tracks' and isinstance(layer, napari.layers.tracks.tracks.Tracks))):
            return True
        return False

    def _on_layer_change(self, e):
        """Callback called when a napari layer is updated so the layer list can be updated also

                Parameters
                ----------
                e: QObject
                    Qt event

        """

        current_text = self._input_layer_box.currentText()
        self._input_layer_box.clear()

        still_here = False
        for layer in self.viewer.layers:
            if self._is_layer_type_instance(layer):
                self._input_layer_box.addItem(layer.name)
                if layer.name == current_text:
                    still_here = True
        if still_here:
            self._input_layer_box.setCurrentText(current_text)

    def num_layers(self):
        return self._input_layer_box.count()

    def layer_name(self):
        return self._input_layer_box.currentText()


class GEMspaPlugin(QWidget):

    """Definition of a GEMspa napari plugin

    Parameters
    ----------
    napari_viewer: Viewer
        Napari viewer

    """

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.title = 'GEMspa plugin'
        self.main_tab = None

        self.image_layer_widget = None
        self.points_layer_widget = None
        self.tracks_layer_widget = None

        self.selected_widget = None
        self.run_btn = None
        self.log_widget = None

        self._init_ui()

    def _init_ui(self):
        """Initialize the plugin graphical interface
        """
        layout = QVBoxLayout()

        # Title of widget
        title_label = QLabel(self.title)
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title_label)

        # Layer combo boxes (image, points, tracks)
        # Will be visible depending on the currently selected widget
        # These widgets know to update their content if the napari layers are modified
        self.image_layer_widget = GEMspaLayerInput(self.viewer, "image")
        layout.addWidget(self.image_layer_widget)

        self.points_layer_widget = GEMspaLayerInput(self.viewer, "points")
        layout.addWidget(self.points_layer_widget)

        self.tracks_layer_widget = GEMspaLayerInput(self.viewer, "tracks")
        layout.addWidget(self.tracks_layer_widget)

        # Tab widget for switching between subwidgets
        self.main_tab = QTabWidget()
        layout.addWidget(self.main_tab)

        # Subwidget 1 : Localization
        widget = GEMspaLocateWidget(self.viewer)
        self.main_tab.addTab(widget, "Locate features")

        # Subwidget 2 : Link features
        widget = GEMspaLinkWidget(self.viewer)
        self.main_tab.addTab(widget, "Link features")

        # Subwidget 3 : Analysis
        widget = GEMspaAnalyzeWidget(self.viewer)
        self.main_tab.addTab(widget, "Analyze tracks")

        # Run button
        self.run_btn = QPushButton('Run')
        layout.addWidget(self.run_btn)

        # Log widget
        self.log_widget = GEMspaLogWidget()
        layout.addWidget(self.log_widget)

        self.setLayout(layout)
        self.main_tab.setCurrentIndex(0)
        self.on_current_tab_changed(0)

        # Connects
        self.main_tab.currentChanged.connect(self.on_current_tab_changed)
        self.run_btn.released.connect(self.run)

    def on_current_tab_changed(self, index):
        self.selected_widget = self.main_tab.currentWidget()

        if self.selected_widget.name == 'GEMspaLocateWidget':
            self.image_layer_widget.setVisible(True)
            self.points_layer_widget.setVisible(False)
            self.tracks_layer_widget.setVisible(False)
        elif self.selected_widget.name == 'GEMspaLinkWidget':
            self.image_layer_widget.setVisible(False)
            self.points_layer_widget.setVisible(True)
            self.tracks_layer_widget.setVisible(False)
        elif self.selected_widget.name == 'GEMspaAnalyzeWidget':
            self.image_layer_widget.setVisible(False)
            self.points_layer_widget.setVisible(False)
            self.tracks_layer_widget.setVisible(True)
        else:
            print("Invalid widget name")
            raise ValueError

    def run(self):
        """Check inputs and start thread"""
        if self.selected_widget.check_inputs():
            if self.selected_widget.name == 'GEMspaLocateWidget':
                if self.image_layer_widget.num_layers() > 0:
                    self.selected_widget.start_task(self.image_layer_widget.layer_name(),
                                                    self.log_widget)
                else:
                    self.selected_widget.show_error('No Image layers.')
            elif self.selected_widget.name == 'GEMspaLinkWidget':
                if self.points_layer_widget.num_layers() > 0:
                    self.selected_widget.start_task(self.points_layer_widget.layer_name(),
                                                    self.log_widget)
                else:
                    self.selected_widget.show_error('No Points layers.')
            elif self.selected_widget.name == 'GEMspaAnalyzeWidget':
                if self.tracks_layer_widget.num_layers() > 0:
                    self.selected_widget.start_task(self.tracks_layer_widget.layer_name(),
                                                    self.log_widget)
                else:
                    self.selected_widget.show_error('No Tracks layers.')
            else:
                print("Invalid widget name")
                raise ValueError




