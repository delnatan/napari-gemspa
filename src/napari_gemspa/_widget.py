"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
#import trackpy as tp
from typing import TYPE_CHECKING
from magicgui import magicgui
from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget

if TYPE_CHECKING:
    import napari


class ExampleQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        btn = QPushButton("Click me!")
        btn.clicked.connect(self._on_click)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(btn)

    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")


# @magicgui(
#      #call_button="Calculate",
#      slider_float={"widget_type": "FloatSlider", 'max': 10},
#      dropdown={"choices": ['first', 'second', 'third']},
# )
# def widget_demo(img_layer: "napari.layers.Image",
#                          maybe: bool,
#                          some_int: int,
#                          spin_float=3.14159,
#                          slider_float=4.5,
#                          string="Text goes here",
#                          dropdown='first'):
#
#     print(f"you have selected {img_layer}")


@magic_factory(
    slider_float={"widget_type": "FloatSlider", 'max': 10},
    dropdown={"choices": ['first', 'second', 'third']},
)
def example_magic_widget(img_layer: "napari.layers.Image",
                         threshold: int,
                         slider_float=4.5,
                         dropdown='first',
                         ):
    print(f"you have selected {img_layer}")

    # Show the gui elements for particle localization

    first_frame = img_layer.data[0]
    # f = tp.locate(raw_image=first_frame,
    #               diameter=11,
    #               minmass=None,
    #               maxsize=None,
    #               separation=None,
    #               noise_size=1,
    #               smoothing_size=None,
    #               threshold=None,
    #               invert=False,
    #               percentile=64,
    #               topn=None,
    #               preprocess=True,
    #               max_iterations=10,
    #               filter_before=None,
    #               filter_after=None,
    #               characterize=True,
    #               engine='auto')




#def example_magic_widget(img_layer: "napari.layers.Image"):
#    print(f"you have selected {img_layer}")


# Uses the `autogenerate: true` flag in the plugin manifest
# to indicate it should be wrapped as a magicgui to autogenerate
# a widget.
def example_function_widget(img_layer: "napari.layers.Image"):
    print(f"you have selected {img_layer}")
