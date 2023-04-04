import napari
import numpy
from skimage import data

def create_viewer():
    # create a new napari viewer object
    viewer = napari.Viewer()

    # add some layers to the viewer
    cells = data.cells3d()[30, 1]
    viewer.add_image(cells)

    # return the viewer object
    return viewer

# create a viewer object using the function
my_viewer = create_viewer()

# display the viewer
napari.run()
