import napari
import numpy

def create_viewer():
    # create a new napari viewer object
    viewer = napari.Viewer()

    # add some layers to the viewer
    data = [(numpy.random.rand(512, 512), {})]
    viewer.add_image(data)

    # return the viewer object
    return viewer

# create a viewer object using the function
my_viewer = create_viewer()

# display the viewer
napari.run()
