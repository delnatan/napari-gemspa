import napari
import os
from skimage import io


def create_viewer():
    # create a new napari viewer object
    viewer = napari.Viewer()

    # add some layers to the viewer - there is an example movie in the same path as this script file
    path = os.path.split(os.path.realpath(__file__))[0]
    example_tif_movie = "example_movie.tif"
    movie = io.imread(os.path.join(path, example_tif_movie))
    viewer.add_image(movie)

    # return the viewer object
    return viewer


if __name__ == "__main__":
    # create a viewer object using the function
    my_viewer = create_viewer()

    # display the viewer
    napari.run()
