import os

import napari
from skimage import io

from napari_gemspa import napari_get_reader


def create_viewer():
    # create a new napari viewer object
    viewer = napari.Viewer()

    # add some layers to the viewer - there is an example movie in the same
    # path as this script file
    path = os.path.split(os.path.realpath(__file__))[0]
    movie = io.imread(
        os.path.join(
            path,
            "../../example_data/example_movie_hpne_CytoGEMs_005_1-100.tif",
        )
    )
    viewer.add_image(movie)

    test_analyze = True
    if test_analyze:
        # read in a points layer
        example_points_layer = os.path.join(
            path, "../../example_data/example_features.txt"
        )
        reader = napari_get_reader(example_points_layer)
        layer_data_list = reader(example_points_layer)
        layer_data_tuple = layer_data_list[0]
        viewer.add_points(layer_data_tuple[0], **layer_data_tuple[1])

        # read in a tracks layer
        example_tracks_layer = os.path.join(
            path, "../../example_data/example_linked_features_filt.txt"
        )
        reader = napari_get_reader(example_tracks_layer)
        layer_data_list = reader(example_tracks_layer)
        layer_data_tuple = layer_data_list[0]
        viewer.add_tracks(layer_data_tuple[0], **layer_data_tuple[1])

        # read in a labels layer
        example_labels_layer = os.path.join(
            path, "../../example_data/example_labeled_mask.tif"
        )
        mask = io.imread(example_labels_layer)
        viewer.add_labels(mask)

    # return the viewer object
    return viewer


if __name__ == "__main__":
    # create a viewer object using the function
    my_viewer = create_viewer()

    # display the viewer
    napari.run()
