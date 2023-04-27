import napari
import os
from skimage import io
from napari_gemspa import locate_widget, link_widget, analyze_traj_widget


def create_viewer():
    # create a new napari viewer object
    viewer = napari.Viewer()

    # add some layers to the viewer - there is an example movie in the same path as this script file
    path = os.path.split(os.path.realpath(__file__))[0]
    example_tif_movie = "example_movie.tif"
    movie = io.imread(os.path.join(path, example_tif_movie))
    viewer.add_image(movie)

    full_test=True
    if full_test:

        # Perform locate...
        my_widget = locate_widget()
        print("Locating spots...")
        layer_data = my_widget(viewer,
                               viewer.layers[0],
                               batch=True,
                               diameter=11,
                               min_mass=75,
                               invert=False)
        #viewer.add_points(layer_data[0], **layer_data[1])

        # Perform link...
        my_widget = link_widget()
        layer_data = my_widget(viewer,
                               viewer.layers[1],
                               link_range=5,
                               memory=0,
                               min_frames=3)
        #viewer.add_tracks(layer_data[0], **layer_data[1])

        # Call analyze...
        my_widget = analyze_traj_widget()
        layer_data = my_widget(viewer,
                               viewer.layers[0],
                               viewer.layers[2],
                               batch=True,
                               track_id=7,
                               microns_per_pixel=0.11,
                               time_lag_sec=0.010,
                               max_lagtime_fit=10,
                               min_len_fit=11,
                               error_term_fit=True)

    # return the viewer object
    return viewer


if __name__ == "__main__":
    # create a viewer object using the function
    my_viewer = create_viewer()

    # display the viewer
    napari.run()
