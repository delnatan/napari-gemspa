import os
from napari_gemspa._gemspa_locate_widget import GEMspaLocateWorker
from skimage import io


def check_update_data(out_dict):
    # TODO check value of out_dict is as expected given the test data

    return len(out_dict.keys()) == 3


def make_state_dict(viewer, layer_names):

    inputs_dict = {'image_layer_name': layer_names['image'],
                   'image_layer_data': viewer.layers[layer_names['image']].data,
                   'image_layer_scale': viewer.layers[layer_names['image']].scale,
                   'frame': viewer.dims.current_step[0]
                   }
    if 'labels' in layer_names:
        inputs_dict['labels_layer_name'] = layer_names['labels']
        inputs_dict['labels_layer_data'] = viewer.layers[layer_names['labels']].data

    return {'name': "GEMspaLocateWidget",
            'inputs': inputs_dict,
            'parameters': {'frame_start': 0,
                           'frame_end': len(viewer.layers[layer_names['image']].data),
                           'diameter': 9,
                           'minmass': 80,
                           'maxsize': None,
                           'separation': 7,
                           'noise_size': 1,
                           'smoothing_size': None,
                           'threshold': None,
                           'percentile': 64,
                           'topn': None,
                           'invert': False,
                           'preprocess': True,
                           'current_frame': False
                           },
            }


def test_locate_worker(qtbot, make_napari_viewer):
    viewer = make_napari_viewer()

    # add some layers to the viewer - there is an example movie in the same path as this script file
    path = "/Users/snk218/Dropbox (NYU Langone Health)/mac_files/holtlab/data_and_results/"
    movie = io.imread(os.path.join(path, "Example_GEM_movies/01_HeLa_nucPfV_2h_DMSO_003.tif"))
    #path = os.path.split(os.path.realpath(__file__))[0]
    #movie = io.imread(os.path.join(path, "../../example_data/example_movie_hpne_CytoGEMs_005_1-100.tif"))
    layer = viewer.add_image(movie)

    # create worker
    my_worker = GEMspaLocateWorker()

    names = {'image': layer.name}

    state_dict = make_state_dict(viewer, names)

    # the following raises error if the worker's update_data signal (which emits a dict()) wasn't raised expected data
    # within the default timeout (60 sec)
    with qtbot.waitSignal(my_worker.update_data, raising=True, check_params_cb=check_update_data, timeout=60000) as blocker:
        my_worker.run(state_dict)



