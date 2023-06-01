import os

from skimage import io

from napari_gemspa._gemspa_locate_widget import GEMspaLocateWorker

current_frame = 50
diameter = 9
minmass = 100
separation = 7


def check_update_data_single(out_dict):
    # return True if all tests pass.
    if (
        "df" in out_dict
        and "scale" in out_dict
        and len(out_dict["scale"]) == 3
        and "diameter" in out_dict
        and out_dict["diameter"] == diameter
        and len(out_dict["df"]) > 60
        and len(out_dict["df"]["frame"].unique()) == 1
        and out_dict["df"]["frame"].unique()[0] == current_frame
    ):
        return True
    else:
        return False


def check_update_data(out_dict):
    # return True if all tests pass.
    if (
        "df" in out_dict
        and "scale" in out_dict
        and "diameter" in out_dict
        and len(out_dict["df"]) > 6000
        and out_dict["diameter"] == diameter
        and len(out_dict["df"]["frame"].unique()) > 1
    ):
        return True
    else:
        return False


def make_state_dict(viewer, layer_names, single_frame):
    inputs_dict = {
        "image_layer_name": layer_names["image"],
        "image_layer_data": viewer.layers[layer_names["image"]].data,
        "image_layer_scale": viewer.layers[layer_names["image"]].scale,
        "frame": current_frame,
    }
    if "labels" in layer_names:
        inputs_dict["labels_layer_name"] = layer_names["labels"]
        inputs_dict["labels_layer_data"] = viewer.layers[layer_names["labels"]].data

    return {
        "name": "GEMspaLocateWidget",
        "inputs": inputs_dict,
        "parameters": {
            "frame_start": 0,
            "frame_end": len(viewer.layers[layer_names["image"]].data),
            "diameter": diameter,
            "minmass": minmass,
            "maxsize": None,
            "separation": separation,
            "noise_size": 1,
            "smoothing_size": None,
            "threshold": None,
            "percentile": 64,
            "topn": None,
            "invert": False,
            "preprocess": True,
            "current_frame": single_frame,
        },
    }


def test_locate_worker(qtbot, make_napari_viewer):
    viewer = make_napari_viewer()

    # add some layers to the viewer - there is an example movie in the same path as this script file
    path = os.path.split(os.path.realpath(__file__))[0]
    movie = io.imread(
        os.path.join(
            path,
            "../../example_data/example_movie_hpne_CytoGEMs_005_1-100.tif",
        )
    )
    layer = viewer.add_image(movie)

    # create worker
    my_worker = GEMspaLocateWorker()
    names = {"image": layer.name}

    # Test running locate for a single frame
    state_dict = make_state_dict(viewer, names, True)
    with qtbot.waitSignal(
        my_worker.update_data,
        raising=True,
        check_params_cb=check_update_data_single,
        timeout=60000,
    ):
        my_worker.run(state_dict)

    # Test running locate for all frames
    state_dict = make_state_dict(viewer, names, False)
    with qtbot.waitSignal(
        my_worker.update_data,
        raising=True,
        check_params_cb=check_update_data,
        timeout=60000,
    ):
        my_worker.run(state_dict)
