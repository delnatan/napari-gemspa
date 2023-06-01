# def test_locate_widget_init(make_napari_viewer):
#     viewer = make_napari_viewer()
#     my_widget = GEMspaLocateWidget()
#
#     # TODO: test widget is correctly initialized
#
#
# def test_locate_widget_state(make_napari_viewer):
#     viewer = make_napari_viewer()
#     my_widget = GEMspaLocateWidget()
#
#     path = os.path.split(os.path.realpath(__file__))[0]
#     movie = io.imread(os.path.join(path, "../../example_data/example_movie_hpne_CytoGEMs_005_1-100.tif"))
#     layer = viewer.add_image(movie)
#     input_layers_dict = {'image': layer.name}
#
#     my_widget.state(input_layers_dict)
#
#     # TODO: test widget state function returns correct values
#
#
# def test_locate_widget_check_inputs(make_napari_viewer):
#
#     viewer = make_napari_viewer()
#     my_widget = GEMspaLocateWidget()
#
#     # TODO: set widget input values first, then check that the function correctly checks the inputs
#
#     my_widget.check_inputs()
#
#
# def test_locate_widget_start_task(make_napari_viewer):
#     viewer = make_napari_viewer()
#     my_widget = GEMspaLocateWidget()
#     log_widget = GEMspaLogWidget()
#
#     path = os.path.split(os.path.realpath(__file__))[0]
#     movie = io.imread(os.path.join(path, "../../example_data/example_movie_hpne_CytoGEMs_005_1-100.tif"))
#     layer = viewer.add_image(movie)
#     input_layers_dict = {'image': layer.name}
#
#     my_widget.start_task(input_layers_dict, log_widget)
#
#     # TODO: check that the Worker is run and returns correct data
