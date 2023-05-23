"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/guides.html?#readers
"""

from ._gemspa_file_import_widget import GEMspaFileImport


def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    # handle both a string and a list of strings
    paths = [path] if isinstance(path, str) else path
    for path in paths:
        # if we know we cannot read the file, we immediately return None.
        if not (path.endswith(".csv") or path.endswith('.txt') or path.endswith('.tsv')):
            return None

    # otherwise we return the *function* that can read ``path``.
    return reader_function


def _read_layer_data(df, columns):

    data = df[columns].to_numpy()

    props = {}
    for col in df.columns:
        if col not in columns:
            props[col] = df[col].to_numpy()

    return data, props


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    # handle both a string and a list of strings
    paths = [path] if isinstance(path, str) else path

    # load each file into pandas data frame, check format and add layer
    layer_data = []
    for path in paths:
        file_import = GEMspaFileImport(path, "gemspa")
        layer_data.append(file_import.get_layer_data())

    return layer_data
