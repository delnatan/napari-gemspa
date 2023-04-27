"""
This module is an example of a bare bones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
import trackpy as tp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from napari.types import ImageData
from napari import Viewer
import napari.types
from typing import TYPE_CHECKING
from magicgui import magic_factory
from qtpy.QtCore import Qt
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QTableWidget, QTableWidgetItem

#from qtpy.QtWidgets import QMainWindow
#from ._analyze_tracks_dialog import Ui_Dialog
from gemspa_spt import ParticleTracks
#from pyqtgraph import PlotWidget, plot
#import pyqtgraph as pg

if TYPE_CHECKING:
    import napari


def show_plots(df, napari_viewer, data_type="features", width=200, height=300):
    dock_widgets = []
    if data_type == "features":
        # Show Mass histogram
        fig, ax = plt.subplots()
        ax.hist(df['mass'], bins=20)
        ax.set(xlabel='mass', ylabel='count')
        ax.set_title('mass')
        dock_widgets.append(napari_viewer.window.add_dock_widget(fig.canvas, name="mass histogram", area="right"))

        # Show subpixel bias (histogram fractional part of x/y positions)
        fig, axs = plt.subplots(1, 2)
        axs[0].hist(np.modf(df['x'].to_numpy())[0], bins=20)
        axs[1].hist(np.modf(df['y'].to_numpy())[0], bins=20)
        axs[0].set(xlabel='x', ylabel='count')
        axs[1].set(xlabel='y', ylabel='count')
        axs[0].set_title('x')
        axs[1].set_title('y')
        dock_widgets.append(napari_viewer.window.add_dock_widget(fig.canvas, name="sub px bias", area="right"))

    elif data_type == "links":
        # Show plot of mass vs. size
        mean_t = df.groupby('particle').mean()

        fig, ax = plt.subplots()
        ax.plot(mean_t['mass'], mean_t['size'], 'ko', alpha=0.1)
        ax.set(xlabel='mass', ylabel='size')
        ax.set_title('mass vs size')
        dock_widgets.append(napari_viewer.window.add_dock_widget(fig.canvas, name="mass vs size", area="right"))

        fig, ax = plt.subplots()
        ax.plot(mean_t['mass'], mean_t['ecc'], 'ko', alpha=0.3)
        ax.set(xlabel='mass', ylabel='eccentricity (0=circular)')
        ax.set_title('mass vs eccentricity')
        dock_widgets.append(napari_viewer.window.add_dock_widget(fig.canvas, name="mass vs eccentricity", area="right"))

    else:
        raise ValueError("data_type invalid.")

    # Resize so height of widgets with the plots are larger
    #for dock_widget in dock_widgets:
    #    dock_widget.setFloating(False)
    #napari_viewer.window.qt_window.resizeDocks(dock_widgets, [height]*len(dock_widgets), Qt.Vertical)
    #napari_viewer.window.qt_window.resizeDocks(dock_widgets, [width]*len(dock_widgets), Qt.Horizontal)


def make_napari_layer(df, data_type='features'):
    if data_type == 'features':
        # From results, annotate image by adding a points layer
        if 'frame' in df.columns:
            data = df[['frame', 'y', 'x']].astype('int').to_numpy()
        else:
            data = df[['y', 'x']].astype('int').to_numpy()

        # return points data
        props = {}
        for col in df.columns:
            if not (col in ['y', 'x', 'frame']):
                props[col] = df[col].to_numpy()
        return napari.types.LayerDataTuple((data,
                                            {'properties': props, 'face_color': 'transparent', 'edge_color': 'red'},
                                            'points'))
    elif data_type == 'links':
        df.index.name = 'index'  # pandas complains when index name and column name are the same
        df = df.sort_values(by=['particle', 'frame'], axis=0, ascending=True)
        if 'z' not in df.columns:
            df['z'] = 0
        data = df[['particle', 'frame', 'z', 'y', 'x']].to_numpy()

        props = {}
        for col in df.columns:
            if col not in ['particle', 'frame', 'z', 'y', 'x']:
                props[col] = df[col].to_numpy()

        return napari.types.LayerDataTuple((data, {'properties': props}, 'tracks'))

    else:
        raise ValueError("data_type invalid.")


def make_trackpy_table(layer, data_type='features'):
    if data_type == 'features':
        df = pd.DataFrame()
        df['y'] = layer.data[:, 1]
        df['x'] = layer.data[:, 2]
        for col in layer.properties.keys():
            df[col] = layer.properties[col]
        df['frame'] = layer.data[:, 0]
    elif data_type == 'links':
        df = pd.DataFrame()
        df['y'] = layer.data[:, 3]
        df['x'] = layer.data[:, 4]
        for col in layer.properties.keys():
            if col != 'track_id':
                df[col] = layer.properties[col]
        df['frame'] = layer.data[:, 1]
        df['particle'] = layer.data[:, 0]
    else:
        raise ValueError("data_type invalid.")

    return df


@magic_factory
def locate_widget(viewer: Viewer,
                  img_layer: napari.layers.Image,
                  batch: bool = False,
                  diameter: int = 11,
                  min_mass: int = 0,
                  invert: bool = False
                  ) -> napari.types.LayerDataTuple:

    if img_layer is not None:
        tp.quiet()
        if batch:
            # process the entire movie - all frames
            f = tp.batch(frames=img_layer.data,
                         diameter=diameter,
                         minmass=min_mass,
                         invert=invert)
        else:
            # Only process the current frame
            cur_frame = img_layer._data_view
            f = tp.locate(raw_image=cur_frame,
                          diameter=diameter,
                          minmass=min_mass,
                          invert=invert)

            show_plots(f, viewer, data_type="features")

        return make_napari_layer(f, data_type='features')


@magic_factory
def link_widget(viewer: Viewer,
                points_layer: napari.layers.Points,
                link_range: int = 5,
                memory: int = 0,
                min_frames: int = 3
                ) -> napari.types.LayerDataTuple:

    if points_layer is not None:
        tp.quiet()

        # build trackpy data frame
        if len(points_layer.data.shape) > 1 and points_layer.data.shape[1] >= 3:

            f = make_trackpy_table(points_layer, data_type='features')

            # perform linking
            t = tp.link(f, search_range=link_range, memory=memory)
            print('Number of particles:', t['particle'].nunique())

            # Filter spurious trajectories
            t1 = tp.filter_stubs(t, threshold=min_frames)
            print('After filter_stubs, number of particles:', t1['particle'].nunique())

            show_plots(t1, viewer, data_type="links")

            show_info(f"Number of particles: {t['particle'].nunique()}\n" +
                      f"After filter stubs, number of particles: {t1['particle'].nunique()}")

            # Return tracks layer
            return make_napari_layer(t1, data_type='links')

        else:
            show_info("Points layer data not compatible with linking.")


@magic_factory
def filter_link_widget(viewer: Viewer,
                       tracks_layer: napari.layers.Tracks,
                       min_frames: int = 3,
                       min_mass: float = 0,
                       max_mass: float = 0,
                       min_size: float = 0,
                       max_size: float = 0,
                       min_ecc: float = 0,
                       max_ecc: float = 1
                       ) -> napari.types.LayerDataTuple:

    if tracks_layer is not None:
        tp.quiet()

        # Load data from tracks layer
        t0 = make_trackpy_table(tracks_layer, data_type="links")

        # filter by min frames
        t1 = tp.filter_stubs(t0, threshold=min_frames)

        # filter by mass, size, eccentricity
        # Show plot of mass vs. size
        mean_t = t1.groupby('particle').mean()

        mean_t = mean_t[mean_t['mass'] >= min_mass]
        if max_mass > 0:
            mean_t = mean_t[mean_t['mass'] <= max_mass]

        mean_t = mean_t[mean_t['size'] >= min_size]
        if max_size > 0:
            mean_t = mean_t[mean_t['size'] <= max_size]

        mean_t = mean_t[mean_t['ecc'] >= min_ecc]
        if max_ecc < 1:
            mean_t = mean_t[mean_t['ecc'] <= max_ecc]

        t2 = t1[t1['particle'].isin(mean_t.index)]

        # show plots
        show_plots(t2, viewer, data_type="links")

        show_info(f"# of particles: {t0['particle'].nunique()}\n" +
                  f"After filter stubs, # of particles: {t1['particle'].nunique()}\n" +
                  f"After filter mass/size/ecc, # of particles: {t2['particle'].nunique()}")

        # Return tracks layer
        return make_napari_layer(t2, data_type='links')


def fill_table_widget(tw, df):
    tw.setColumnCount(len(df.columns))
    tw.setHorizontalHeaderLabels(df.columns)
    tw.setRowCount(len(df))

    for i, row in enumerate(df.iterrows()):
        for j, col in enumerate(df.columns):
            tw.setItem(i, j, QTableWidgetItem(str(row[1][col])))


@magic_factory
def analyze_traj_widget(viewer: Viewer,
                        img_layer: napari.layers.Image,
                        tracks_layer: napari.layers.Tracks,
                        batch: bool = False,
                        track_id: int = 0,
                        microns_per_pixel: float = 0.11,
                        time_lag_sec: float = 0.010,
                        min_len_fit: int = 11,
                        max_lagtime_fit: int = 10,
                        error_term_fit: bool = True
                        ) -> napari.types.LayerDataTuple:

    if tracks_layer is not None:
        # Read tracks layer data
        tracks = ParticleTracks(tracks_layer.data)

        if batch:
            # MSD for all tracks
            tracks.microns_per_pixel = microns_per_pixel
            tracks.time_lag_sec = time_lag_sec

            # Ensemble average eff-D (linear) and alpha (log-log)
            msds = tracks.msd_all_tracks()
            ens_msds, n_ens_tracks = tracks.ensemble_avg_msd()

            # Ensemble MSD plot - only showing the 2d (sum)
            fig, axs = plt.subplots(1, 2)
            axs[0].scatter(ens_msds[1:max_lagtime_fit + 1, 0], ens_msds[1:max_lagtime_fit + 1, 4])
            axs[1].scatter(ens_msds[1:max_lagtime_fit + 1, 0], ens_msds[1:max_lagtime_fit + 1, 4])
            axs[1].set_xscale('log')
            axs[1].set_yscale('log')
            axs[0].set(xlabel='time lag (sec)', ylabel='msd')
            axs[1].set(xlabel='time lag (sec)', ylabel='msd')
            axs[0].set_title('ensemble-avg msd')
            axs[1].set_title('ensemble-avg msd (log-log)')
            viewer.window.add_dock_widget(fig.canvas, name="ensemble-avg MSD", area="right")

            # fit ensemble MSD, get D and alpha
            D, E, r_squared1 = tracks.fit_msd_linear(t=ens_msds[1:, 0], msd=ens_msds[1:, 4], dim=2,
                                                     max_lagtime=max_lagtime_fit, err=error_term_fit)
            K, alpha, r_squared2 = tracks.fit_msd_loglog(t=ens_msds[1:, 0], msd=ens_msds[1:, 4], dim=2,
                                                         max_lagtime=max_lagtime_fit)
            data = [['x+y',
                     round(D, 4),
                     round(E, 4),
                     round(r_squared1, 2),
                     round(K, 4),
                     round(alpha, 4),
                     round(r_squared2, 2)]]
            data = pd.DataFrame(data, columns=['dim', 'D', 'E', 'r_sq (lin)', 'K', 'a', 'r_sq (log)'])

            # Table of results
            table_widget = QTableWidget()
            fill_table_widget(table_widget, data)
            viewer.window.add_dock_widget(table_widget, name="Results", area="bottom")

            # fit the msd of each track - linear and loglog scale
            tracks.fit_msd_all_tracks(linear_fit=True, min_len=min_len_fit, max_lagtime=max_lagtime_fit,
                                      err=error_term_fit)
            tracks.fit_msd_all_tracks(linear_fit=False, min_len=min_len_fit, max_lagtime=max_lagtime_fit,
                                      err=error_term_fit)

            # Gather the fit data and fill table
            table_widget = QTableWidget()
            data = pd.DataFrame(np.concatenate([tracks.linear_fit_results, tracks.loglog_fit_results[:, 2:]], axis=1),
                                columns=['track_id', 'dim', 'D', 'E', 'r_sq (lin)', 'K', 'a', 'r_sq (log)'])
            data.drop('dim', axis=1, inplace=True)
            data = data.round({'D': 4, 'E': 4, 'r_sq (lin)': 2, 'K': 4, 'a': 4, 'r_sq (log)': 2})

            fill_table_widget(table_widget, data)
            viewer.window.add_dock_widget(table_widget, name="Results-all", area="bottom")

            show_info(f"Total number of tracks: {len(tracks.track_ids)}\n" +
                      f"After length filter, number of tracks: {len(data)}\n")

            # Add information from fitting to properties of a newTracks Layer
            track_data = pd.DataFrame(tracks.tracks, columns=ParticleTracks.file_columns['napari'])
            data = track_data.merge(data, how='right', on='track_id')

            props = {}
            for col in data.columns:
                if col not in ParticleTracks.file_columns['napari']:
                    props[col] = data[col].to_numpy()

            return napari.types.LayerDataTuple((data[ParticleTracks.file_columns['napari']],
                                                {'properties': props},
                                                'tracks'))

        else:
            # MSD for the track
            tracks.microns_per_pixel = microns_per_pixel
            tracks.time_lag_sec = time_lag_sec

            msd1 = tracks.msd(track_id, fft=True)

            # Fit for Diffusion coefficient etc
            dim_dict = {'x': [3, 1], 'y': [2, 1], 'x+y': [4, 2]}
            data = []
            for dim in dim_dict.keys():
                col = dim_dict[dim][0]
                d = dim_dict[dim][1]
                D, E, r_squared1 = tracks.fit_msd_linear(t=msd1[1:, 0], msd=msd1[1:, col], dim=d,
                                                         max_lagtime=max_lagtime_fit, err=error_term_fit)
                K, alpha, r_squared2 = tracks.fit_msd_loglog(t=msd1[1:, 0], msd=msd1[1:, col], dim=d,
                                                             max_lagtime=max_lagtime_fit)
                data.append([dim,
                             round(D, 4),
                             round(E, 4),
                             round(r_squared1, 2),
                             round(K, 4),
                             round(alpha, 4),
                             round(r_squared2, 2)])
            data = pd.DataFrame(data, columns=['dim', 'D', 'E', 'r_sq (lin)', 'K', 'a', 'r_sq (log)'])

            # Make plots, display on new docked widgets
            # (MSD of track, with fit line, linear and loglog scale)
            # Display the D's, E's, r_sq's AND K's, alpha's, r_sq's
            fig, axs = plt.subplots(1, 2)
            axs[0].scatter(msd1[1:max_lagtime_fit+1, 0], msd1[1:max_lagtime_fit+1, 4])
            axs[1].scatter(msd1[1:max_lagtime_fit+1, 0], msd1[1:max_lagtime_fit+1, 4])
            axs[1].set_xscale('log')
            axs[1].set_yscale('log')
            axs[0].set(xlabel='time lag (sec)', ylabel='msd')
            axs[1].set(xlabel='time lag (sec)', ylabel='msd')
            axs[0].set_title('msd')
            axs[1].set_title('msd (log-log)')
            viewer.window.add_dock_widget(fig.canvas, name="MSD", area="right")

            # Table of results
            table_widget = QTableWidget()
            fill_table_widget(table_widget, data)
            viewer.window.add_dock_widget(table_widget, name="Results", area="bottom")


# ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ####
# class AnalyzeWidget(QMainWindow):
#     def __init__(self, napari_viewer, parent=None):
#         super().__init__(parent)
#
#         self.ui = Ui_Dialog()
#         self.ui.setupUi(self)
#         self.connectSignalsSlots()
#
#         self.viewer = napari_viewer
#
#     def connectSignalsSlots(self):
#         self.ui.go_pushButton.clicked.connect(self._on_click_go_pushButton)
#
#     def _on_click_go_pushButton(self):
#         # get the tracks layer
#         # perform analysis
#         print("In Run")


# class AnalyzeMainWindow(QMainWindow):
#
#     def __init__(self, napari_viewer, *args, **kwargs):
#         super(AnalyzeMainWindow, self).__init__(*args, **kwargs)
#
#         self.viewer = napari_viewer
#
#         # draw the input GUI elements
#         # batch: bool = False,
#         # track_id: int = 0,
#         # microns_per_pixel: float = 0.11,
#         # frames_per_sec: float = 0.01,
#         # fit_max_lagtime: int = 11,
#         # fit_error_term: bool = True,
#
#         #btn = QtWidgets.QPushButton("Click me!")
#         #btn.clicked.connect(self._on_click)
#
#         #self.setLayout(QHBoxLayout())
#         #self.layout().addWidget(btn)
#
#     def _on_click(self):
#
#         #if tracks_layer is not None:
#         # # Read tracks layer data
#         #         tracks = ParticleTracks(tracks_layer.data)
#         #
#         #         if batch:
#         #             pass
#         #             # Ensemble average eff-D (linear) and alpha (log-log)
#         #
#         #             # Output plots
#         #
#         #             # add to properties of tracks layer: will contain information on:
#         #             # eff-D, alpha, etc
#         #
#         #             # Return tracks layer
#         #             return ()
#         #         else:
#         #             # MSD for the track
#         #             tracks.microns_per_pixel = microns_per_pixel
#         #             tracks.frames_per_sec = frames_per_sec
#         #             lag, msd_x, msd_y, msd_2d = tracks.msd(track_id=track_id)
#         #
#         #             # Fit for Diffusion coefficient (linear, assume alpha==1)
#         #             tracks.fit_max_lagtime = fit_max_lagtime
#         #             tracks.fit_error_term = fit_error_term
#         #             D, Err, r_sq = tracks.linear_fit_D(track_id=track_id)
#         #
#         #             # Fit for alpha (log-log fit; alpha is the slope)
#         #             K, alpha, r_sq = tracks.linear_fit_D(track_id=track_id)
#         #
#         #             # Make plots, display on widget
#         #             #main = MainWindow()
#         #             #main.show()
#
#         self.graphWidget = pg.PlotWidget()
#         self.setCentralWidget(self.graphWidget)
#
#         hour = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#         temperature = [30, 32, 34, 32, 33, 31, 29, 32, 35, 45]
#
#         # plot data: x, y values
#         self.graphWidget.plot(hour, temperature)


from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout, QPushButton, QWidget, QAbstractItemView
import qtpy.QtCore


class TracksPropertiesTable(QWidget):
    """Widget to display results from analysis of track data
    Parameters
    ----------
    napari_viewer: QWidget
        The napari viewer
    """
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.layer_name = ''

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.tableWidget = QTableWidget()
        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.tableWidget)
        self.setLayout(layout)

    def reload(self):
        """Reload the particles properties from the layers to the table widget"""
        particles = self.viewer.layers[self.layer_name].data
        print(particles)
        properties = self.viewer.layers[self.layer_name].properties
        headers = []
        if particles.shape[1] == 3:
            headers = ['T', 'Y', 'X']
        elif particles.shape[1] == 4:
            headers = ['T', 'Z', 'Y', 'X']

        for key in properties:
            headers.append(key)
        self.tableWidget.setColumnCount(len(headers))
        self.tableWidget.setHorizontalHeaderLabels(headers)
        self.tableWidget.setRowCount(particles.shape[0])

        col = 0
        for line in range(particles.shape[0]):
            col = -1

            # T
            if particles.shape[1] == 4:
                col += 1
                self.tableWidget.setItem(line, col,
                                         QTableWidgetItem(
                                             str(particles[line, col])))
            # T or Z
            if particles.shape[1] >= 3:
                col += 1
                self.tableWidget.setItem(line, col,
                                         QTableWidgetItem(
                                             str(particles[line, col])))
            # Y
            col += 1
            self.tableWidget.setItem(line, col,
                                     QTableWidgetItem(
                                         str(particles[line, col])))
            # X
            col += 1
            self.tableWidget.setItem(line, col,
                                     QTableWidgetItem(
                                         str(particles[line, col])))
        # properties
        for key in properties:
            col += 1
            prop = properties[key]
            for line in range(len(prop)):
                self.tableWidget.setItem(line, col,
                                         QTableWidgetItem(str(prop[line])))


class AnalyzeQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.properties_viewer = TracksPropertiesTable(self.viewer)
        self.properties_viewer.setVisible(False)
        self.properties_viewer.setWindowFlag(qtpy.QtCore.Qt.WindowStaysOnTopHint, True)

        btn = QPushButton("Click me!")
        btn.clicked.connect(self._on_click)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(btn)

    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")

        # properties_viewer.reload()
        self.properties_viewer.show()
