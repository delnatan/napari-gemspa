from ._gemspa_widget import GEMspaWidget, GEMspaWorker
import pandas as pd
import numpy as np
from gemspa_spt import ParticleTracks
from qtpy.QtWidgets import (QGridLayout, QLabel, QLineEdit, QCheckBox)
from qtpy.QtCore import Slot


"""Defines: GEMspaAnalyzeWidget, GEMspaAnalyzeWorker"""


class GEMspaAnalyzeWorker(GEMspaWorker):
    """Worker for the Analyze plugin

    """

    def __init__(self):
        super().__init__()

    @Slot(dict)
    def run(self, state):
        """Execute the processing"""

        input_params = state['inputs']
        state_params = state['parameters']

        batch = state_params['batch']
        track_id = state_params['track_id']
        microns_per_pixel = state_params['microns_per_pixel']
        time_lag_sec = state_params['time_lag_sec']
        min_len_fit = state_params['min_len_fit']
        max_lagtime_fit = state_params['max_lagtime_fit']
        error_term_fit = state_params['error_term_fit']

        tracks_layer_data = input_params['tracks_layer_data']
        tracks = ParticleTracks(tracks_layer_data)
        tracks.microns_per_pixel = microns_per_pixel
        tracks.time_lag_sec = time_lag_sec

        out_data = dict()
        if batch:

            # Ensemble average effective Diffusion (linear) and alpha (log-log)
            msds = tracks.msd_all_tracks()
            ens_msds, n_ens_tracks = tracks.ensemble_avg_msd()

            # fit ensemble MSD, get D and alpha
            D, E, r_squared1 = tracks.fit_msd_linear(t=ens_msds[1:, 0], msd=ens_msds[1:, 4], dim=2,
                                                     max_lagtime=max_lagtime_fit, err=error_term_fit)
            K, alpha, r_squared2 = tracks.fit_msd_loglog(t=ens_msds[1:, 0], msd=ens_msds[1:, 4], dim=2,
                                                         max_lagtime=max_lagtime_fit)
            data = [['sum',
                     D,
                     E,
                     r_squared1,
                     K,
                     alpha,
                     r_squared2]]
            ens_fit_data = pd.DataFrame(data, columns=['dim', 'D', 'E', 'r_sq (lin)', 'K', 'a', 'r_sq (log)'])

            # fit the msd of each track - linear and loglog scale
            tracks.fit_msd_all_tracks(linear_fit=True, min_len=min_len_fit, max_lagtime=max_lagtime_fit,
                                      err=error_term_fit)
            tracks.fit_msd_all_tracks(linear_fit=False, min_len=min_len_fit, max_lagtime=max_lagtime_fit,
                                      err=error_term_fit)

            self.log.emit(f"Total number of tracks: {len(tracks.track_ids)}\n" +
                          f"After length filter, number of tracks: {len(tracks.linear_fit_results)}\n")

            # emit the output data:

            # Gather the fit data
            all_fit_data = pd.DataFrame(np.concatenate([tracks.linear_fit_results, tracks.loglog_fit_results[:, 2:]], axis=1),
                                      columns=['track_id', 'dim', 'D', 'E', 'r_sq (lin)', 'K', 'a', 'r_sq (log)'])
            all_fit_data.drop('dim', axis=1, inplace=True)
            # all_fit_data = all_fit_data.round({'D': 4, 'E': 4, 'r_sq (lin)': 2, 'K': 4, 'a': 4, 'r_sq (log)': 2})

            # Merge fit results with track data
            track_data = pd.DataFrame(tracks.tracks, columns=['track_id', 't', 'z', 'y', 'x'])
            merged_data = track_data.merge(all_fit_data, how='right', on='track_id')

            # Add frame start and frame end to the fit results
            frames = merged_data.groupby('track_id', as_index=False).agg(frame_start=('t', 'min'), frame_end=('t', 'max'))
            merged_data = frames.merge(merged_data, how='right', on='track_id')

            out_data = {'df': merged_data,
                        'summary_data': {'ens_fit_results': ens_fit_data,
                                         'ens_msd': ens_msds[1:max_lagtime_fit + 1, [0, 4]]
                                         }
                        }

        else:
            if track_id in tracks.track_lengths[:, 0]:

                msd = tracks.msd(track_id, fft=True)
                frames = tracks.tracks[tracks.tracks[:, 0] == track_id, 1]

                # Fit for Diffusion coefficient etc
                data = []
                for dim in tracks.dim_columns.keys():
                    if dim == 'z':
                        continue
                    col = tracks.dim_columns[dim]

                    # there is no track id so reduce column index by 1
                    col -= 1

                    if dim == 'sum':
                        d = tracks.dimension
                    else:
                        d = 1

                    D, E, r_squared1 = tracks.fit_msd_linear(t=msd[1:, 0], msd=msd[1:, col], dim=d,
                                                             max_lagtime=max_lagtime_fit, err=error_term_fit)
                    K, alpha, r_squared2 = tracks.fit_msd_loglog(t=msd[1:, 0], msd=msd[1:, col], dim=d,
                                                                 max_lagtime=max_lagtime_fit)
                    data.append([track_id,
                                 frames.min(),
                                 frames.max(),
                                 dim,
                                 D,
                                 E,
                                 r_squared1,
                                 K,
                                 alpha,
                                 r_squared2])

                data = pd.DataFrame(data, columns=['track_id', 'start', 'end', 'dim',
                                                   'D', 'E', 'r_sq (lin)', 'K', 'a', 'r_sq (log)'])

                out_data = {'summary_data': {'fit_results': data,
                                             'msd': msd[1:max_lagtime_fit + 1, [0, 4]]}
                            }
            else:
                self.log.emit(f"Track id {track_id} not found.")

        self.update_data.emit(out_data)
        super().run()


class GEMspaAnalyzeWidget(GEMspaWidget):
    """Widget for Locate Features plugin"""

    name = "GEMspaAnalyzeWidget"

    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)

        self._batch_check = None
        self._track_id_value = None
        self._microns_per_pixel_value = None
        self._time_lag_sec_value = None
        self._min_len_fit_value = None
        self._max_lagtime_fit_value = None
        self._error_term_fit_check = None

        self.init_ui()

    def init_ui(self):

        # Set up the input GUI items
        self._batch_check = QCheckBox('Process all tracks')
        self._track_id_value = QLineEdit('')
        self._microns_per_pixel_value = QLineEdit('0.11')
        self._time_lag_sec_value = QLineEdit('0.010')
        self._min_len_fit_value = QLineEdit('11')
        self._max_lagtime_fit_value = QLineEdit('10')
        self._error_term_fit_check = QCheckBox('Fit with error term')

        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self._batch_check, 0, 0, 1, 2)

        layout.addWidget(QLabel('Track id'), 1, 0)
        layout.addWidget(self._track_id_value, 1, 1)

        layout.addWidget(QLabel('Microns per px'), 2, 0)
        layout.addWidget(self._microns_per_pixel_value, 2, 1)

        layout.addWidget(QLabel('Time lag (s)'), 3, 0)
        layout.addWidget(self._time_lag_sec_value, 3, 1)

        layout.addWidget(QLabel('Min. track len (fit)'), 4, 0)
        layout.addWidget(self._min_len_fit_value, 4, 1)

        layout.addWidget(QLabel('Max. lag time (fit)'), 5, 0)
        layout.addWidget(self._max_lagtime_fit_value, 5, 1)

        layout.addWidget(self._error_term_fit_check, 6, 0, 1, 2)

        self.setLayout(layout)

    def start_task(self, layer_name, log_widget):
        # initialize worker and start task
        self.worker = GEMspaAnalyzeWorker()
        super().start_task(layer_name, log_widget)

    def check_inputs(self):

        if not self._batch_check.isChecked():
            # track id
            try:
                _ = int(self._track_id_value.text())
            except ValueError as err:
                self.show_error(f"Track id input must be an integer")
                return False

        # Microns per px
        try:
            _ = float(self._microns_per_pixel_value.text())
        except ValueError as err:
            self.show_error(f"Microns per pixel input must be a number")
            return False

        # Time lag (s)
        try:
            _ = float(self._time_lag_sec_value.text())
        except ValueError as err:
            self.show_error(f"Time lag input must be a number")
            return False

        # Min len (fit)
        try:
            _ = int(self._min_len_fit_value.text())
        except ValueError as err:
            self.show_error(f"Min len (fit) input must be an integer")
            return False

        # Max lag (fit)
        try:
            _ = int(self._max_lagtime_fit_value.text())
        except ValueError as err:
            self.show_error(f"Max lag time (fit) must be an integer")
            return False

        return True

    def state(self, layer_name) -> dict:
        if self._batch_check.isChecked():
            track_id = 0
        else:
            track_id = int(self._track_id_value.text())

        return {'name': self.name,
                'inputs': {'tracks_layer_name': layer_name,
                           'tracks_layer_data': self.viewer.layers[layer_name].data,
                           'tracks_layer_props': self.viewer.layers[layer_name].properties
                           },
                'parameters': {'batch': self._batch_check.isChecked(),
                               'track_id': track_id,
                               'microns_per_pixel': float(self._microns_per_pixel_value.text()),
                               'time_lag_sec': float(self._time_lag_sec_value.text()),
                               'min_len_fit': int(self._min_len_fit_value.text()),
                               'max_lagtime_fit': int(self._max_lagtime_fit_value.text()),
                               'error_term_fit': self._error_term_fit_check.isChecked()
                               },
                }

    def update_data(self, out_dict):
        """Set the worker outputs to napari layers"""

        layer = None
        if 'df' in out_dict:
            df = out_dict['df']
            data = df[['track_id', 't', 'z', 'y', 'x']].to_numpy()

            props = {}
            for col in df.columns:
                if col not in ['track_id', 't', 'z', 'y', 'x']:
                    props[col] = df[col].to_numpy()

            layer = self.viewer.add_tracks(data, properties=props, name='Analyzed tracks')

            # Show table of properties (analysis results) - only show one line for each track
            df_unique = df.drop_duplicates(subset='track_id', keep='first')
            df_unique = df_unique.drop(labels=['t', 'z', 'y', 'x'], axis=1)

            properties_viewer = self._new_properties_viewer(layer.name)
            properties_viewer.reload_from_pandas(df_unique)
            properties_viewer.show()

        if 'summary_data' in out_dict:
            if layer is not None:
                title = layer.name
            else:
                title = "Single track"
            plots_viewer = self._new_plots_viewer(title)
            plots_viewer.plot_analyze_results(out_dict['summary_data'])
            plots_viewer.show()


# Plot of all tracks on static image (all time lags)
# 
# When selecting a particular track, show it’s track on a plot as well as the MSDs
#
# Add step size and each angle to the tracks layer data

