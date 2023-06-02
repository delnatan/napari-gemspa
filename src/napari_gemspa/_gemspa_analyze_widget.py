"""Defines: GEMspaAnalyzeWidget, GEMspaAnalyzeWorker"""

import numpy as np
import pandas as pd
from gemspa_spt import ParticleTracks
from qtpy.QtCore import Slot
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from ._gemspa_widget import GEMspaWidget, GEMspaWorker
from ._utils import convert_to_float, convert_to_int, remove_outside_mask, show_error


class GEMspaAnalyzeWorker(GEMspaWorker):
    """Worker for the Analyze plugin"""

    def __init__(self):
        super().__init__()

    @Slot(dict)
    def run(self, state):
        """Execute the processing"""

        input_params = state["inputs"]
        state_params = state["parameters"]

        batch = state_params["batch"]
        track_id = state_params["track_id"]
        microns_per_pixel = state_params["microns_per_pixel"]
        time_lag_sec = state_params["time_lag_sec"]
        min_len_fit = state_params["min_len_fit"]
        max_lagtime_fit = state_params["max_lagtime_fit"]
        error_term_fit = state_params["error_term_fit"]

        # TODO
        # frame_start = state_params['frame_start']
        # frame_end = state_params['frame_end']

        tracks_layer_data = input_params["tracks_layer_data"]

        # Find out 2d or 3d
        if tracks_layer_data.shape[1] == 5:
            z_col = True
            track_cols = ["track_id", "frame", "z", "y", "x"]
        elif tracks_layer_data.shape[1] == 4:
            z_col = False
            track_cols = ["track_id", "frame", "y", "x"]
        else:
            raise ValueError("Tracks layer has an unexpected dimension.")

        # create data frame of track data
        track_data_df = pd.DataFrame(tracks_layer_data, columns=track_cols)

        # process the entire movie - limited by frame start/end
        # TODO
        # if batch and (frame_start is not None or frame_end is not None):
        #     frame_start, frame_end = fix_frame_limits(frame_start, frame_end, track_data_df['frame'].max() + 1)
        #     track_data_df = track_data_df[(track_data_df['frame'] >= frame_start) &
        #                                   (track_data_df['frame'] <= frame_end)]
        #     self.log.emit(f"After filter for frames from {frame_start} to {frame_end}, number of tracks: " +
        #                   f"{track_data_df['track_id'].nunique()}")

        # if labels layer was chosen, remove all tracks that contain points that are outside labeled regions
        # *note* this function is not implemented for 3D (z-dimensional) data
        # *note* if not run in batch mode (only analyze a single track), the labels layer is ignored
        if batch and "labels_layer_data" in input_params:
            labeled_mask = input_params["labels_layer_data"]
            track_data_df = remove_outside_mask(
                track_data_df, labeled_mask, id_column="track_id"
            )
            self.log.emit(
                f"Removed particles outside of mask region, number of particles: {track_data_df['track_id'].nunique()}"
            )

        # Create the gemspa-spt class object for performing analysis
        if not z_col:
            # ParticleTracks class expects a column for z-dimension
            tracks = ParticleTracks(np.insert(track_data_df.to_numpy(), 2, 0, axis=1))
        else:
            tracks = ParticleTracks(track_data_df.to_numpy())
        tracks.microns_per_pixel = microns_per_pixel
        tracks.time_lag_sec = time_lag_sec

        # add the x/y/z positions in microns to the track data for output later
        if "z" in track_data_df:
            track_data_df["z (microns)"] = track_data_df["z"] * microns_per_pixel
        track_data_df["y (microns)"] = track_data_df["y"] * microns_per_pixel
        track_data_df["x (microns)"] = track_data_df["x"] * microns_per_pixel

        out_data = dict()
        if batch:
            # Ensemble average effective Diffusion (linear) and alpha (log-log)
            msds = tracks.msd_all_tracks()
            step_sizes = tracks.step_size_all_tracks()
            r_of_g = tracks.r_of_g_all_tracks()
            ens_msds, n_ens_tracks = tracks.ensemble_avg_msd()

            # fit ensemble MSD, get D and alpha
            D, E, r_squared1 = tracks.fit_msd_linear(
                t=ens_msds[1:, 0],
                msd=ens_msds[1:, 4],
                dim=2,
                max_lagtime=max_lagtime_fit,
                err=error_term_fit,
            )
            K, alpha, r_squared2 = tracks.fit_msd_loglog(
                t=ens_msds[1:, 0],
                msd=ens_msds[1:, 4],
                dim=2,
                max_lagtime=max_lagtime_fit,
            )
            data = [["sum", D, E, r_squared1, K, alpha, r_squared2]]
            ens_fit_data = pd.DataFrame(
                data,
                columns=[
                    "dim",
                    "D",
                    "E",
                    "r_sq (lin)",
                    "K",
                    "a",
                    "r_sq (log)",
                ],
            )
            ens_fit_data = ens_fit_data.round(
                {
                    "D": 4,
                    "E": 4,
                    "r_sq (lin)": 2,
                    "K": 4,
                    "a": 4,
                    "r_sq (log)": 2,
                }
            )

            summary_data = {
                "ens_fit_results": ens_fit_data,
                "ens_msd": ens_msds[1 : max_lagtime_fit + 1, [0, 4]],
            }

            # fit the msd of each track - linear and loglog scale
            tracks.fit_msd_all_tracks(
                linear_fit=True,
                min_len=min_len_fit,
                max_lagtime=max_lagtime_fit,
                err=error_term_fit,
            )
            tracks.fit_msd_all_tracks(
                linear_fit=False,
                min_len=min_len_fit,
                max_lagtime=max_lagtime_fit,
                err=error_term_fit,
            )

            self.log.emit(
                f"Total number of tracks: {len(tracks.track_ids)}\n"
                + f"After length filter, number of tracks: {len(tracks.linear_fit_results)}\n"
            )

            # Gather the fit data and r-of-g
            all_fit_data = np.concatenate(
                [tracks.linear_fit_results, tracks.loglog_fit_results[:, 2:]],
                axis=1,
            )
            all_fit_data = pd.DataFrame(
                all_fit_data,
                columns=[
                    "track_id",
                    "dim",
                    "D",
                    "E",
                    "r_sq (lin)",
                    "K",
                    "a",
                    "r_sq (log)",
                ],
            )
            all_fit_data.drop("dim", axis=1, inplace=True)
            all_fit_data = all_fit_data.round(
                {
                    "D": 4,
                    "E": 4,
                    "r_sq (lin)": 2,
                    "K": 4,
                    "a": 4,
                    "r_sq (log)": 2,
                }
            )

            # Merge fit results with track data
            msds = pd.DataFrame(msds[:, [1, 5]], columns=["tau", "msd"])
            msds.msd = msds.msd.where(msds.tau > 0, other=np.nan)
            msds = msds.round({"msd": 4})

            step_sizes = pd.DataFrame(step_sizes[:, [1, 5]], columns=["t", "step_size"])
            step_sizes.step_size = step_sizes.step_size.where(
                step_sizes.t > 0, other=np.nan
            )
            step_sizes = step_sizes.round({"step_size": 4})

            r_of_g = pd.DataFrame(r_of_g[:, 1:], columns=["t", "radius_gyration"])
            r_of_g.radius_gyration = r_of_g.radius_gyration.where(
                r_of_g.t > 0, other=np.nan
            )
            r_of_g = r_of_g.round({"radius_gyration": 4})

            track_lengths = pd.DataFrame(
                tracks.track_lengths, columns=["track_id", "track_length"]
            )

            # time is t from step size, already in table
            r_of_g.drop("t", axis=1, inplace=True)

            track_data_df = pd.concat([track_data_df, msds, step_sizes, r_of_g], axis=1)

            track_data_df = track_data_df.merge(
                track_lengths, how="left", on="track_id"
            )

            merged_data = track_data_df.merge(
                all_fit_data, how="left", on="track_id", indicator="_merge"
            )
            merged_data["MSD_fitted"] = 0
            merged_data["MSD_fitted"].where(
                merged_data["_merge"] != "both", other=1, inplace=True
            )
            merged_data.drop("_merge", axis=1, inplace=True)
            # TODO: confirm right_only does not exist in the column values

            # Add frame start and frame end to the fit results
            frames = merged_data.groupby("track_id", as_index=False).agg(
                frame_start=("frame", "min"), frame_end=("frame", "max")
            )
            merged_data = frames.merge(merged_data, how="right", on="track_id")

            # emit the output data
            out_data = {
                "df": merged_data,
                "summary_data": summary_data,
                "batch": batch,
                "tracks_layer_scale": input_params["tracks_layer_scale"],
                "image_layer_shape": input_params["image_layer_shape"],
                "color_by": state["color_by"],
                "color_by_min_max": state["color_by_min_max"],
            }
        else:
            if track_id in tracks.track_lengths[:, 0]:
                msds = tracks.msd(track_id, fft=True)
                step_sizes = tracks.step_size(track_id)
                r_of_g = tracks.r_of_g(track_id)
                frames = tracks.tracks[tracks.tracks[:, 0] == track_id, 1]

                # Fit for Diffusion coefficient etc
                data = []
                for dim in tracks.dim_columns.keys():
                    if not z_col and dim == "z":
                        continue
                    col = tracks.dim_columns[dim]

                    # there is no track id so reduce column index by 1
                    col -= 1

                    if dim == "sum":
                        d = tracks.dimension
                    else:
                        d = 1

                    D, E, r_squared1 = tracks.fit_msd_linear(
                        t=msds[1:, 0],
                        msd=msds[1:, col],
                        dim=d,
                        max_lagtime=max_lagtime_fit,
                        err=error_term_fit,
                    )
                    K, alpha, r_squared2 = tracks.fit_msd_loglog(
                        t=msds[1:, 0],
                        msd=msds[1:, col],
                        dim=d,
                        max_lagtime=max_lagtime_fit,
                    )

                    data.append(
                        [
                            track_id,
                            frames.min(),
                            frames.max(),
                            dim,
                            D,
                            E,
                            r_squared1,
                            K,
                            alpha,
                            r_squared2,
                        ]
                    )

                data = pd.DataFrame(
                    data,
                    columns=[
                        "track_id",
                        "start",
                        "end",
                        "dim",
                        "D",
                        "E",
                        "r_sq (lin)",
                        "K",
                        "a",
                        "r_sq (log)",
                    ],
                )

                # Radius of gyration, for entire track only
                data["radius_gyration"] = 0
                data["radius_gyration"].where(
                    data["dim"] != "sum",
                    other=tracks.r_of_g(track_id, full=False),
                    inplace=True,
                )

                data = data.round(
                    {
                        "D": 4,
                        "E": 4,
                        "r_sq (lin)": 2,
                        "K": 4,
                        "a": 4,
                        "r_sq (log)": 2,
                        "radius_gyration": 4,
                    }
                )

                summary_data = {
                    "fit_results": data,
                    "msd": msds[1 : max_lagtime_fit + 1, [0, 4]],
                }

                track_data_df = track_data_df[track_data_df["track_id"] == track_id]
                track_data_df.index = range(len(track_data_df))

                msds = pd.DataFrame(msds[:, [0, 4]], columns=["tau", "msd"])
                msds = msds.round({"msd": 4})

                step_sizes = pd.DataFrame(
                    step_sizes[:, [0, 4]], columns=["t", "step_size"]
                )
                step_sizes = step_sizes.round({"step_size": 4})

                r_of_g = pd.DataFrame(r_of_g[:, 1:], columns=["radius_gyration"])
                r_of_g = r_of_g.round({"radius_gyration": 4})

                track_data_df = pd.concat(
                    [track_data_df, msds, step_sizes, r_of_g], axis=1
                )

                out_data = {
                    "df": track_data_df,
                    "summary_data": summary_data,
                    "batch": batch,
                }
            else:
                self.log.emit(f"Track id {track_id} not found.")

        self.update_data.emit(out_data)
        super().run()


class GEMspaAnalyzeWidget(GEMspaWidget):
    """Widget for Analysis plugin"""

    name = "GEMspaAnalyzeWidget"

    def __init__(self, napari_viewer, title="Analyze tracks with GEMspa:"):
        super().__init__(napari_viewer, title)

        self._batch_check = QCheckBox("Process all tracks")
        self._error_term_fit_check = QCheckBox("Fit with error term")
        self._color_by_combo = QComboBox()

        # Add check boxes for how to draw rainbow tracks
        self._color_by_checks = [
            QCheckBox("Track id"),
            QCheckBox("Diffusion coefficient (D)"),
            QCheckBox("Anomalous exponent (alpha)"),
            QCheckBox("Step size (microns)"),
            QCheckBox("Goodness-of-fit for D (R-sq)"),
            QCheckBox("Track length (frames)"),
            QCheckBox("Time (from track start)"),
            QCheckBox("Time (from movie start)"),
        ]
        self._columns_map = {
            "Track id": "track_id",
            "Diffusion coefficient (D)": "D",
            "Anomalous exponent (alpha)": "a",
            "Step size (microns)": "step_size",
            "Goodness-of-fit for D (R-sq)": "r_sq (lin)",
            "Track length (frames)": "track_length",
            "Time (from movie start)": "frame",
            "Time (from track start)": "t",
        }

        self._color_by_min_D_input = QLineEdit("0")
        self._color_by_max_D_input = QLineEdit("2")
        self._color_by_min_alpha_input = QLineEdit("0")
        self._color_by_max_alpha_input = QLineEdit("2")

        self._input_values = {
            "Track id": QLineEdit(""),
            "Microns per px": QLineEdit("0.134"),
            "Time lag (sec)": QLineEdit("0.010"),
            "Min track len for fit (frames)": QLineEdit("11"),
            "Max time lag for fit (frames)": QLineEdit("10"),
        }
        # These must be input by the user
        self._required_inputs = [
            "Microns per px",
            "Time lag (sec)",
            "Min track len for fit (frames)",
            "Max time lag for fit (frames)",
        ]
        self.init_ui()

    def init_ui(self):
        # Set up the input GUI items
        layout = QVBoxLayout()

        grid_layout = QGridLayout()
        grid_layout.setContentsMargins(0, 0, 0, 0)
        i = 0

        if self.title is not None:
            grid_layout.addWidget(QLabel(self.title), i, 0)
            i += 1

        self._batch_check.setChecked(True)
        grid_layout.addWidget(self._batch_check, i, 0, 1, 2)
        i += 1

        for key in self._input_values.keys():
            grid_layout.addWidget(QLabel(key), i, 0)
            grid_layout.addWidget(self._input_values[key], i, 1)
            self._input_values[key].setFixedWidth(50)
            i += 1

        grid_layout.addWidget(self._error_term_fit_check, i, 0, 1, 2)
        i += 1

        grid_layout.addWidget(QLabel("Show plot of tracks colored by:"), i, 0)
        i += 1

        for check_box in self._color_by_checks:
            grid_layout.addWidget(check_box, i, 0, 1, 2)
            if check_box.text() == "Diffusion coefficient (D)":
                grid_layout.addWidget(QLabel("Min"), i, 1)
                grid_layout.addWidget(self._color_by_min_D_input, i, 2)
                self._color_by_min_D_input.setFixedWidth(50)

                grid_layout.addWidget(QLabel("Max"), i, 3)
                grid_layout.addWidget(self._color_by_max_D_input, i, 4)
                self._color_by_max_D_input.setFixedWidth(50)

            if check_box.text() == "Anomalous exponent (alpha)":
                grid_layout.addWidget(QLabel("Min"), i, 1)
                grid_layout.addWidget(self._color_by_min_alpha_input, i, 2)
                self._color_by_min_alpha_input.setFixedWidth(50)

                grid_layout.addWidget(QLabel("Max"), i, 3)
                grid_layout.addWidget(self._color_by_max_alpha_input, i, 4)
                self._color_by_max_alpha_input.setFixedWidth(50)

            i += 1

        self._color_by_checks[0].setChecked(True)
        self._color_by_checks[1].setChecked(True)

        layout.addLayout(grid_layout)
        layout.addStretch()
        self.setLayout(layout)

    def start_task(self, layer_names, log_widget):
        # initialize worker and start task
        self.worker = GEMspaAnalyzeWorker()
        super().start_task(layer_names, log_widget)

    def check_inputs(self):
        # Special case for track id, it is required if batch is not checked
        keys = list(self._input_values.keys())
        required_keys = self._required_inputs[:]
        if self._batch_check.isChecked():
            # ignore track id completely
            keys.remove("Track id")
        else:
            # it is required, if batch is not checked
            required_keys.append("Track id")

        valid = True
        for key in keys:
            text = self._input_values[key].text()
            if key in required_keys or text:
                # if input is not blank, check it is a number
                try:
                    _ = float(text)
                except ValueError:
                    show_error(f"{key} input must be a number")
                    valid = False

        for line_input in [
            self._color_by_min_D_input,
            self._color_by_max_D_input,
            self._color_by_min_alpha_input,
            self._color_by_max_alpha_input,
        ]:
            text = line_input.text()
            if text:
                # if input is not blank, check it is a number
                try:
                    _ = float(text)
                except ValueError:
                    show_error("Min/Max for color by must be a number")
                    valid = False

        return valid

    def state(self, layer_names) -> dict:
        inputs_dict = {
            "tracks_layer_name": layer_names["tracks"],
            "image_layer_name": layer_names["image"],
            "tracks_layer_scale": self.viewer.layers[layer_names["tracks"]].scale,
            "image_layer_shape": self.viewer.layers[layer_names["image"]].data.shape,
            "tracks_layer_data": self.viewer.layers[layer_names["tracks"]].data,
            "tracks_layer_props": self.viewer.layers[layer_names["tracks"]].properties,
        }
        if "labels" in layer_names:
            inputs_dict["labels_layer_name"] = layer_names["labels"]
            inputs_dict["labels_layer_data"] = self.viewer.layers[
                layer_names["labels"]
            ].data

        if self._batch_check.isChecked():
            track_id = None
        else:
            track_id = convert_to_int(self._input_values["Track id"].text())

        color_by = {}
        for check_box in self._color_by_checks:
            color_by[check_box.text()] = check_box.isChecked()

        return {
            "name": self.name,
            "inputs": inputs_dict,
            "parameters": {
                "track_id": track_id,
                "microns_per_pixel": convert_to_float(
                    self._input_values["Microns per px"].text()
                ),
                "time_lag_sec": convert_to_float(
                    self._input_values["Time lag (sec)"].text()
                ),
                "min_len_fit": convert_to_int(
                    self._input_values["Min track len for fit (frames)"].text()
                ),
                "max_lagtime_fit": convert_to_int(
                    self._input_values["Max time lag for fit (frames)"].text()
                ),
                "batch": self._batch_check.isChecked(),
                "error_term_fit": self._error_term_fit_check.isChecked(),
            },
            "color_by": color_by,
            "color_by_min_max": {
                "D": [
                    convert_to_float(self._color_by_min_D_input.text()),
                    convert_to_float(self._color_by_max_D_input.text()),
                ],
                "a": [
                    convert_to_float(self._color_by_min_alpha_input.text()),
                    convert_to_float(self._color_by_max_alpha_input.text()),
                ],
            },
        }

    def update_data(self, out_dict):
        """Set the worker outputs to napari layers"""

        layer = None
        title = ""
        if "df" in out_dict:
            df = out_dict["df"]

            if out_dict["batch"]:
                # Add layer if run in batch mode
                kwargs = {
                    "scale": out_dict["tracks_layer_scale"],
                    "blending": "translucent",
                    "tail_length": int(df["frame"].max()),
                    "name": "Analyzed tracks",
                }

                layer = self._add_napari_layer("tracks", df, **kwargs)
                title = layer.name

                # Show full rainbow tracks plot for each type requested
                img_shape = out_dict["image_layer_shape"][
                    1:
                ]  # assume first dimension of image shape is frame
                aspect_ratio = img_shape[0] / img_shape[1]
                color_by = out_dict["color_by"]
                for key in color_by.keys():
                    if color_by[key]:
                        col = self._columns_map[key]
                        plots_viewer = self._new_plots_viewer(
                            f"{title}: {key}",
                            figsize=(6, 6 * aspect_ratio),
                            close_last=False,
                        )
                        if col in out_dict["color_by_min_max"]:
                            color_range = out_dict["color_by_min_max"][col]
                            print(
                                f"Min/Max for {col} = {color_range[0]}/{color_range[1]}"
                            )
                        else:
                            color_range = None
                        plots_viewer.plot_rainbow_tracks(
                            df, img_shape, col, color_range
                        )
                        plots_viewer.show()
            else:
                title = "Analyzed track"

            if self.display_table_view:
                # Show table of properties (analysis results)
                if out_dict["batch"]:
                    # only show one line for each track if run in batch mode
                    # take last not first when dropping duplicates to pull the r-of-g for full length track
                    # remove the tracks that were filtered for length (fit results are all nan's)
                    df = df.drop_duplicates(subset="track_id", keep="last")
                    df = df[df["MSD_fitted"] == 1]
                    df = df.drop(
                        labels=[
                            "frame",
                            "y",
                            "x",
                            "y (microns)",
                            "x (microns)",
                            "tau",
                            "msd",
                            "t",
                            "step_size",
                            "MSD_fitted",
                        ],
                        axis=1,
                    )
                    if "z" in df.columns:
                        # also drop z if it exists
                        df = df.drop(labels=["z", "z (microns)"], axis=1)

                properties_viewer = self._new_properties_viewer(title, close_last=False)
                properties_viewer.reload_from_pandas(df)
                properties_viewer.show()

            if "summary_data" in out_dict:
                if out_dict["batch"]:
                    plots_viewer = self._new_plots_viewer(title, close_last=False)
                else:
                    plots_viewer = self._new_plots_viewer(
                        title, figsize=(10, 3), close_last=False
                    )

                plots_viewer.plot_analyze_results(out_dict)
                plots_viewer.show()

                if out_dict["batch"]:
                    plots_viewer = self._new_plots_viewer(title, close_last=False)
                    plots_viewer.plot_tracks_info(out_dict)
                    plots_viewer.show()
