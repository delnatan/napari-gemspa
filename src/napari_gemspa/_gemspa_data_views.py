"""Defines: GEMspaPlottingWindow, GEMspaTableWidget, GEMspaTableWindow"""

import matplotlib as mpl
import napari
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from qtpy.QtCore import QCoreApplication, QRect, Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QAction,
    QApplication,
    QFileDialog,
    QMainWindow,
    QMenu,
    QMenuBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class GEMspaPlottingWindow(QMainWindow):
    """Main window for showing plots"""

    def __init__(self, napari_viewer, figsize=(8, 3), *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.viewer = napari_viewer

        self.centralWidget = QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.verticalLayout = QVBoxLayout(self.centralWidget)

        self.canvas = FigureCanvas(Figure(figsize=figsize))

        # Ideally one would use self.addToolBar here, but it is slightly
        # incompatible between PyQt6 and other bindings, so we just add the
        # toolbar as a plain widget instead.
        self.verticalLayout.addWidget(NavigationToolbar(self.canvas, self))
        self.verticalLayout.addWidget(self.canvas)

    @staticmethod
    def _fill_table(t, df):
        t.clearContents()
        t.setColumnCount(len(df.columns))
        t.setHorizontalHeaderLabels(df.columns)
        t.setRowCount(len(df))

        for i, row in enumerate(df.iterrows()):
            for j, col in enumerate(df.columns):
                val = row[1][col]
                t.setItem(i, j, QTableWidgetItem(str(val)))

    def plot_locate_results(self, df):
        self.canvas.figure.clear()
        axs = self.canvas.figure.subplots(1, 3)

        # Show Mass histogram
        axs[0].hist(df["mass"], bins=20)
        axs[0].set(xlabel="mass", ylabel="count")
        axs[0].set_title("mass")

        # Show subpixel bias (histogram fractional part of x/y positions)
        axs[1].hist(np.modf(df["x"])[0], bins=20)
        axs[1].set(xlabel="x", ylabel="count")
        axs[1].set_title("sub px bias (x)")

        axs[2].hist(np.modf(df["y"])[0], bins=20)
        axs[2].set(xlabel="y", ylabel="count")
        axs[2].set_title("sub px bias (y)")

        self.canvas.figure.tight_layout()

    def plot_link_results(self, df):
        self.canvas.figure.clear()
        axs = self.canvas.figure.subplots(1, 3)

        # Show plots of mass vs. size and mass vs. eccentricity
        mean_t = df.groupby("track_id").mean()

        axs[0].plot(mean_t["mass"], mean_t["size"], "ko", alpha=0.3)
        axs[0].set(xlabel="mass", ylabel="size")
        axs[0].set_title("mass vs size")

        axs[1].plot(mean_t["mass"], mean_t["ecc"], "ko", alpha=0.3)
        axs[1].set(xlabel="mass", ylabel="eccentricity (0=circular)")
        axs[1].set_title("mass vs eccentricity")

        # Show track length histogram
        track_lengths = df["track_id"].value_counts(sort=False)

        axs[2].hist(track_lengths, bins=30)
        axs[2].set(xlabel="track length", ylabel="counts")
        axs[2].set_title("Track lengths histogram")

        self.canvas.figure.tight_layout()

    def plot_rainbow_tracks(
        self, df, img_shape=None, color_by="track_id", color_range=None
    ):
        self.canvas.figure.clear()
        ax = self.canvas.figure.subplots(1, 1)

        if color_range is not None:
            min_val = color_range[0]
            max_val = color_range[1]

            if max_val < min_val:
                print("Rainbow tracks: Max/Min are invalid, using defaults...")
                color_range = None

        if color_range is None:
            min_val = 0
            if color_by in ["step_size", "frame", "track_length"]:
                max_val = df[color_by].max()
            elif color_by == "D":
                max_val = 2
            elif color_by == "a":
                max_val = 2
            elif color_by == "r_sq (lin)":
                max_val = 1

        if img_shape is not None:
            max_y = img_shape[0]
        else:
            max_y = df["y"].max()

        def get_color(v):
            if v < min_val:
                v = min_val
            if v > max_val:
                v = max_val
            return (v - min_val) / (max_val - min_val)

        # Plot all tracks
        for group in df.groupby("track_id"):
            track_data = group[1]
            if color_by == "track_id":
                ax.plot(track_data["x"], track_data["y"], "-")
            elif color_by in ["D", "a", "r_sq (lin)", "track_length"]:
                val = track_data.iloc[0][color_by]
                if not np.isnan(val):
                    ax.plot(
                        track_data["x"],
                        track_data["y"],
                        "-",
                        color=mpl.cm.jet(get_color(val)),
                    )
            elif color_by in ["step_size", "frame"]:
                for step_i in range(1, len(track_data), 1):
                    val = track_data.iloc[step_i][color_by]
                    if not np.isnan(val):
                        ax.plot(
                            [
                                track_data.iloc[step_i - 1]["x"],
                                track_data.iloc[step_i]["x"],
                            ],
                            [
                                track_data.iloc[step_i - 1]["y"],
                                track_data.iloc[step_i]["y"],
                            ],
                            "-",
                            color=mpl.cm.jet(get_color(val)),
                        )
            elif color_by == "t":
                max_val = len(track_data)
                for step_i in range(1, max_val, 1):
                    show_color = step_i / max_val
                    ax.plot(
                        [
                            track_data.iloc[step_i - 1]["x"],
                            track_data.iloc[step_i]["x"],
                        ],
                        [
                            track_data.iloc[step_i - 1]["y"],
                            track_data.iloc[step_i]["y"],
                        ],
                        "-",
                        color=mpl.cm.jet(show_color),
                    )
            else:
                print("Rainbow tracks: Unknown color_by option, using track_id...")
                ax.plot(group[1]["x"], group[1]["y"], "-")

        if img_shape is not None and len(img_shape) > 1:
            ax.set_xlim(0, img_shape[1])
            ax.set_ylim(0, img_shape[0])

        ax.set_ylim(max_y, 0)

        if color_by != "track_id":
            if color_by == "t":
                norm = mpl.colors.Normalize(vmin=0, vmax=1)
                cmap = mpl.cm.ScalarMappable(norm=norm, cmap="jet")
                cbar = self.canvas.figure.colorbar(cmap, ax=ax)
                cbar.set_ticks([0, 1])
                cbar.set_ticklabels(["Start", "End"])
            else:
                norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
                cmap = mpl.cm.ScalarMappable(norm=norm, cmap="jet")
                self.canvas.figure.colorbar(cmap, ax=ax)

        self.canvas.figure.tight_layout()

    def plot_tracks_info(self, out_data):
        df = out_data["df"]

        self.canvas.figure.clear()
        axs = self.canvas.figure.subplots(1, 3)

        # Show track length histogram
        track_lengths = df["track_id"].value_counts(sort=False)

        axs[0].hist(track_lengths, bins=30)
        axs[0].set(xlabel="track length", ylabel="counts")
        axs[0].set_title("Track lengths (# frames)")

        # Show r of g histogram - only final r of g
        df = df.drop_duplicates(subset="track_id", keep="last")
        rg = df["radius_gyration"]

        axs[1].hist(rg, bins=30)
        axs[1].set(xlabel="radius", ylabel="counts")
        axs[1].set_title("Radius of gyration (microns)")

        # Scatter
        axs[2].plot(track_lengths, rg, "ko", alpha=0.3)
        axs[2].set(xlabel="Track length", ylabel="Radius of gyration")
        axs[2].set_title("Scatter")

        self.canvas.figure.tight_layout()

    def plot_analyze_results(self, out_data):
        summary_table = QTableWidget()
        self.verticalLayout.addWidget(summary_table)

        if (
            "ens_fit_results" in out_data["summary_data"]
            and "ens_msd" in out_data["summary_data"]
        ):
            plot_data = out_data["summary_data"]

            msd = plot_data["ens_msd"]
            df = plot_data["ens_fit_results"]

            # Table of data
            self._fill_table(summary_table, df)

            D = df.iloc[0]["D"]
            alpha = df.iloc[0]["a"]

            # Ensemble MSD plot
            self.canvas.figure.clear()
            axs = self.canvas.figure.subplots(1, 2)
            axs[0].scatter(msd[:, 0], msd[:, 1])
            axs[1].scatter(msd[:, 0], msd[:, 1])
            axs[1].set_xscale("log", base=10)
            axs[1].set_yscale("log", base=10)
            axs[0].set(xlabel=r"$\tau$ $(s)$", ylabel=r"$MSD$ ($\mu m^{2}$)")
            axs[1].set(
                xlabel=r"$log_{10}$ $\tau$ $(s)$",
                ylabel=r"$log_{10}$ $MSD$ ($\mu m^{2}$)",
            )
            axs[0].set_title(f"ens-avg MSD (2d)\nD = {D} " + r"$\mu m^{2}$/s")
            axs[1].set_title("ens-avg log-log MSD (2d)\n" + r"$\alpha$ = " + f"{alpha}")

        elif (
            "fit_results" in out_data["summary_data"]
            and "msd" in out_data["summary_data"]
        ):
            plot_data = out_data["summary_data"]
            df = out_data["df"]
            msd = plot_data["msd"]
            summary_df = plot_data["fit_results"]

            # Table of data
            self._fill_table(summary_table, summary_df)

            D = summary_df[summary_df.dim == "sum"].iloc[0]["D"]
            alpha = summary_df[summary_df.dim == "sum"].iloc[0]["a"]
            track_id = int(summary_df.iloc[0]["track_id"])

            # Make plots (MSD of track, with fit line, linear and loglog scale)
            self.canvas.figure.clear()
            axs = self.canvas.figure.subplots(1, 4)
            axs[0].scatter(msd[:, 0], msd[:, 1])
            axs[1].scatter(msd[:, 0], msd[:, 1])
            axs[1].set_xscale("log", base=10)
            axs[1].set_yscale("log", base=10)
            axs[0].set(xlabel=r"$\tau$ $(s)$", ylabel=r"$MSD$ ($\mu m^{2}$)")
            axs[1].set(
                xlabel=r"$log_{10}$ $\tau$ $(s)$",
                ylabel=r"$log_{10}$ $MSD$ ($\mu m^{2}$)",
            )
            axs[0].set_title(f"track {track_id} MSD (2d)\nD = {D} " + r"$\mu m^{2}$/s")
            axs[1].set_title(
                f"track {track_id} log-log MSD (2d)\n" + r"$\alpha$ = " + f"{alpha}"
            )

            # Make plot of radius of gyration...
            axs[2].plot(df["t"], df["radius_gyration"])
            axs[2].set_title(f"track {track_id}")
            axs[2].set(xlabel="t (s)", ylabel="radius gyration " + r"($\mu m$)")

            # Make plot of the track itself...
            x_values = df["x (microns)"].to_numpy()
            y_values = df["y (microns)"].to_numpy()
            x_min = np.min(x_values)
            y_min = np.min(y_values)

            axs[3].plot(x_values - x_min, y_values - y_min)
            axs[3].plot(
                x_values[0] - x_min,
                y_values[0] - y_min,
                ".",
                color="green",
            )
            axs[3].plot(
                x_values[-1] - x_min,
                y_values[-1] - y_min,
                ".",
                color="red",
            )
            axs[3].set_title(f"track {track_id}")
            axs[3].set(xlabel=r"x ($\mu m$)", ylabel=r"y ($\mu m$)")

        self.canvas.figure.tight_layout()


class GEMspaTableWidget(QTableWidget):
    """
    this class extends QTableWidget
    * supports copying multiple cell's text onto the clipboard
    * formatted specifically to work with multiple-cell paste into programs
      like google sheets, excel, or numbers
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def copy_selection_to_clipboard(self, sep="\t"):
        copied_cells = sorted(self.selectedIndexes())
        if len(copied_cells) > 0:
            copy_text = ""
            max_column = copied_cells[-1].column()
            for c in copied_cells:
                copy_text += self.item(c.row(), c.column()).text()
                if c.column() == max_column:
                    copy_text += "\n"
                else:
                    copy_text += sep

            QApplication.clipboard().setText(copy_text)

    def save_to_file(self, fname):
        nrows = self.rowCount()
        ncols = self.columnCount()
        headers = []
        for j in range(ncols):
            headers.append(self.horizontalHeaderItem(j).text())

        full_data = []
        for i in range(nrows):
            data = []
            for j in range(ncols):
                data.append(self.item(i, j).text())
            full_data.append(data)

        df = pd.DataFrame(full_data, columns=headers)
        df.to_csv(fname, sep="\t", index=False)

    def select_all(self):
        self.selectAll()

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        if event.key() == Qt.Key_C and (
            QApplication.keyboardModifiers() & Qt.ControlModifier
        ):
            self.copy_selection_to_clipboard()


class GEMspaTableWindow(QMainWindow):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.setGeometry(QRect(100, 100, 600, 200))

        self.centralWidget = QWidget(self)
        self.verticalLayout = QVBoxLayout(self.centralWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.tableWidget = GEMspaTableWidget(self.centralWidget)
        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.verticalLayout.addWidget(self.tableWidget)
        self.setCentralWidget(self.centralWidget)

        self.menubar = QMenuBar(self)
        self.menubar.setObjectName("menubar")
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuEdit = QMenu(self.menubar)
        self.menuEdit.setObjectName("menuEdit")
        self.actionSave = QAction(self)
        self.actionSave.setObjectName("actionSave")
        self.actionSave_As = QAction(self)
        self.actionSave_As.setObjectName("actionSave_As")
        self.actionClose = QAction(self)
        self.actionClose.setObjectName("actionClose")
        self.actionCopy = QAction(self)
        self.actionCopy.setObjectName("actionCopy")
        self.actionSelect_All = QAction(self)
        self.actionSelect_All.setObjectName("actionSelect_All")
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addAction(self.actionSave_As)
        self.menuFile.addAction(self.actionClose)
        self.menuEdit.addAction(self.actionCopy)
        self.menuEdit.addAction(self.actionSelect_All)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())
        self.setMenuBar(self.menubar)

        self.retranslateUi()
        self.connectSignalsSlots()

    def retranslateUi(self):
        _translate = QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "MainWindow"))

        self.menuFile.setTitle(_translate("MainWindow", "&File"))
        self.menuEdit.setTitle(_translate("MainWindow", "Edit"))
        self.actionCopy.setText(_translate("MainWindow", "&Copy"))
        self.actionCopy.setToolTip(_translate("MainWindow", "Copy selected text"))
        self.actionCopy.setShortcut(_translate("MainWindow", "Ctrl+C"))
        self.actionSelect_All.setText(_translate("MainWindow", "Select &All"))
        self.actionSelect_All.setShortcut(_translate("MainWindow", "Ctrl+A"))
        self.actionSave.setText(_translate("MainWindow", "&Save"))
        self.actionSave.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.actionSave_As.setText(_translate("MainWindow", "Save As..."))
        self.actionClose.setText(_translate("MainWindow", "Close"))
        self.actionClose.setShortcut(_translate("MainWindow", "Ctrl+W"))

    def connectSignalsSlots(self):
        self.actionCopy.triggered.connect(self.copy)
        self.actionSelect_All.triggered.connect(self.select_all)
        self.actionSave.triggered.connect(self.save)
        self.actionSave_As.triggered.connect(self.save_as)
        self.actionClose.triggered.connect(self.close)

    def copy(self):
        self.tableWidget.copy_selection_to_clipboard()

    def select_all(self):
        self.tableWidget.select_all()

    def save(self):
        pass

    def save_as(self):
        file_name = QFileDialog.getSaveFileName(
            self, "Save file...", "", "Text, tab-delimited (*.txt)"
        )
        print(f"Saving {file_name}...")
        self.tableWidget.save_to_file(file_name[0])

    def reload_from_pandas(self, df):
        """Fill the table with the data in a pandas data frame"""
        self.tableWidget.clearContents()
        self.tableWidget.setColumnCount(len(df.columns))
        self.tableWidget.setHorizontalHeaderLabels(df.columns)
        self.tableWidget.setRowCount(len(df))

        for i, row in enumerate(df.iterrows()):
            for j, col in enumerate(df.columns):
                self.tableWidget.setItem(i, j, QTableWidgetItem(str(row[1][col])))

    def reload_from_layer(self, layer_name):
        """Reload the properties from the layer to the table widget"""

        data = self.viewer.layers[layer_name].data
        properties = self.viewer.layers[layer_name].properties
        headers = []
        if isinstance(
            self.viewer.layers[layer_name], napari.layers.points.points.Points
        ):
            if data.shape[1] == 3:
                headers = ["frame", "y", "x"]
            elif data.shape[1] == 4:
                headers = ["frame", "z", "y", "x"]

        elif isinstance(
            self.viewer.layers[layer_name], napari.layers.tracks.tracks.Tracks
        ):
            if data.shape[1] == 4:
                headers = ["track_id", "frame", "y", "x"]
            elif data.shape[1] == 5:
                headers = ["track_id", "frame", "z", "y", "x"]

            # Avoid repeated column: napari adds track_id column to properties
            del properties["track_id"]

        for key in properties:
            headers.append(key)

        self.tableWidget.setColumnCount(len(headers))
        self.tableWidget.setHorizontalHeaderLabels(headers)
        self.tableWidget.setRowCount(data.shape[0])

        for line in range(data.shape[0]):
            for col in range(data.shape[1]):
                self.tableWidget.setItem(
                    line, col, QTableWidgetItem(str(data[line, col]))
                )

        # properties
        col = data.shape[1]
        for key in properties:
            prop = properties[key]
            for line in range(len(prop)):
                self.tableWidget.setItem(line, col, QTableWidgetItem(str(prop[line])))
            col += 1
