
import pandas as pd
import numpy as np
from qtpy.QtCore import Qt, QCoreApplication, QRect
from qtpy.QtWidgets import (QWidget, QApplication, QVBoxLayout,
                            QTableWidget, QAbstractItemView, QTableWidgetItem,
                            QMainWindow, QMenuBar, QMenu, QAction, QFileDialog)

from matplotlib.backends.backend_qtagg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure


"""Defines: GEMspaPlottingWindow, GEMspaTableWidget, GEMspaTableWindow"""


class GEMspaPlottingWindow(QMainWindow):
    """Main window for showing plots"""

    def __init__(self, napari_viewer, figsize=(5, 3), *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.viewer = napari_viewer
        self.layer_name = ''

        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.verticalLayout = QVBoxLayout(self.centralWidget)

        self.canvas = FigureCanvas(Figure(figsize=figsize))

        # Ideally one would use self.addToolBar here, but it is slightly
        # incompatible between PyQt6 and other bindings, so we just add the
        # toolbar as a plain widget instead.
        self.verticalLayout.addWidget(NavigationToolbar(self.canvas, self))
        self.verticalLayout.addWidget(self.canvas)

    def reload(self):
        """Reload the properties from the layer to the figure"""

        data = self.viewer.layers[self.layer_name].data
        properties = self.viewer.layers[self.layer_name].properties
        type = self.viewer.layers[self.layer_name].as_layer_data_tuple()[2]  # TODO: is there a better way to do this?

        if type == 'points':
            if data.shape[1] == 3:
                y_index = 1
                x_index = 2
                # headers = ['t', 'y', 'x']
            elif data.shape[1] == 4:
                y_index = 2
                x_index = 3
                # headers = ['t', 'z', 'y', 'x']

            # TODO: implement for other types (only for Points layer right now)
            self.canvas.figure.clear()
            self._axs = self.canvas.figure.subplots(1, 3)

            # Show Mass histogram
            self._axs[0].hist(properties['mass'], bins=20)
            self._axs[0].set(xlabel='mass', ylabel='count')
            self._axs[0].set_title('mass')

            # Show subpixel bias (histogram fractional part of x/y positions)
            self._axs[1].hist(np.modf(data[:, x_index])[0], bins=20)
            self._axs[1].set(xlabel='x', ylabel='count')
            self._axs[1].set_title('sub px bias (x)')

            self._axs[2].hist(np.modf(data[:, y_index])[0], bins=20)
            self._axs[2].set(xlabel='y', ylabel='count')
            self._axs[2].set_title('sub px bias (y)')

        elif type == 'tracks':
            pass

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

    def copy_selection_to_clipboard(self, sep='\t'):
        copied_cells = sorted(self.selectedIndexes())
        if(len(copied_cells) > 0):

            copy_text = ''
            max_column = copied_cells[-1].column()
            for c in copied_cells:
                copy_text += self.item(c.row(), c.column()).text()
                if c.column() == max_column:
                    copy_text += '\n'
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
        df.to_csv(fname, sep='\t', index=False)

    def select_all(self):
        self.selectAll()

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        if event.key() == Qt.Key_C and (QApplication.keyboardModifiers() & Qt.ControlModifier):
            self.copy_selection_to_clipboard()


class GEMspaTableWindow(QMainWindow):

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.layer_name = ''

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
        file_name = QFileDialog.getSaveFileName(self, "Save file...", "", "Text, tab-delimited (*.txt)")
        print(f"Saving {file_name}...")
        self.tableWidget.save_to_file(file_name[0])

    def reload(self):
        """Reload the properties from the layer to the table widget"""

        data = self.viewer.layers[self.layer_name].data
        properties = self.viewer.layers[self.layer_name].properties
        type = self.viewer.layers[self.layer_name].as_layer_data_tuple()[2]  # TODO: is there a better way to do this?

        if type == 'points':

            # Assume layer is Points layer, TODO: add tracks layer
            # Other layer types not supported

            headers = []
            if data.shape[1] == 3:
                headers = ['t', 'y', 'x']
            elif data.shape[1] == 4:
                headers = ['t', 'z', 'y', 'x']

            for key in properties:
                headers.append(key)

            self.tableWidget.setColumnCount(len(headers))
            self.tableWidget.setHorizontalHeaderLabels(headers)
            self.tableWidget.setRowCount(data.shape[0])

            for line in range(data.shape[0]):
                for col in range(data.shape[1]):
                    self.tableWidget.setItem(line, col, QTableWidgetItem(str(data[line, col])))

            # properties
            col = data.shape[1]
            for key in properties:
                prop = properties[key]
                for line in range(len(prop)):
                    self.tableWidget.setItem(line, col, QTableWidgetItem(str(prop[line])))
                col += 1

        elif type == 'tracks':
            pass