# -*- coding: utf-8 -*-
# @Author  : LG

import os.path
from PyQt5 import QtCore, QtWidgets
from ISAT.widgets.plugin_base import PluginBase
from ISAT_plugin_auto_annotate.yolo import DetectModel
from ISAT.widgets.polygon import Rect


class AutoAnnotatePlugin(PluginBase):
    def __init__(self):
        super().__init__()

    def init_plugin(self, mainwindow):
        self.mainwindow = mainwindow
        self.detector = None
        self.category_dict = {}
        self.init_ui()

    def enable_plugin(self):
        self.mainwindow.change_contour_mode(contour_mode='max_only')
        self.mainwindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.dock)
        self.dock.show()
        self.enabled = True

    def disable_plugin(self):
        self.mainwindow.removeDockWidget(self.dock)
        self.enabled = False

    def get_plugin_author(self) -> str:
        return "yatengLG"

    def get_plugin_version(self) -> str:
        return "1.0.0"

    def get_plugin_description(self) -> str:
        return "Auto annotation by ultralytics."

    def init_ui(self):
        self.dock = QtWidgets.QDockWidget(self.mainwindow)
        self.dock.setWindowTitle('Auto annotation by ultralytics')
        main_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        widget = QtWidgets.QWidget()
        widget.setMaximumHeight(36)
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(widget)

        self.checkpoint_edit = QtWidgets.QLineEdit()
        self.checkpoint_edit.setPlaceholderText('yolo onnx model.')
        self.checkpoint_edit.setReadOnly(True)
        self.checkpoint_button = QtWidgets.QPushButton('open')
        self.checkpoint_button.clicked.connect(self.load_detector)
        layout.addWidget(self.checkpoint_edit)
        layout.addWidget(self.checkpoint_button)

        widget = QtWidgets.QWidget()
        widget.setMaximumHeight(36)
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(widget)

        self.category_edit = QtWidgets.QLineEdit()
        self.category_edit.setPlaceholderText('category csv.')
        self.category_edit.setReadOnly(True)
        self.category_button = QtWidgets.QPushButton('open')
        self.category_button.clicked.connect(self.load_category)
        layout.addWidget(self.category_edit)
        layout.addWidget(self.category_button)

        self.result_table = QtWidgets.QTableWidget()
        self.result_table.setColumnCount(6)
        self.result_table.horizontalHeader().setSectionResizeMode(5, QtWidgets.QHeaderView.Stretch)

        self.result_table.setHorizontalHeaderLabels(['xmin', 'xmax', 'ymin', 'ymax', 'score', 'category'])
        self.result_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.result_table.setMinimumHeight(150)
        main_layout.addWidget(self.result_table)

        self.processbar = QtWidgets.QProgressBar()
        self.processbar.setMaximum(100)
        self.processbar.setValue(0)
        main_layout.addWidget(self.processbar)

        self.dock.setWidget(main_widget)
        self.mainwindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.dock)

        if not self.enabled:
            self.disable_plugin()

    def load_detector(self):
        filter = "onnx (*.onnx)"
        path, suffix = QtWidgets.QFileDialog.getOpenFileName(self.mainwindow,
                                                             directory=os.path.dirname(os.path.abspath(__file__)),
                                                             caption='Yolo onnx model file.',
                                                             filter=filter)
        if path:
            try:
                self.detector = DetectModel(path, 0.25)
                self.checkpoint_edit.setText(os.path.split(path)[-1])
            except Exception as e:
                print(e)

    def load_category(self):
        filter = "csv (*.csv)"
        path, suffix = QtWidgets.QFileDialog.getOpenFileName(self.mainwindow,
                                                             directory=os.path.dirname(os.path.abspath(__file__)),
                                                             caption='Category csv file.',
                                                             filter=filter)
        if path:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for index, line in enumerate(lines):
                        self.category_dict[index] = line.strip('\r\n')

                self.category_edit.setText(os.path.split(path)[-1])

            except Exception as e:
                print(e)

    def after_sam_encode_finished_event(self, index):
        if self.detector is None:
            return

        if self.mainwindow.use_segment_anything:
            if self.mainwindow.current_index == index:
                self.auto_annotate()

    def auto_annotate(self):
        if self.detector is None:
            return

        self.mainwindow.scene.accept_mouse_events = False
        self.checkpoint_button.setEnabled(False)
        self.result_table.setRowCount(0)
        self.processbar.setValue(0)

        try:
            current_image = os.path.join(self.mainwindow.image_root, self.mainwindow.files_list[self.mainwindow.current_index])
            detect_result = self.detector(current_image) # [n, 6]

            self.processbar.setMaximum(len(detect_result))

            for i, result in enumerate(detect_result):
                xmin, ymin, xmax, ymax, score, category_index = result
                category_index = round(category_index)
                category = str(self.category_dict.get(category_index, category_index))

                self.mainwindow.scene.start_segment_anything_box()
                self.mainwindow.scene.current_sam_rect = Rect()
                self.mainwindow.scene.addItem(self.mainwindow.scene.current_sam_rect)
                self.mainwindow.scene.current_sam_rect.addPoint(QtCore.QPointF(xmin, ymin))
                self.mainwindow.scene.current_sam_rect.addPoint(QtCore.QPointF(xmax, ymax))
                self.mainwindow.scene.update_mask()
                self.mainwindow.current_category = category
                self.mainwindow.scene.finish_draw()


                self.result_table.insertRow(self.result_table.rowCount())
                self.result_table.setItem(i, 0, QtWidgets.QTableWidgetItem('{:4.0f}'.format(xmin)))
                self.result_table.setItem(i, 1, QtWidgets.QTableWidgetItem('{:4.0f}'.format(ymin)))
                self.result_table.setItem(i, 2, QtWidgets.QTableWidgetItem('{:4.0f}'.format(xmax)))
                self.result_table.setItem(i, 3, QtWidgets.QTableWidgetItem('{:4.0f}'.format(ymax)))
                self.result_table.setItem(i, 4, QtWidgets.QTableWidgetItem('{:.2f}'.format(score)))
                self.result_table.setItem(i, 5, QtWidgets.QTableWidgetItem('{}'.format(category)))
                self.processbar.setValue(i+1)
                QtCore.QThread.msleep(100)

        except Exception as e:
            print(e)
        finally:
            self.mainwindow.scene.accept_mouse_events = True
            self.checkpoint_button.setEnabled(True)
