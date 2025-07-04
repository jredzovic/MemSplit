# --- imports ---
import napari
import numpy as np
import os
import mrcfile
from napari.layers import Labels
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QRadioButton, QListWidget, QLabel,
    QPushButton, QSpinBox, QDoubleSpinBox, QComboBox,
    QMessageBox, QFileDialog, QFrame, QCheckBox
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QKeyEvent
from skimage import filters, segmentation, measure
from skimage.measure import label as connected_components
import time

# --- main class ---
class LabelPickerWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.label_layer = None
        self.selected_labels = []
        self._undo_stack = []
        self.save_path = None
        self.score_path = None

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Watershed Segmentation Section
        self._add_section_header(layout, "Watershed Segmentation")
        self.browse_score_btn = QPushButton("Browse Score Volume…")
        self.browse_score_btn.clicked.connect(self._browse_score_volume)
        layout.addWidget(self.browse_score_btn)

        self.load_score_checkbox = QCheckBox("Load score volume into viewer")
        self.load_score_checkbox.setChecked(True)
        layout.addWidget(self.load_score_checkbox)

        layout.addWidget(QLabel("Seed Threshold (absolute):"))
        self.seed_threshold_input = QDoubleSpinBox()
        self.seed_threshold_input.setMinimum(0.0)
        self.seed_threshold_input.setSingleStep(0.1)
        self.seed_threshold_input.setValue(2.0)
        layout.addWidget(self.seed_threshold_input)

        self.run_watershed_btn = QPushButton("Run Watershed")
        self.run_watershed_btn.clicked.connect(self._run_watershed_segmentation)
        layout.addWidget(self.run_watershed_btn)

        self._add_separator(layout)

        # Z-Axis Cleaning Section
        self._add_section_header(layout, "Z-Axis Cleaning")
        self.z_start = QSpinBox()
        self.z_stop = QSpinBox()
        self.z_start.setPrefix("Start Z: ")
        self.z_stop.setPrefix("Stop Z: ")
        self.z_start.setMinimum(0)
        self.z_stop.setMinimum(0)
        self.z_start.setMaximum(10000)
        self.z_stop.setMaximum(10000)
        layout.addWidget(self.z_start)
        layout.addWidget(self.z_stop)

        self.clean_btn = QPushButton("Clean Z Range")
        self.clean_btn.clicked.connect(self._clean_z_range)
        layout.addWidget(self.clean_btn)

        self._add_separator(layout)

        # Connected Components Section
        self._add_section_header(layout, "Connected Components")
        self.cc_new_layer_checkbox = QCheckBox("Create New Layer")
        self.cc_new_layer_checkbox.setChecked(True)
        layout.addWidget(self.cc_new_layer_checkbox)

        self.cc_btn = QPushButton("Run Connected Components")
        self.cc_btn.clicked.connect(self._run_connected_components)
        layout.addWidget(self.cc_btn)

        self._add_separator(layout)

        # Label Selector Section
        self._add_section_header(layout, "Label Selector")
        self.toggle_btn = QRadioButton("Label Selector Mode")
        self.toggle_btn.toggled.connect(self._toggle_label_selector)
        layout.addWidget(self.toggle_btn)

        layout.addWidget(QLabel("Picked Labels:"))
        self.label_list = QListWidget()
        self.label_list.setSelectionMode(self.label_list.MultiSelection)
        layout.addWidget(self.label_list)

        self.remove_btn = QPushButton("Remove Selected")
        self.remove_btn.clicked.connect(self._remove_selected_labels)
        layout.addWidget(self.remove_btn)

        layout.addWidget(QLabel("Merge picked labels into:"))
        self.merge_input = QSpinBox()
        self.merge_input.setMinimum(0)
        self.merge_input.setMaximum(500)
        self.merge_input.setValue(0)
        layout.addWidget(self.merge_input)

        self.merge_btn = QPushButton("Merge Labels")
        self.merge_btn.clicked.connect(self._merge_labels)
        layout.addWidget(self.merge_btn)

        self.split_btn = QPushButton("Split Selected Label (CC)")
        self.split_btn.clicked.connect(self._split_selected_label)
        layout.addWidget(self.split_btn)

        self.label_list.keyPressEvent = self._key_press_event_override
        self.setFocusPolicy(Qt.StrongFocus)

        self._add_separator(layout)

        # Save Segmentation Section
        self._add_section_header(layout, "Save Segmentation")

        layout.addWidget(QLabel("Select Label Layer"))
        self.layer_selector = QComboBox()
        layout.addWidget(self.layer_selector)

        self._update_layer_selector()
        viewer.layers.events.inserted.connect(self._update_layer_selector)
        viewer.layers.events.removed.connect(self._update_layer_selector)

        layout.addWidget(QLabel("Voxel Size (Å)"))
        self.voxel_size_input = QDoubleSpinBox()
        self.voxel_size_input.setDecimals(3)
        self.voxel_size_input.setMinimum(0.001)
        self.voxel_size_input.setSingleStep(0.1)
        self.voxel_size_input.setValue(1.0)
        layout.addWidget(self.voxel_size_input)

        self.browse_btn = QPushButton("Browse Output Path…")
        self.browse_btn.clicked.connect(self._browse_output_path)
        layout.addWidget(self.browse_btn)

        self.save_btn = QPushButton("Save Segmentation")
        self.save_btn.clicked.connect(self._save_segmentation)
        layout.addWidget(self.save_btn)

    # --- replaced split method ---
    def _split_selected_label(self):
        if not self.label_layer:
            self.label_layer = self.viewer.layers.selection.active
        if not self.label_layer:
            QMessageBox.warning(self, "No Label Layer", "Please select a label layer.")
            return

        if not self.selected_labels:
            QMessageBox.warning(self, "Selection Error", "Please select at least one label to split.")
            return

        data = self.label_layer.data
        mask = np.isin(data, self.selected_labels)

        if not np.any(mask):
            QMessageBox.warning(self, "Label Missing", "No voxels found for selected labels.")
            return

        components = connected_components(mask, connectivity=1)
        new_data = data.copy()
        start_id = data.max() + 1
        num_components = components.max()

        self._undo_stack.clear()
        self._undo_stack.append(data.copy())

        for i in range(1, num_components + 1):
            new_data[(components == i) & mask] = start_id
            start_id += 1

        new_data[(components == 0) & mask] = 0

        self.label_layer.data = new_data
        self.selected_labels.clear()
        self.label_list.clear()
        print(f"Split labels into {num_components} components.")

    # --- all other methods unchanged ---
    def _add_section_header(self, layout, title):
        header = QLabel(title)
        header.setStyleSheet("font-weight: bold; font-size: 14pt;")
        layout.addWidget(header)

    def _add_separator(self, layout):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("""
            background-color: gray;
            max-height: 1px;
            margin-top: 10px;
            margin-bottom: 10px;
        """)
        layout.addWidget(line)

    def _browse_score_volume(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Score Volume", "", "MRC Files (*.mrc)")
        if file_path:
            self.score_path = file_path.strip('"').strip("'")

    def _run_watershed_segmentation(self):
        if not self.score_path or not os.path.isfile(self.score_path):
            QMessageBox.warning(self, "Missing Score Volume", "Please select a valid score volume file (.mrc).")
            return
        try:
            with mrcfile.open(self.score_path, permissive=True) as mrc:
                scores = mrc.data.copy()
            if scores.ndim != 3:
                QMessageBox.warning(self, "Invalid Data", "Watershed requires a 3D score volume.")
                return
            if self.load_score_checkbox.isChecked():
                self.viewer.add_image(scores, name=os.path.basename(self.score_path))
            start_time = time.time()
            binary_mask = scores > 0
            gradient = filters.sobel(scores)
            threshold = self.seed_threshold_input.value()
            binary_seeds = scores > threshold
            seeds = measure.label(binary_seeds)
            labels = segmentation.watershed(gradient, seeds, mask=binary_mask)
            connected = measure.label(labels)
            elapsed = time.time() - start_time
            print(f"Watershed segmentation completed in {elapsed:.2f} seconds.")
            base_name = os.path.splitext(os.path.basename(self.score_path))[0]
            self.viewer.add_labels(connected, name=f"{base_name}_watershed")
        except Exception as e:
            QMessageBox.critical(self, "Watershed Error", f"An error occurred:\n{e}")

    def _run_connected_components(self):
        self.label_layer = self.viewer.layers.selection.active
        if not self.label_layer:
            QMessageBox.warning(self, "No label layer", "Please select a label layer first.")
            return
        data = self.label_layer.data
        result = connected_components(data, connectivity=1)
        try:
            result = result.astype(np.uint16)
        except ValueError:
            result = result.astype(np.uint32)
        if self.cc_new_layer_checkbox.isChecked():
            self.viewer.add_labels(result, name=f"relabelled_{self.label_layer.name}")
        else:
            self.label_layer.data = result

    def _update_layer_selector(self, event=None):
        self.layer_selector.clear()
        label_layers = [layer.name for layer in self.viewer.layers if isinstance(layer, Labels)]
        self.layer_selector.addItems(label_layers)
        if label_layers and self.layer_selector.currentIndex() == -1:
            self.layer_selector.setCurrentIndex(0)

    def _get_selected_label_layer(self):
        name = self.layer_selector.currentText()
        if name:
            for layer in self.viewer.layers:
                if layer.name == name:
                    return layer
        return self.viewer.layers.selection.active

    def _browse_output_path(self):
        layer = self._get_selected_label_layer()
        default_name = f"{layer.name}_modified.mrc" if layer else "segmentation_saved.mrc"
        file_path, _ = QFileDialog.getSaveFileName(self, "Save As", default_name, "MRC Files (*.mrc)")
        if file_path:
            self.save_path = file_path.strip('"\'')

    def _save_segmentation(self):
        layer = self._get_selected_label_layer()
        if not layer:
            QMessageBox.warning(self, "No Label Layer", "Please select a label layer.")
            return
        if not self.save_path:
            QMessageBox.warning(self, "Missing File Path", "Please choose a file path to save.")
            return
        voxel = self.voxel_size_input.value()
        data = layer.data
        try:
            with mrcfile.new(self.save_path, overwrite=True) as mrc:
                mrc.set_data(data.astype(np.uint8))
                mrc.voxel_size = (voxel, voxel, voxel)
        except Exception:
            try:
                with mrcfile.new(self.save_path, overwrite=True) as mrc:
                    mrc.set_data(data.astype(np.uint16))
                    mrc.voxel_size = (voxel, voxel, voxel)
            except Exception:
                with mrcfile.new(self.save_path, overwrite=True) as mrc:
                    mrc.set_data(data.astype(np.uint32))
                    mrc.voxel_size = (voxel, voxel, voxel)
        QMessageBox.information(self, "Saved", f"Saved segmentation to:\n{self.save_path}")

    def keyPressEvent(self, event: QKeyEvent):
        if (event.key() == Qt.Key_Z) and (event.modifiers() & Qt.ControlModifier):
            self._handle_undo()

    def _handle_undo(self):
        if self.toggle_btn.isChecked() and self.selected_labels:
            last_label = self.selected_labels.pop()
            self.label_list.takeItem(self.label_list.count() - 1)
        elif self._undo_stack and self.label_layer:
            self.label_layer.data = self._undo_stack.pop()

    def _clean_z_range(self):
        self.label_layer = self.viewer.layers.selection.active
        if not self.label_layer:
            QMessageBox.warning(self, "No label layer", "Please select a label layer first.")
            return
        data = self.label_layer.data
        if data.ndim != 3:
            QMessageBox.warning(self, "Unsupported Data", "Z cleaning requires a 3D label layer.")
            return
        z_dim = data.shape[0]
        self.z_start.setMaximum(z_dim)
        self.z_stop.setMaximum(z_dim)
        z_start = self.z_start.value()
        z_stop = self.z_stop.value()
        if z_start >= z_stop or z_stop > z_dim:
            QMessageBox.warning(self, "Invalid Z Range", f"Z range is invalid or out of bounds. Data has {z_dim} slices.")
            return
        cleaned = data.copy()
        cleaned[:z_start, :, :] = 0
        cleaned[z_stop:, :, :] = 0
        self.viewer.add_labels(cleaned, name="cleaned_segmentation")

    def _toggle_label_selector(self, checked):
        if checked:
            self.label_layer = self.viewer.layers.selection.active
            if self.label_layer:
                self.label_layer.mode = 'pick'
            if not hasattr(self, '_mouse_callback_added'):
                self.viewer.mouse_drag_callbacks.append(self._on_click)
                self._mouse_callback_added = True
        else:
            if hasattr(self, '_mouse_callback_added'):
                self.viewer.mouse_drag_callbacks.remove(self._on_click)
                del self._mouse_callback_added

    def _on_click(self, viewer, event):
        if not event.is_dragging and self.label_layer:
            try:
                data = self.label_layer.data
                if viewer.dims.ndisplay == 2:
                    coord = np.round(event.position).astype(int)
                    if data.ndim == 2:
                        y, x = coord[0], coord[1]
                        label_id = int(data[y, x])
                    elif data.ndim == 3:
                        z = viewer.dims.current_step[0]
                        y, x = coord[1], coord[2]
                        label_id = int(data[z, y, x])
                    else:
                        return
                else:
                    value = self.label_layer.get_value(position=event.position, view_direction=viewer.camera.view_direction, dims_displayed=viewer.dims.displayed, world=True)
                    if value is None:
                        return
                    label_id = int(value)
                self.selected_labels.append(label_id)
                self.label_list.addItem(str(label_id))
            except Exception as e:
                print(f"Label picking error: {e}")

    def _remove_selected_labels(self):
        selected_items = self.label_list.selectedItems()
        for item in selected_items:
            label_value = int(item.text())
            if label_value in self.selected_labels:
                self.selected_labels.remove(label_value)
            self.label_list.takeItem(self.label_list.row(item))

    def _key_press_event_override(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self._remove_selected_labels()
        elif (event.key() == Qt.Key_Z) and (event.modifiers() & Qt.ControlModifier):
            self._handle_undo()
        else:
            QListWidget.keyPressEvent(self.label_list, event)

    def _merge_labels(self):
        if not self.selected_labels or not self.label_layer:
            return
        self._undo_stack.clear()
        self._undo_stack.append(self.label_layer.data.copy())
        target_label = self.merge_input.value()
        data = self.label_layer.data
        existing_labels = np.unique(data)
        safe_id = int(existing_labels.max() + 1)
        if target_label != 0 and target_label in existing_labels and target_label not in self.selected_labels:
            data[data == target_label] = safe_id
        for lbl in self.selected_labels:
            if lbl != target_label:
                data[data == lbl] = target_label
        self.selected_labels.clear()
        self.label_list.clear()
        self.label_layer.data = data


# --- launch viewer ---
if __name__ == "__main__":
    viewer = napari.Viewer()
    widget = LabelPickerWidget(viewer)
    viewer.window.add_dock_widget(widget, name="Label Toolkit")
    napari.run()
