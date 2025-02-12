import napari
import numpy as np
import mrcfile
from skimage import filters, segmentation, measure
from skimage.draw import disk
from magicgui import widgets
from qtpy.QtWidgets import (QPushButton, QVBoxLayout, QWidget, QLabel, 
                            QListWidget, QSpinBox, QToolButton)
from qtpy.QtGui import QIcon
from napari.utils.colormaps import DirectLabelColormap
import hashlib


def generate_color_from_id(label_id):
    """Generate consistent color from label ID using hash"""
    hash_obj = hashlib.sha256(str(label_id).encode()).digest()
    return tuple(float((b + 128) % 256) / 255 for b in hash_obj[:3]) + (1,)


def ensure_unique_ids(segmentation):
    """Ensure that each connected component in the segmentation has a unique ID."""
    return measure.label(segmentation, connectivity=1)


class MembraneSegmenter(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.selected_labels = set()
        self.original_colors = {}
        self.merge_id = 1
        self.label_layer = None
        self.brush_size = 1

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        self._create_segmentation_controls()
        self._create_selection_tools()

    def _create_segmentation_controls(self):
        """Create segmentation controls"""
        self.tomogram_path = widgets.FileEdit(label="Tomogram", mode='r', filter='*.mrc')
        self.score_path = widgets.FileEdit(label="Score Volume", mode='r', filter='*.mrc')
        self.threshold_seed = widgets.FloatSlider(label="Seed Threshold", max=11.0, step=0.1)
        self.threshold_mask = widgets.FloatSlider(label="Mask Threshold", min=-6.0, max=5.0, step=0.1)
        self.process_btn = QPushButton("Run Segmentation")
        
        self.layout.addWidget(QLabel("<b>Segmentation Controls</b>"))
        self.layout.addWidget(self.tomogram_path.native)
        self.layout.addWidget(self.score_path.native)
        self.layout.addWidget(self.threshold_seed.native)
        self.layout.addWidget(self.threshold_mask.native)
        self.layout.addWidget(self.process_btn)
        
        self.process_btn.clicked.connect(self.run_segmentation)

    def _create_selection_tools(self):
        """Create selection tools"""
        # Brush size slider
        brush_size_slider = widgets.Slider(label="Brush Size", min=1, max=50, value=self.brush_size)
        brush_size_slider.changed.connect(lambda value: setattr(self, 'brush_size', value))
        self.layout.addWidget(brush_size_slider.native)

        # Custom circular selection tool button
        self.selection_btn = QToolButton()
        self.selection_btn.setCheckable(True)
        self.selection_btn.setText("Circular Select")  # Add text for visibility
        self.selection_btn.setToolTip("Toggle Circular Selection Tool")
        self.selection_btn.clicked.connect(self._toggle_circular_selection)
        self.layout.addWidget(self.selection_btn)

        # Selected labels list
        self.label_list = QListWidget()
        self.label_list.setSelectionMode(QListWidget.MultiSelection)
        self.layout.addWidget(QLabel("Selected Labels:"))
        self.layout.addWidget(self.label_list)

        # Merge controls
        merge_layout = QVBoxLayout()
        
        self.merge_id_spin = QSpinBox()
        self.merge_id_spin.setMinimum(1)
        merge_layout.addWidget(QLabel("Merge ID:"))
        merge_layout.addWidget(self.merge_id_spin)

        merge_button = QPushButton("Merge Labels")
        merge_button.clicked.connect(self._merge_labels)
        
        merge_layout.addWidget(merge_button)
        
        self.layout.addLayout(merge_layout)

    def _toggle_circular_selection(self, active):
        """Toggle circular selection mode"""
        if active:
            print("Circular selection tool enabled.")
            self.viewer.cursor.style = 'circle'
            self.viewer.cursor.size = int(self.brush_size * 2)
            if not hasattr(self, '_circular_callback_added'):
                self.viewer.mouse_drag_callbacks.append(self._on_circular_select)
                self._circular_callback_added = True
        else:
            print("Circular selection tool disabled.")
            self.viewer.cursor.style = 'standard'

    def _on_circular_select(self, viewer, event):
        """Handle circular selection on mouse drag"""
        if not event.is_dragging:
            coord = np.round(event.position).astype(int)
            
            if not (self.label_layer and coord.size >= 2):
                return
            
            current_slice = viewer.dims.current_step[0]
            radius = int(self.brush_size)
            
            try:
                if len(coord) == 3:  # For 3D data
                    y, x = coord[1], coord[2]
                    slice_data = self.label_layer.data[current_slice]
                else:  # For 2D data
                    y, x = coord[0], coord[1]
                    slice_data = self.label_layer.data
                
                rr, cc = disk((y, x), radius, shape=slice_data.shape)
                labels_in_region = np.unique(slice_data[rr, cc])
                
                for label in labels_in_region:
                    if label != 0:
                        self._update_selection(label)
                
            except IndexError:
                pass

    def _update_selection(self, label_id):
        """Update selected labels and colors"""
        if label_id in self.selected_labels:
            self.selected_labels.remove(label_id)
        else:
            self.selected_labels.add(label_id)
            if label_id not in self.original_colors:
                self.original_colors[label_id] = generate_color_from_id(label_id)
        
        color_dict = {0: (0, 0, 0, 0)}
        for lid in np.unique(self.label_layer.data):
            if lid == 0:
                continue
            color_dict[lid] = (0, 1, 0, 1) if lid in self.selected_labels else \
                              self.original_colors.get(lid, generate_color_from_id(lid))
        
        self.label_layer.colormap = DirectLabelColormap(color_dict=color_dict)
        self.label_list.clear()
        self.label_list.addItems(map(str, sorted(self.selected_labels)))

    def _merge_labels(self):
        """Merge selected labels into a new ID"""
        if not (self.selected_labels and self.label_layer):
            return
        
        new_id = int(self.merge_id_spin.value())
        mask = np.isin(self.label_layer.data, list(self.selected_labels))
        self.label_layer.data[mask] = new_id
        
        new_color = generate_color_from_id(new_id)
        for label in list(self.selected_labels):
            if label != new_id:
                del self.original_colors[label]
        
        self.original_colors[new_id] = new_color
        self.selected_labels.clear()
        self._update_colormap()
        
    def _update_colormap(self):
        """Refresh colormap after merging or selection changes"""
        all_labels = np.unique(self.label_layer.data)
        color_dict = {0: (0, 0, 0, 0)}
        
        for lid in all_labels:
            if lid > 0:
                color_dict[lid] = self.original_colors.get(lid, generate_color_from_id(lid))
        
        self.label_layer.colormap = DirectLabelColormap(color_dict=color_dict)

    def run_segmentation(self):
        """Run segmentation pipeline"""
        if self.tomogram_path.value and self.score_path.value:
            try:
                with mrcfile.open(self.tomogram_path.value) as mrc:
                    tomogram = mrc.data.astype(np.float32)
                with mrcfile.open(self.score_path.value) as mrc:
                    scores = mrc.data.astype(np.float32)

                gradient = filters.sobel(scores)
                binary_image = scores > self.threshold_mask.value
                binary_seeds = scores > self.threshold_seed.value
                
                seeds = measure.label(binary_seeds.astype(np.uint32))
                labels = segmentation.watershed(gradient, seeds, mask=binary_image)
                
                result = ensure_unique_ids(labels).astype(np.uint32)

                if self.label_layer:
                    self.label_layer.data = result
                else:
                    all_labels = np.unique(result)
                    color_dict = {0: (0, 0, 0, 0)}
                    self.original_colors = {}
                    
                    for lid in all_labels:
                        if lid > 0:
                            color = generate_color_from_id(lid)
                            color_dict[lid] = color
                            self.original_colors[lid] = color
                    
                    self.label_layer = self.viewer.add_labels(
                        result,
                        name='Segmentation',
                        colormap=DirectLabelColormap(color_dict=color_dict),
                    )
                    self.label_layer.mode = "pick"

                self.selected_labels.clear()
                self.label_list.clear()

            except Exception as e:
                print(f"Segmentation error: {e}")

if __name__ == "__main__":
    viewer = napari.Viewer()
    segmenter = MembraneSegmenter(viewer=viewer)
    viewer.window.add_dock_widget(segmenter, name="Membrane Toolkit")
    napari.run()
