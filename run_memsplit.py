# run_memsplit.py
import napari

from widget import LabelPickerWidget

if __name__ == "__main__":
    viewer = napari.Viewer()
    widget = LabelPickerWidget(viewer)
    viewer.window.add_dock_widget(widget, name="Label Toolkit")
    napari.run()

