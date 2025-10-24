# Memsplit — napari plugin

Automated classification and post‑processing of **semantic membrane segmentations** from probability/score volumes (e.g. outputs from [MemBrain‑seg](https://github.com/teamtomo/membrain-seg)).

---

## ✨ Features

* **Watershed from score volume** (.mrc) using Sobel gradient + seed thresholding; optionally load the score volume into the viewer.
* **Z‑axis artifact cleanup** by zeroing all labels outside a chosen slice range.
* **Connected components** (relabeling) with option to write to a new or the same layer.
* **Interactive label picking** ("Label Selector Mode") to collect label IDs by clicking in the viewer.
* **Merge picked labels** into a target label ID or background (0), with safe conflict handling.
* **Split selected labels** by connected components (useful to separate touching instances).
* **Fast save to .mrc** with user‑specified **voxel size (Å)**; automatic dtype fallback (uint8→uint16→uint32).
* **Undo** (Ctrl/⌘+Z) for picks and data edits where available.

> Designed for 2D/3D label layers; optimized for 3D tomograms.

---

## 🧩 Inputs & outputs

* **Inputs:**

  * 3D **score/probability volume** (MRC format), e.g. from MemBrain‑seg.
  * Existing **Labels** layers in napari.
* **Outputs:**

  * New/modified **Labels** layers in napari.
  * **.mrc** files with label data and voxel size written into header.

> Tip: Use strictly positive scores for the foreground mask. Negative or zero scores are treated as background during watershed.

---

## 🔧 Installation

Memsplit is a standard napari dock widget.

### Option A — Install from source (developer mode)

```bash
# inside a fresh environment with Python ≥3.9
pip install -U pip
pip install -e .
```

This expects a `pyproject.toml`/`setup.cfg` with napari plugin entry points. If you’re working from a single script, see **Option B**.

### Option B — Run the widget script directly

If you have `circular_select.py` in your project:

```bash
python circular_select.py
```

This will open napari and add the dock widget **“Label Toolkit”**.

Dependencies (installed automatically if packaged): `napari`, `numpy`, `mrcfile`, `scikit-image`, `qtpy`.

---

## 🚀 Quick start (GUI)

1. **Open napari** and ensure your score/segmentation volumes are available.
2. From the **Memsplit / Label Toolkit** dock widget:

   ### A) Watershed from score volume

   1. Click **Browse Score Volume…** and select a 3D `.mrc` score/probability map.
   2. (Optional) Keep **Load score volume into viewer** checked to visualize the map.
   3. Set **Seed Threshold (absolute)** — seeds are created where `score > threshold` (start around `2.0`, adjust as needed).
   4. Click **Run Watershed**.

      * Internally: foreground mask `score > 0`, Sobel gradient for the landscape, labeled seeds from the threshold, then `skimage.segmentation.watershed`.
      * A new **Labels** layer named `<basename>_watershed` is added.

   ### B) Z‑axis cleanup

   1. Set **Start Z** and **Stop Z** slice indices.
   2. Click **Clean Z Range** to zero labels outside `[Start Z, Stop Z)`.
   3. A new **Labels** layer called **cleaned_segmentation** appears.

   ### C) Connected components (relabel)

   1. Activate a **Labels** layer in the layer list.
   2. Toggle **Create New Layer** if you prefer a separate output.
   3. Click **Run Connected Components**.

   ### D) Label selection & editing

   1. Toggle **Label Selector Mode**.
   2. Click in the viewer to **pick label IDs**; they are listed under **Picked Labels**.
   3. Use **Remove Selected** to delete entries from the pick list.
   4. **Merge picked labels**:

      * Set **Merge picked labels into** (an integer ID; `0` merges into background).
      * Click **Merge Labels**.
   5. **Split selected label(s)** by CC:

      * Select one or more picked labels in the list.
      * Click **Split Selected Label (CC)** to split each into separate components and assign new IDs.

   ### E) Save to .mrc

   1. In **Save Segmentation**, choose your **Label Layer**.
   2. Set **Voxel Size (Å)** — written into the MRC header as isotropic `(vx, vy, vz)`.
   3. **Browse Output Path…** and select a filename (e.g. `my_seg.mrc`).
   4. Click **Save Segmentation** (automatic dtype fallback ensures success on large labels).

---

## ⌨️ Shortcuts

* **Ctrl/⌘+Z** — Undo last action (picks or previous label data in supported steps).
* **Delete / Backspace** — Remove highlighted IDs from **Picked Labels** list.

---

## 🎯 Typical workflows

* **Post‑process MemBrain‑seg output**

  1. Load the probability map, run **Watershed**, 2) **Connected components** to relabel instances, 3) **Split Selected Label (CC)** if touching instances remain, 4) **Merge** small artifacts into background or a main instance, 5) **Z‑axis cleanup**, 6) **Save**.

* **Manual curation of watershed instances**
  Pick suspect IDs, **Split** to separate, then **Merge** appropriate pieces back into the correct category/ID.

---

## ⚠️ Notes & limitations

* Watershed assumes **scores > 0** define the valid mask; set your preprocessing accordingly.
* **Seed Threshold** is **absolute** in score units; if your scores are already normalized to [0,1], start with ~0.2–0.4.
* Input must be **3D** for watershed and Z‑cleanup.
* Very large label counts may trigger automatic upcasting from `uint8` → `uint16` → `uint32` on save.

---

## 🧪 Testing data

* Any 3D probability map in **MRC** format (e.g., from MemBrain‑seg). For convenience, start with a small tomogram to dial in thresholds.

---

## 🛠 Troubleshooting

* **“Missing Score Volume”**: Pick a valid `.mrc` file (3D) via **Browse Score Volume…**.
* **“Invalid Data”** during watershed: Ensure the score volume is 3D.
* **“No label layer”** for CC/Z‑cleanup: Select a Labels layer in the viewer first.
* **“Invalid Z Range”**: `Stop Z` must be > `Start Z` and ≤ number of slices.
* **Saved file looks empty**: Confirm voxel size and that labels are non‑zero in the exported layer.

---

## 📦 API (advanced)

While Memsplit is GUI‑oriented, it relies on `skimage.filters.sobel`, `skimage.measure.label`, and `skimage.segmentation.watershed`. Advanced users may adapt the widget code to batch‑process volumes.

---

## 🙌 Acknowledgements

* Probability maps typically produced by **MemBrain‑seg** — membrane segmentation for cryo‑ET.
* Built with **napari**, **NumPy**, **scikit‑image**, **mrcfile**, and **Qt**.

---

## 📄 License

Specify your license here (e.g., MIT/BSD‑3‑Clause). Include a `LICENSE` file in the repository.

---

## 📚 Citation

If this tool was useful in your research, please cite this repository and MemBrain‑seg (see their repository for citation guidance).
