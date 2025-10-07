<<<<<<< HEAD
# tagforge
Minimal, cross-platform image annotation tool built for fast dataset creation in computer vision workflows.
=======
# Image Annotator

A simple Tkinter app for bounding-box annotation and dataset export (YOLO/COCO), plus dataset augmentation using Albumentations.

## Features
- **Draw boxes**: Click–drag to draw bounding boxes on images.
- **Multi-class**: Type a class name in the Class box; new names are auto-added to `classes.txt`.
- **Autosave**: Annotations persist in `annotations.json` inside your image folder.
- **Navigation**: Prev/Next buttons and Left/Right arrow keys.
- **Manage boxes**: Select in the list, press Delete or use Clear All.
- **Export formats**: YOLO (images/labels/classes.txt) and COCO JSON.
- **Augmentation**: Generates augmented images and updated YOLO labels with best-practice transforms.

## Requirements
- Windows
- Python 3.10+ recommended

## Setup
```powershell
# Go to the project directory
# Set this project as your active workspace if your IDE supports it

py -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## Run
```powershell
.\.venv\Scripts\python .\main.py
```

## Usage
- **Open Folder**: Choose a folder containing images (`.jpg,.jpeg,.png,.bmp,.tif,.tiff,.webp`).
- **Class**: Use the dropdown or type to add a new class. New classes persist to `classes.txt` in the same folder.
- **Draw**: Click–drag on the image to create a box. Boxes save automatically.
- **Select/Remove**: Click a box entry, press Delete or click Clear All.
- **Navigate**: Prev/Next buttons or Left/Right arrows.
- **Export YOLO**: Choose an output directory. App writes `images/`, `labels/`, and `classes.txt`. Labels are in normalized `class cx cy w h` format.
- **Export COCO**: Choose a JSON file path to save COCO annotations.
- **Augment**: Choose an output directory and how many augmentations per image. App writes augmented `images/` and YOLO `labels/` with boxes transformed to match.

## Notes
- Classes are loaded from and saved to `classes.txt` in your image folder. Edit it to reorder/rename carefully.
- COCO export includes categories from your current `classes.txt` and boxes as `[x, y, width, height]` in pixels.
- Augmentations include flips, brightness/contrast, hue/saturation, noise, blur, and shift/scale/rotate with bbox-aware transforms.

## Known limitations
- Boxes cannot be moved/resized after creation (delete and redraw).
- No zoom/pan. Large images are fitted to the window.
- Single-object type per box (no segmentation/polygons).

## Troubleshooting
- If the app doesn’t start, ensure the virtual environment is active and dependencies installed.
- If no images are found, verify extensions and that the folder contains images directly (no subfolders processed).
- If Albumentations errors on bboxes, ensure you have at least one valid box per image before augmenting.
>>>>>>> 22d386a (Initial commit)
