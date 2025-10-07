import os
import json
import shutil
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import albumentations as A

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def clamp(v, a, b):
    return max(a, min(b, v))


class ImageAnnotatorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Annotator")
        self.geometry("1200x800")
        self.minsize(900, 600)
        self.image_dir = None
        self.image_paths = []
        self.index = 0
        self.current_image = None
        self.current_image_tk = None
        self.current_image_size = (0, 0)
        self.scale = 1.0
        self.offset = (0, 0)
        self.drawing = False
        self.start_pt = None
        self.temp_rect_id = None
        self.rect_ids = []
        self.selected_idx = None
        self.annotations = {}
        self.image_sizes = {}
        self.classes = ["object"]
        self.current_label = tk.StringVar(value=self.classes[0])
        self._build_ui()

    def _build_ui(self):
        topbar = ttk.Frame(self)
        topbar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(topbar, text="Open Folder", command=self.open_folder).pack(side=tk.LEFT, padx=4, pady=4)
        ttk.Button(topbar, text="Prev", command=self.prev_image).pack(side=tk.LEFT, padx=4, pady=4)
        ttk.Button(topbar, text="Next", command=self.next_image).pack(side=tk.LEFT, padx=4, pady=4)
        ttk.Button(topbar, text="Save", command=self.save_all).pack(side=tk.LEFT, padx=4, pady=4)
        ttk.Button(topbar, text="Export YOLO", command=self.export_yolo).pack(side=tk.LEFT, padx=4, pady=4)
        ttk.Button(topbar, text="Export COCO", command=self.export_coco).pack(side=tk.LEFT, padx=4, pady=4)
        ttk.Button(topbar, text="Augment", command=self.augment_dialog).pack(side=tk.LEFT, padx=4, pady=4)
        self.status_var = tk.StringVar(value="")
        ttk.Label(topbar, textvariable=self.status_var).pack(side=tk.RIGHT, padx=8)

        body = ttk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(body)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(left, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self.refresh_canvas())
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        right = ttk.Frame(body, width=320)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        info_frame = ttk.Frame(right)
        info_frame.pack(fill=tk.X, padx=8, pady=8)
        self.path_var = tk.StringVar(value="")
        ttk.Label(info_frame, textvariable=self.path_var).pack(anchor=tk.W)

        class_frame = ttk.LabelFrame(right, text="Class")
        class_frame.pack(fill=tk.X, padx=8, pady=8)
        self.class_combo = ttk.Combobox(class_frame, textvariable=self.current_label, values=self.classes, state="normal")
        self.class_combo.pack(fill=tk.X, padx=6, pady=6)

        boxes_frame = ttk.LabelFrame(right, text="Boxes")
        boxes_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.box_list = tk.Listbox(boxes_frame, height=12)
        self.box_list.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.box_list.bind("<<ListboxSelect>>", self.on_select_box)

        btns = ttk.Frame(boxes_frame)
        btns.pack(fill=tk.X)
        ttk.Button(btns, text="Delete", command=self.delete_selected).pack(side=tk.LEFT, padx=4, pady=4)
        ttk.Button(btns, text="Clear All", command=self.clear_all_boxes).pack(side=tk.LEFT, padx=4, pady=4)

        self.bind("<Left>", lambda e: self.prev_image())
        self.bind("<Right>", lambda e: self.next_image())
        self.bind("<Delete>", lambda e: self.delete_selected())

    def open_folder(self):
        d = filedialog.askdirectory()
        if not d:
            return
        self.image_dir = Path(d)
        self.image_paths = sorted([p for p in self.image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
        if not self.image_paths:
            messagebox.showerror("No images", "No images found in the selected folder.")
            return
        self.load_state()
        self.index = 0
        self.load_current_image()

    def load_state(self):
        self.annotations = {}
        self.image_sizes = {}
        self.classes = ["object"]
        ann_path = self.image_dir / "annotations.json"
        if ann_path.exists():
            try:
                data = json.loads(ann_path.read_text(encoding="utf-8"))
                self.annotations = data.get("annotations", {})
                self.image_sizes = data.get("image_sizes", {})
                self.classes = data.get("classes", ["object"]) or ["object"]
            except Exception:
                pass
        classes_path = self.image_dir / "classes.txt"
        if classes_path.exists():
            try:
                lines = [x.strip() for x in classes_path.read_text(encoding="utf-8").splitlines() if x.strip()]
                if lines:
                    self.classes = lines
            except Exception:
                pass
        self.current_label.set(self.classes[0])
        self.class_combo.configure(values=self.classes)

    def save_state(self):
        if not self.image_dir:
            return
        out = {
            "annotations": self.annotations,
            "image_sizes": self.image_sizes,
            "classes": self.classes,
        }
        (self.image_dir / "annotations.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.image_dir / "classes.txt").write_text("\n".join(self.classes), encoding="utf-8")

    def load_current_image(self):
        if not self.image_paths:
            return
        p = self.image_paths[self.index]
        self.path_var.set(f"{self.index+1}/{len(self.image_paths)}  {p.name}")
        img = Image.open(p).convert("RGB")
        self.current_image = img
        self.current_image_size = img.size
        key = p.name
        if key not in self.image_sizes:
            self.image_sizes[key] = {"width": img.width, "height": img.height}
        if key not in self.annotations:
            self.annotations[key] = []
        self.refresh_boxes_list()
        self.refresh_canvas()

    def compute_fit(self, w, h, cw, ch):
        if w == 0 or h == 0 or cw == 0 or ch == 0:
            return 1.0, (0, 0), (w, h)
        s = min(cw / w, ch / h)
        dw, dh = int(w * s), int(h * s)
        ox = (cw - dw) // 2
        oy = (ch - dh) // 2
        return s, (ox, oy), (dw, dh)

    def refresh_canvas(self):
        self.canvas.delete("all")
        if self.current_image is None:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        w, h = self.current_image_size
        self.scale, self.offset, disp_size = self.compute_fit(w, h, cw, ch)
        disp = self.current_image.resize(disp_size, Image.LANCZOS)
        self.current_image_tk = ImageTk.PhotoImage(disp)
        self.canvas.create_image(self.offset[0], self.offset[1], anchor=tk.NW, image=self.current_image_tk)
        self.rect_ids = []
        for i, ann in enumerate(self.annotations.get(self.image_paths[self.index].name, [])):
            x1, y1, x2, y2 = ann["bbox"]
            sx1 = self.offset[0] + int(x1 * self.scale)
            sy1 = self.offset[1] + int(y1 * self.scale)
            sx2 = self.offset[0] + int(x2 * self.scale)
            sy2 = self.offset[1] + int(y2 * self.scale)
            color = "#00ff88" if i != self.selected_idx else "#ffcc00"
            rid = self.canvas.create_rectangle(sx1, sy1, sx2, sy2, outline=color, width=2)
            self.rect_ids.append(rid)

    def image_coords(self, sx, sy):
        x = (sx - self.offset[0]) / max(self.scale, 1e-8)
        y = (sy - self.offset[1]) / max(self.scale, 1e-8)
        x = clamp(x, 0, self.current_image_size[0])
        y = clamp(y, 0, self.current_image_size[1])
        return int(x), int(y)

    def on_mouse_down(self, e):
        if self.current_image is None:
            return
        x, y = self.image_coords(e.x, e.y)
        self.drawing = True
        self.start_pt = (x, y)
        self.temp_rect_id = None

    def on_mouse_drag(self, e):
        if not self.drawing or self.current_image is None or self.start_pt is None:
            return
        x, y = self.image_coords(e.x, e.y)
        x1, y1 = self.start_pt
        sx1 = self.offset[0] + int(min(x1, x) * self.scale)
        sy1 = self.offset[1] + int(min(y1, y) * self.scale)
        sx2 = self.offset[0] + int(max(x1, x) * self.scale)
        sy2 = self.offset[1] + int(max(y1, y) * self.scale)
        if self.temp_rect_id is None:
            self.temp_rect_id = self.canvas.create_rectangle(sx1, sy1, sx2, sy2, outline="#00bfff", dash=(4, 2), width=2)
        else:
            self.canvas.coords(self.temp_rect_id, sx1, sy1, sx2, sy2)

    def on_mouse_up(self, e):
        if not self.drawing or self.current_image is None or self.start_pt is None:
            return
        x, y = self.image_coords(e.x, e.y)
        x1, y1 = self.start_pt
        self.drawing = False
        self.start_pt = None
        if self.temp_rect_id is not None:
            self.canvas.delete(self.temp_rect_id)
            self.temp_rect_id = None
        bx1, by1 = min(x1, x), min(y1, y)
        bx2, by2 = max(x1, x), max(y1, y)
        if bx2 - bx1 < 3 or by2 - by1 < 3:
            return
        label = self.current_label.get().strip() or "object"
        if label not in self.classes:
            self.classes.append(label)
            self.class_combo.configure(values=self.classes)
        key = self.image_paths[self.index].name
        self.annotations.setdefault(key, []).append({"bbox": [int(bx1), int(by1), int(bx2), int(by2)], "label": label})
        self.refresh_boxes_list()
        self.refresh_canvas()
        self.save_state()

    def refresh_boxes_list(self):
        self.box_list.delete(0, tk.END)
        key = self.image_paths[self.index].name if self.image_paths else None
        if not key:
            return
        for ann in self.annotations.get(key, []):
            x1, y1, x2, y2 = ann["bbox"]
            label = ann.get("label", "object")
            self.box_list.insert(tk.END, f"{label} [{x1},{y1},{x2},{y2}]")
        self.selected_idx = None

    def on_select_box(self, e):
        sel = self.box_list.curselection()
        self.selected_idx = sel[0] if sel else None
        self.refresh_canvas()

    def delete_selected(self):
        key = self.image_paths[self.index].name if self.image_paths else None
        if key is None:
            return
        sel = self.box_list.curselection()
        if not sel:
            return
        idx = sel[0]
        if 0 <= idx < len(self.annotations.get(key, [])):
            self.annotations[key].pop(idx)
            self.refresh_boxes_list()
            self.refresh_canvas()
            self.save_state()

    def clear_all_boxes(self):
        key = self.image_paths[self.index].name if self.image_paths else None
        if key is None:
            return
        if not self.annotations.get(key):
            return
        if not messagebox.askyesno("Confirm", "Clear all boxes for this image?"):
            return
        self.annotations[key] = []
        self.refresh_boxes_list()
        self.refresh_canvas()
        self.save_state()

    def prev_image(self):
        if not self.image_paths:
            return
        self.index = (self.index - 1) % len(self.image_paths)
        self.load_current_image()

    def next_image(self):
        if not self.image_paths:
            return
        self.index = (self.index + 1) % len(self.image_paths)
        self.load_current_image()

    def save_all(self):
        self.save_state()
        self.status_var.set("Saved")
        self.after(1200, lambda: self.status_var.set(""))

    def export_yolo(self):
        if not self.image_paths:
            return
        out_dir = filedialog.askdirectory(title="Select export directory for YOLO")
        if not out_dir:
            return
        out = Path(out_dir)
        img_out = out / "images"
        lbl_out = out / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        (out / "classes.txt").write_text("\n".join(self.classes), encoding="utf-8")
        for p in self.image_paths:
            key = p.name
            anns = self.annotations.get(key, [])
            shutil.copy2(p, img_out / p.name)
            try:
                with Image.open(p) as im:
                    w, h = im.size
            except Exception:
                size = self.image_sizes.get(key, {})
                w = size.get("width", 0)
                h = size.get("height", 0)
            if not w or not h:
                continue
            lines = []
            for ann in anns:
                x1, y1, x2, y2 = ann["bbox"]
                cx = ((x1 + x2) / 2.0) / w
                cy = ((y1 + y2) / 2.0) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                cid = self.classes.index(ann.get("label", "object"))
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (lbl_out / (p.stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
        messagebox.showinfo("Export YOLO", f"Exported to {out}")

    def export_coco(self):
        if not self.image_paths:
            return
        out_file = filedialog.asksaveasfilename(title="Save COCO JSON", defaultextension=".json", filetypes=[("JSON","*.json")], initialfile="annotations.coco.json")
        if not out_file:
            return
        cat_id_map = {name: i + 1 for i, name in enumerate(self.classes)}
        images = []
        annotations = []
        ann_id = 1
        for i, p in enumerate(self.image_paths, start=1):
            key = p.name
            size = self.image_sizes.get(key) or {}
            w = size.get("width", 0)
            h = size.get("height", 0)
            if w == 0 or h == 0:
                try:
                    img = Image.open(p)
                    w, h = img.size
                except Exception:
                    continue
            images.append({"id": i, "file_name": p.name, "width": w, "height": h})
            for ann in self.annotations.get(key, []):
                x1, y1, x2, y2 = ann["bbox"]
                bw = x2 - x1
                bh = y2 - y1
                cat = cat_id_map.get(ann.get("label", "object"), 1)
                annotations.append({
                    "id": ann_id,
                    "image_id": i,
                    "category_id": cat,
                    "bbox": [x1, y1, bw, bh],
                    "area": float(bw * bh),
                    "iscrowd": 0,
                    "segmentation": [],
                })
                ann_id += 1
        categories = [{"id": i + 1, "name": name, "supercategory": ""} for i, name in enumerate(self.classes)]
        coco = {"images": images, "annotations": annotations, "categories": categories}
        Path(out_file).write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")
        messagebox.showinfo("Export COCO", f"Saved {out_file}")

    def augment_dialog(self):
        if not self.image_paths:
            return
        dlg = tk.Toplevel(self)
        dlg.title("Augment Dataset")
        dlg.transient(self)
        ttk.Label(dlg, text="Output directory").grid(row=0, column=0, sticky=tk.W, padx=6, pady=6)
        out_var = tk.StringVar()
        entry = ttk.Entry(dlg, textvariable=out_var, width=40)
        entry.grid(row=0, column=1, padx=6, pady=6)
        def browse():
            d = filedialog.askdirectory(title="Select output directory for augmented set")
            if d:
                out_var.set(d)
        ttk.Button(dlg, text="Browse", command=browse).grid(row=0, column=2, padx=6, pady=6)
        ttk.Label(dlg, text="Augmentations per image").grid(row=1, column=0, sticky=tk.W, padx=6, pady=6)
        count_var = tk.IntVar(value=5)
        ttk.Spinbox(dlg, from_=1, to=100, textvariable=count_var, width=8).grid(row=1, column=1, sticky=tk.W, padx=6, pady=6)
        def run():
            out = out_var.get().strip()
            if not out:
                messagebox.showerror("Error", "Select an output directory")
                return
            self.augment_dataset(Path(out), count_var.get())
            dlg.destroy()
        ttk.Button(dlg, text="Start", command=run).grid(row=2, column=0, padx=6, pady=12)
        ttk.Button(dlg, text="Cancel", command=dlg.destroy).grid(row=2, column=1, padx=6, pady=12)
        dlg.grab_set()
        dlg.wait_window()

    def augment_dataset(self, out_dir: Path, per_image: int):
        out_images = out_dir / "images"
        out_labels = out_dir / "labels"
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)
        (out_dir / "classes.txt").write_text("\n".join(self.classes), encoding="utf-8")
        total = 0
        for p in self.image_paths:
            key = p.name
            anns = self.annotations.get(key, [])
            if not anns:
                continue
            try:
                img = cv2.imread(str(p))
                if img is None:
                    continue
                h, w = img.shape[:2]
            except Exception:
                continue
            bboxes = []
            labels = []
            for ann in anns:
                x1, y1, x2, y2 = ann["bbox"]
                bboxes.append([float(x1), float(y1), float(x2), float(y2)])
                labels.append(ann.get("label", "object"))
            aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.3),
                A.GaussNoise(p=0.2),
                A.MotionBlur(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.8),
                A.RGBShift(p=0.2),
                A.ChannelShuffle(p=0.05),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_area=16, min_visibility=0.3))
            created = 0
            attempts = 0
            while created < per_image and attempts < per_image * 12:
                attempts += 1
                res = aug(image=img, bboxes=bboxes, labels=labels)
                aboxes = res["bboxes"]
                alabels = res["labels"]
                if not aboxes:
                    continue
                out_name = f"{p.stem}_aug_{created+1:02d}{p.suffix}"
                out_path = out_images / out_name
                cv2.imwrite(str(out_path), res["image"])
                yolo_lines = []
                for bb, lb in zip(aboxes, alabels):
                    x1, y1, x2, y2 = bb
                    cx = ((x1 + x2) / 2.0) / w
                    cy = ((y1 + y2) / 2.0) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    cid = self.classes.index(lb)
                    yolo_lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                (out_labels / f"{Path(out_name).stem}.txt").write_text("\n".join(yolo_lines), encoding="utf-8")
                created += 1
                total += 1
        messagebox.showinfo("Augment", f"Created {total} augmented images in {out_dir}")


def main():
    app = ImageAnnotatorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
