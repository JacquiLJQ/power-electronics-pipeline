import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont


CONFIG = {
    "output_dir": r"runs/pipeline_test",
    "canvas_width": 1100,
    "canvas_height": 760,
    "class_id_to_name": {
        0: "ac_src",
        1: "battery",
        2: "cap",
        3: "curr_src",
        4: "diode",
        5: "inductor",
        6: "resistor",
        7: "swi_ideal",
        8: "swi_real",
        9: "volt_src",
        10: "xformer",
    },
    "box_color": (0, 255, 0),
    "text_bg": (0, 255, 0),
    "text_fg": (0, 0, 0),
    "selected_box_color": (255, 255, 0),
    "draft_box_color": (255, 255, 0),
    "line_width": 2,
    "font_size": 12,
    "click_box_expand": 4,
    "min_box_size": 3,
}


def load_json(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, json_path: Path):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def get_font(size=18):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def sort_index_entries(index_entries):
    def is_modified(x):
        return (
            x.get("has_same_class_merge", False)
            or x.get("has_disconnected_removal", False)
            or x.get("has_cross_class_resolve", False)
            or x.get("has_manual_edit", False)
        )

    def key_fn(x):
        return (
            0 if is_modified(x) else 1,
            0 if x.get("has_cross_class_resolve", False) else 1,
            0 if x.get("has_disconnected_removal", False) else 1,
            0 if x.get("has_same_class_merge", False) else 1,
            0 if x.get("has_manual_edit", False) else 1,
            x.get("image_name", ""),
        )

    return sorted(index_entries, key=key_fn)


def is_entry_flagged(entry):
    return (
        entry.get("has_same_class_merge", False)
        or entry.get("has_disconnected_removal", False)
        or entry.get("has_cross_class_resolve", False)
        or entry.get("has_manual_edit", False)
    )


def fit_image_size(orig_w, orig_h, target_w, target_h):
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    return new_w, new_h, scale


def xyxy_to_xywh_abs(x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return [cx, cy, w, h]


def xyxy_to_yolo_line(cls_id, x1, y1, x2, y2, img_w, img_h):
    cx, cy, w, h = xyxy_to_xywh_abs(x1, y1, x2, y2)
    return f"{cls_id} {cx/img_w:.6f} {cy/img_h:.6f} {w/img_w:.6f} {h/img_h:.6f}"


def draw_label(draw, x1, y1, text, bg_color, fg_color, font, alpha=120):
    bbox = draw.textbbox((x1, y1), text, font=font)
    tx1, ty1, tx2, ty2 = bbox

    pad_x = 4
    pad_y = 2
    bg = [tx1 - pad_x, ty1 - pad_y, tx2 + pad_x, ty2 + pad_y]

    draw.rectangle(bg, fill=(*bg_color, alpha))
    draw.text((x1, y1), text, fill=(*fg_color, 255), font=font)


def point_in_box(px, py, box, expand=0):
    x1, y1, x2, y2 = box
    return (x1 - expand) <= px <= (x2 + expand) and (y1 - expand) <= py <= (y2 + expand)


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def render_annotated_image(image_json, gui_cfg, draft_box=None, selected_indices=None):
    if selected_indices is None:
        selected_indices = set()

    image_path = Path(image_json["image_path"])
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    base_img = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font = get_font(gui_cfg["font_size"])
    alpha = 120

    for idx, comp in enumerate(image_json.get("components", [])):
        cls_id = comp["class_id"]
        x1, y1, x2, y2 = comp["bbox_xyxy"]

        if idx in selected_indices:
            outline_color = (*gui_cfg["selected_box_color"], 220)
            line_width = gui_cfg["line_width"] + 2
        else:
            outline_color = (*gui_cfg["box_color"], alpha)
            line_width = gui_cfg["line_width"]

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=outline_color,
            width=line_width,
        )

        class_name = comp.get("class_name")
        label_text = f"{cls_id}" if class_name else str(cls_id)  #:{class_name}
        label_y = max(0, y1 - 22)

        draw_label(
            draw,
            x1,
            label_y,
            label_text,
            gui_cfg["text_bg"],
            gui_cfg["text_fg"],
            font,
            alpha=120,
        )

    if draft_box is not None:
        x1, y1, x2, y2 = draft_box
        draw.rectangle(
            [x1, y1, x2, y2],
            outline=(*gui_cfg["draft_box_color"], 180),
            width=2,
        )

    out = Image.alpha_composite(base_img, overlay)
    return out.convert("RGB")


class ReviewGUI:
    def __init__(self, root, gui_cfg):
        self.root = root
        self.cfg = gui_cfg
        self.root.title("Circuit BBox Review GUI")

        self.base_output_dir = Path(self.cfg["output_dir"]).resolve()
        self.postcheck_dir = self.base_output_dir / "postcheck"
        self.index_path = self.postcheck_dir / "index.json"

        if not self.index_path.exists():
            raise FileNotFoundError(f"index.json not found: {self.index_path}")

        self.index_entries = load_json(self.index_path)
        self.index_entries = sort_index_entries(self.index_entries)

        if len(self.index_entries) == 0:
            raise RuntimeError("No entries found in index.json")

        self.current_idx = 0
        self.tk_image = None
        self.current_image_json = None
        self.current_json_path = None

        self.selected_class_id = tk.IntVar(value=0)
        self.delete_mode = tk.BooleanVar(value=False)

        self.selected_component_indices = set()

        self.dragging = False
        self.drag_start_img = None
        self.drag_end_img = None

        self.display_scale = 1.0
        self.display_offset_x = 0
        self.display_offset_y = 0
        self.display_img_w = 0
        self.display_img_h = 0

        self.build_layout()
        self.load_current_image()

    def build_layout(self):
        self.root.geometry("1850x980")

        outer = ttk.Frame(self.root)
        outer.pack(fill="both", expand=True)

        left_frame = ttk.Frame(outer)
        left_frame.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        ctrl = ttk.Frame(left_frame)
        ctrl.pack(side="top", fill="x")

        ttk.Button(ctrl, text="Prev", command=self.prev_image).pack(side="left", padx=4)
        ttk.Button(ctrl, text="Next", command=self.next_image).pack(side="left", padx=4)
        ttk.Button(ctrl, text="Next Modified", command=self.next_flagged_image).pack(
            side="left", padx=4
        )

        ttk.Button(ctrl, text="Save", command=self.save_current).pack(
            side="left", padx=12
        )
        ttk.Button(
            ctrl,
            text="Change Selected Class",
            command=self.change_selected_component_class,
        ).pack(side="left", padx=6)
        ttk.Button(ctrl, text="Clear Selection", command=self.clear_selection).pack(
            side="left", padx=6
        )

        self.delete_mode_check = ttk.Checkbutton(
            ctrl, text="Delete Mode", variable=self.delete_mode
        )
        self.delete_mode_check.pack(side="left", padx=12)

        self.position_label = ttk.Label(ctrl, text="")
        self.position_label.pack(side="left", padx=12)

        self.image_info_label = ttk.Label(ctrl, text="")
        self.image_info_label.pack(side="left", padx=12)

        self.canvas = tk.Canvas(
            left_frame,
            width=self.cfg["canvas_width"],
            height=self.cfg["canvas_height"],
            bg="gray20",
        )
        self.canvas.pack(side="top", fill="both", expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_left_press)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)

        right_frame = ttk.Frame(outer, width=460)
        right_frame.pack(side="right", fill="y", padx=8, pady=8)

        ttk.Label(right_frame, text="Review Panel", font=("Arial", 14, "bold")).pack(
            anchor="w", pady=(0, 10)
        )

        self.info_text = tk.Text(right_frame, width=54, height=18, wrap="word")
        self.info_text.pack(fill="x", pady=(0, 10))

        ttk.Label(right_frame, text="Target Class", font=("Arial", 12, "bold")).pack(
            anchor="w", pady=(8, 4)
        )

        class_frame = ttk.Frame(right_frame)
        class_frame.pack(fill="x", pady=(0, 10))

        mapping = self.cfg["class_id_to_name"]
        for cls_id in sorted(mapping.keys()):
            ttk.Radiobutton(
                class_frame,
                text=f"{cls_id} -> {mapping[cls_id]}",
                value=cls_id,
                variable=self.selected_class_id,
            ).pack(anchor="w")

        ttk.Label(right_frame, text="Instructions", font=("Arial", 12, "bold")).pack(
            anchor="w", pady=(8, 4)
        )

        self.legend_text = tk.Text(right_frame, width=54, height=16, wrap="word")
        self.legend_text.pack(fill="both", expand=True)
        self.fill_legend()

    def fill_legend(self):
        self.legend_text.config(state="normal")
        self.legend_text.delete("1.0", tk.END)
        self.legend_text.insert(
            tk.END,
            "Left click on existing bbox:\n"
            "  Toggle selection for that bbox.\n"
            "  Multiple bboxes can be selected.\n\n"
            "Left drag on empty area:\n"
            "  Add a bbox with the selected class.\n\n"
            "Change Selected Class:\n"
            "  Change ALL selected bboxes to the chosen class.\n\n"
            "Delete Mode ON:\n"
            "  Click inside an existing bbox to delete it.\n\n"
            "Save:\n"
            "  Write updated components back to current image JSON\n"
            "  and rewrite export_labels/*.txt.\n",
        )
        self.legend_text.config(state="disabled")

    def get_current_entry(self):
        return self.index_entries[self.current_idx]

    def load_current_image(self):
        entry = self.get_current_entry()
        json_path = Path(entry["json_path"])
        self.current_json_path = json_path
        self.selected_component_indices.clear()

        if not json_path.exists():
            messagebox.showerror("Error", f"JSON not found:\n{json_path}")
            return

        self.current_image_json = load_json(json_path)

        flags = self.current_image_json.setdefault("flags", {})
        summary = self.current_image_json.setdefault("summary", {})
        flags.setdefault("has_manual_edit", False)
        summary.setdefault("manual_edit_count", 0)
        summary["final_component_count"] = len(
            self.current_image_json.get("components", [])
        )

        self.update_info_panel(entry, self.current_image_json)
        self.update_canvas()
        self.update_top_labels(entry)

    def update_top_labels(self, entry):
        total = len(self.index_entries)
        modified = is_entry_flagged(entry)
        self.position_label.config(text=f"Image {self.current_idx + 1}/{total}")
        self.image_info_label.config(
            text=f"{entry['image_name']} | modified={modified}"
        )

    def update_info_panel(self, entry, image_json):
        self.info_text.config(state="normal")
        self.info_text.delete("1.0", tk.END)

        flags = image_json.get("flags", {})
        summary = image_json.get("summary", {})
        components = image_json.get("components", [])

        self.info_text.insert(tk.END, f"Image: {image_json.get('image_name', '')}\n")
        self.info_text.insert(tk.END, f"Path: {image_json.get('image_path', '')}\n")
        self.info_text.insert(
            tk.END,
            f"Size: {image_json.get('image_width', '')} x {image_json.get('image_height', '')}\n\n",
        )

        self.info_text.insert(tk.END, "Modification Flags:\n")
        self.info_text.insert(
            tk.END,
            f"  has_same_class_merge: {flags.get('has_same_class_merge', False)}\n",
        )
        self.info_text.insert(
            tk.END,
            f"  has_disconnected_removal: {flags.get('has_disconnected_removal', False)}\n",
        )
        self.info_text.insert(
            tk.END,
            f"  has_cross_class_resolve: {flags.get('has_cross_class_resolve', False)}\n",
        )
        self.info_text.insert(
            tk.END, f"  has_manual_edit: {flags.get('has_manual_edit', False)}\n\n"
        )

        self.info_text.insert(tk.END, "Summary:\n")
        self.info_text.insert(
            tk.END, f"  raw_component_count: {summary.get('raw_component_count', 0)}\n"
        )
        self.info_text.insert(
            tk.END,
            f"  final_component_count: {summary.get('final_component_count', 0)}\n",
        )
        self.info_text.insert(
            tk.END,
            f"  merged_same_class_count: {summary.get('merged_same_class_count', 0)}\n",
        )
        self.info_text.insert(
            tk.END,
            f"  removed_cross_class_count: {summary.get('removed_cross_class_count', 0)}\n",
        )
        self.info_text.insert(
            tk.END,
            f"  removed_disconnected_count: {summary.get('removed_disconnected_count', 0)}\n",
        )
        self.info_text.insert(
            tk.END, f"  manual_edit_count: {summary.get('manual_edit_count', 0)}\n"
        )

        self.info_text.insert(
            tk.END, f"\nVisible final components: {len(components)}\n"
        )

        selected_indices = sorted(self.selected_component_indices)
        self.info_text.insert(
            tk.END, f"\nSelected bbox count: {len(selected_indices)}\n"
        )

        if selected_indices:
            self.info_text.insert(tk.END, "Selected bboxes:\n")
            for idx in selected_indices[:20]:
                if 0 <= idx < len(components):
                    comp = components[idx]
                    self.info_text.insert(
                        tk.END,
                        f"  idx={idx} | {comp.get('component_id')} | "
                        f"class={comp.get('class_id')}:{comp.get('class_name')}\n",
                    )

        self.info_text.config(state="disabled")

    def update_canvas(self, draft_box=None):
        image_json = self.current_image_json

        rendered = render_annotated_image(
            image_json,
            self.cfg,
            draft_box=draft_box,
            selected_indices=self.selected_component_indices,
        )

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w <= 1:
            canvas_w = self.cfg["canvas_width"]
        if canvas_h <= 1:
            canvas_h = self.cfg["canvas_height"]

        new_w, new_h, scale = fit_image_size(
            rendered.width, rendered.height, canvas_w, canvas_h
        )
        rendered = rendered.resize((new_w, new_h), Image.Resampling.NEAREST)

        self.display_scale = scale
        self.display_img_w = new_w
        self.display_img_h = new_h
        self.display_offset_x = (canvas_w - new_w) // 2
        self.display_offset_y = (canvas_h - new_h) // 2

        self.tk_image = ImageTk.PhotoImage(rendered)

        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_w // 2, canvas_h // 2, image=self.tk_image, anchor="center"
        )

    def canvas_to_image_coords(self, canvas_x, canvas_y):
        ix = (canvas_x - self.display_offset_x) / self.display_scale
        iy = (canvas_y - self.display_offset_y) / self.display_scale
        return ix, iy

    def clamp_image_coords(self, x, y):
        img_w = self.current_image_json["image_width"]
        img_h = self.current_image_json["image_height"]
        x = max(0, min(img_w - 1, x))
        y = max(0, min(img_h - 1, y))
        return x, y

    def make_new_component_id(self):
        comps = self.current_image_json.get("components", [])
        used = set()
        for c in comps:
            cid = c.get("component_id", "")
            if cid.startswith("manual_"):
                try:
                    used.add(int(cid.split("_")[1]))
                except Exception:
                    pass
        k = 0
        while k in used:
            k += 1
        return f"manual_{k}"

    def add_component(self, cls_id, x1, y1, x2, y2):
        x1, x2 = sorted([int(round(x1)), int(round(x2))])
        y1, y2 = sorted([int(round(y1)), int(round(y2))])

        if x2 - x1 < self.cfg["min_box_size"] or y2 - y1 < self.cfg["min_box_size"]:
            return False

        class_name = self.cfg["class_id_to_name"].get(cls_id)
        comp = {
            "component_id": self.make_new_component_id(),
            "class_id": cls_id,
            "class_name": class_name,
            "confidence": 1.0,
            "bbox_xyxy": [x1, y1, x2, y2],
            "bbox_xywh_abs": xyxy_to_xywh_abs(x1, y1, x2, y2),
            "source": "manual_add",
            "merged_from": [],
        }
        self.current_image_json["components"].append(comp)
        self.mark_manual_edit()
        return True

    def find_component_at(self, x, y):
        comps = self.current_image_json.get("components", [])
        candidates = []
        for idx, comp in enumerate(comps):
            box = comp["bbox_xyxy"]
            if point_in_box(x, y, box, expand=self.cfg["click_box_expand"]):
                candidates.append((box_area(box), idx))

        if not candidates:
            return None

        candidates.sort()
        return candidates[0][1]

    def toggle_component_selection(self, idx):
        if idx in self.selected_component_indices:
            self.selected_component_indices.remove(idx)
        else:
            self.selected_component_indices.add(idx)

    def delete_component_at_index(self, hit_idx):
        if hit_idx is None:
            return False

        comps = self.current_image_json.get("components", [])
        if not (0 <= hit_idx < len(comps)):
            return False

        del comps[hit_idx]

        new_selected = set()
        for idx in self.selected_component_indices:
            if idx == hit_idx:
                continue
            elif idx > hit_idx:
                new_selected.add(idx - 1)
            else:
                new_selected.add(idx)
        self.selected_component_indices = new_selected

        self.mark_manual_edit()
        return True

    def change_selected_component_class(self):
        if not self.selected_component_indices:
            messagebox.showwarning("Warning", "No bbox selected.")
            return

        comps = self.current_image_json.get("components", [])
        new_cls_id = self.selected_class_id.get()
        new_class_name = self.cfg["class_id_to_name"].get(new_cls_id)

        changed = 0
        valid_indices = sorted(
            [idx for idx in self.selected_component_indices if 0 <= idx < len(comps)]
        )

        for idx in valid_indices:
            comp = comps[idx]
            old_cls_id = comp["class_id"]
            if old_cls_id == new_cls_id:
                continue

            comp["class_id"] = new_cls_id
            comp["class_name"] = new_class_name
            comp["source"] = "manual_class_change"
            changed += 1

        if changed > 0:
            self.delete_mode.set(False)
            self.mark_manual_edit()
            self.refresh_current_view()

    def clear_selection(self):
        self.selected_component_indices.clear()
        self.refresh_current_view()

    def mark_manual_edit(self):
        flags = self.current_image_json.setdefault("flags", {})
        summary = self.current_image_json.setdefault("summary", {})

        flags["has_manual_edit"] = True
        summary["manual_edit_count"] = summary.get("manual_edit_count", 0) + 1
        summary["final_component_count"] = len(
            self.current_image_json.get("components", [])
        )

    def on_left_press(self, event):
        if self.current_image_json is None:
            return

        x, y = self.canvas_to_image_coords(event.x, event.y)
        x, y = self.clamp_image_coords(x, y)

        hit_idx = self.find_component_at(x, y)

        if self.delete_mode.get():
            if hit_idx is not None:
                deleted = self.delete_component_at_index(hit_idx)
                if deleted:
                    self.refresh_current_view()
            return

        if hit_idx is not None:
            self.toggle_component_selection(hit_idx)
            self.refresh_current_view()
            return

        self.dragging = True
        self.drag_start_img = (x, y)
        self.drag_end_img = (x, y)

    def on_left_drag(self, event):
        if not self.dragging:
            return

        x, y = self.canvas_to_image_coords(event.x, event.y)
        x, y = self.clamp_image_coords(x, y)
        self.drag_end_img = (x, y)

        x1, y1 = self.drag_start_img
        x2, y2 = self.drag_end_img
        self.update_canvas(draft_box=[x1, y1, x2, y2])

    def on_left_release(self, event):
        if not self.dragging:
            return

        x, y = self.canvas_to_image_coords(event.x, event.y)
        x, y = self.clamp_image_coords(x, y)
        self.drag_end_img = (x, y)

        x1, y1 = self.drag_start_img
        x2, y2 = self.drag_end_img

        cls_id = self.selected_class_id.get()
        added = self.add_component(cls_id, x1, y1, x2, y2)

        self.selected_component_indices.clear()
        self.dragging = False
        self.drag_start_img = None
        self.drag_end_img = None

        if added:
            self.refresh_current_view()
        else:
            self.update_canvas()

    def refresh_current_view(self):
        entry = self.get_current_entry()
        self.update_info_panel(entry, self.current_image_json)
        self.update_canvas()
        self.update_top_labels(entry)

    def save_current(self):
        if self.current_image_json is None or self.current_json_path is None:
            return

        image_json = self.current_image_json
        summary = image_json.setdefault("summary", {})
        flags = image_json.setdefault("flags", {})

        flags.setdefault("has_manual_edit", False)
        summary.setdefault("manual_edit_count", 0)
        summary["final_component_count"] = len(image_json.get("components", []))

        for comp in image_json.get("components", []):
            x1, y1, x2, y2 = comp["bbox_xyxy"]
            comp["bbox_xywh_abs"] = xyxy_to_xywh_abs(x1, y1, x2, y2)

        save_json(image_json, self.current_json_path)

        entry = self.get_current_entry()
        export_label_path = entry.get("export_label_path")

        if export_label_path:
            self.save_export_labels(Path(export_label_path), image_json)
        else:
            messagebox.showwarning(
                "Warning",
                "JSON saved, but export_label_path is missing, so YOLO txt was not updated.",
            )

        self.update_index_entry_after_save()
        messagebox.showinfo("Saved", f"Saved:\n{self.current_json_path}")

    def save_export_labels(self, out_txt: Path, image_json):
        out_txt.parent.mkdir(parents=True, exist_ok=True)

        img_w = image_json["image_width"]
        img_h = image_json["image_height"]

        with open(out_txt, "w", encoding="utf-8") as f:
            for comp in image_json.get("components", []):
                cls_id = comp["class_id"]
                x1, y1, x2, y2 = comp["bbox_xyxy"]
                f.write(xyxy_to_yolo_line(cls_id, x1, y1, x2, y2, img_w, img_h) + "\n")

    def update_index_entry_after_save(self):
        entry = self.get_current_entry()
        flags = self.current_image_json.get("flags", {})
        summary = self.current_image_json.get("summary", {})

        entry["has_same_class_merge"] = flags.get("has_same_class_merge", False)
        entry["has_disconnected_removal"] = flags.get("has_disconnected_removal", False)
        entry["has_cross_class_resolve"] = flags.get("has_cross_class_resolve", False)
        entry["has_manual_edit"] = flags.get("has_manual_edit", False)
        entry["final_component_count"] = summary.get("final_component_count", 0)

        save_json(self.index_entries, self.index_path)

    def prev_image(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_current_image()

    def next_image(self):
        if self.current_idx < len(self.index_entries) - 1:
            self.current_idx += 1
            self.load_current_image()

    def next_flagged_image(self):
        for i in range(self.current_idx + 1, len(self.index_entries)):
            e = self.index_entries[i]
            if (
                e.get("has_same_class_merge", False)
                or e.get("has_disconnected_removal", False)
                or e.get("has_cross_class_resolve", False)
                or e.get("has_manual_edit", False)
            ):
                self.current_idx = i
                self.load_current_image()
                return
        messagebox.showinfo("Info", "No later modified image found.")


def main():
    root = tk.Tk()
    app = ReviewGUI(root, CONFIG)
    root.mainloop()


if __name__ == "__main__":
    main()
