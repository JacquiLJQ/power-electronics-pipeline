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
    "line_width": 1,
    "font_size": 11,
}


def load_json(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


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
        )

    def key_fn(x):
        return (
            0 if is_modified(x) else 1,
            0 if x.get("has_cross_class_resolve", False) else 1,
            0 if x.get("has_disconnected_removal", False) else 1,
            0 if x.get("has_same_class_merge", False) else 1,
            x.get("image_name", ""),
        )

    return sorted(index_entries, key=key_fn)


def is_entry_flagged(entry):
    return (
        entry.get("has_same_class_merge", False)
        or entry.get("has_disconnected_removal", False)
        or entry.get("has_cross_class_resolve", False)
    )


def fit_image_size(orig_w, orig_h, target_w, target_h):
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    return new_w, new_h, scale


def draw_label(draw, x1, y1, text, bg_color, fg_color, font, alpha=120):
    bbox = draw.textbbox((x1, y1), text, font=font)
    tx1, ty1, tx2, ty2 = bbox

    pad_x = 4
    pad_y = 2

    bg = [
        tx1 - pad_x,
        ty1 - pad_y,
        tx2 + pad_x,
        ty2 + pad_y,
    ]

    bg_rgba = (*bg_color, alpha)
    draw.rectangle(bg, fill=bg_rgba)

    fg_rgba = (*fg_color, 255)
    draw.text((x1, y1), text, fill=fg_rgba, font=font)


def render_annotated_image(image_json, gui_cfg):
    image_path = Path(image_json["image_path"])
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    base_img = Image.open(image_path).convert("RGBA")

    overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font = get_font(gui_cfg["font_size"])
    alpha = 120

    for comp in image_json.get("components", []):
        cls_id = comp["class_id"]
        x1, y1, x2, y2 = comp["bbox_xyxy"]

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=(*gui_cfg["box_color"], alpha),
            width=gui_cfg["line_width"],
        )

        class_name = comp.get("class_name")
        label_text = f"{cls_id}:{class_name}" if class_name else str(cls_id)
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

        self.build_layout()
        self.load_current_image()

    def build_layout(self):
        self.root.geometry("1600x900")

        outer = ttk.Frame(self.root)
        outer.pack(fill="both", expand=True)

        left_frame = ttk.Frame(outer)
        left_frame.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        ctrl = ttk.Frame(left_frame)
        ctrl.pack(side="top", fill="x")

        self.prev_btn = ttk.Button(ctrl, text="Prev", command=self.prev_image)
        self.prev_btn.pack(side="left", padx=4)

        self.next_btn = ttk.Button(ctrl, text="Next", command=self.next_image)
        self.next_btn.pack(side="left", padx=4)

        self.goto_flagged_btn = ttk.Button(
            ctrl, text="Next Modified", command=self.next_flagged_image
        )
        self.goto_flagged_btn.pack(side="left", padx=4)

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

        right_frame = ttk.Frame(outer, width=420)
        right_frame.pack(side="right", fill="y", padx=8, pady=8)

        ttk.Label(right_frame, text="Review Panel", font=("Arial", 14, "bold")).pack(
            anchor="w", pady=(0, 10)
        )

        self.info_text = tk.Text(right_frame, width=50, height=22, wrap="word")
        self.info_text.pack(fill="x", pady=(0, 10))

        ttk.Label(right_frame, text="Class Legend", font=("Arial", 12, "bold")).pack(
            anchor="w", pady=(8, 4)
        )
        self.legend_text = tk.Text(right_frame, width=50, height=20, wrap="word")
        self.legend_text.pack(fill="both", expand=True)

        self.fill_legend()

    def fill_legend(self):
        self.legend_text.config(state="normal")
        self.legend_text.delete("1.0", tk.END)

        self.legend_text.insert(tk.END, "Box color:\n")
        self.legend_text.insert(tk.END, "  Green = final kept component\n\n")

        self.legend_text.insert(
            tk.END,
            "Flags shown in the right panel:\n"
            "  has_same_class_merge\n"
            "  has_disconnected_removal\n"
            "  has_cross_class_resolve\n\n",
        )

        self.legend_text.insert(tk.END, "Class ID mapping:\n")
        mapping = self.cfg.get("class_id_to_name", {})
        for cls_id in sorted(mapping.keys()):
            self.legend_text.insert(tk.END, f"  {cls_id} -> {mapping[cls_id]}\n")

        self.legend_text.config(state="disabled")

    def get_current_entry(self):
        return self.index_entries[self.current_idx]

    def load_current_image(self):
        entry = self.get_current_entry()
        json_path = Path(entry["json_path"])

        if not json_path.exists():
            messagebox.showerror("Error", f"JSON not found:\n{json_path}")
            return

        image_json = load_json(json_path)
        self.current_image_json = image_json

        self.update_info_panel(entry, image_json)
        self.update_canvas(image_json)
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
            f"  has_cross_class_resolve: {flags.get('has_cross_class_resolve', False)}\n\n",
        )

        self.info_text.insert(tk.END, "Summary:\n")
        self.info_text.insert(
            tk.END,
            f"  raw_component_count: {summary.get('raw_component_count', 0)}\n",
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
            tk.END, f"\nVisible final components: {len(components)}\n"
        )
        self.info_text.config(state="disabled")

    def update_canvas(self, image_json):
        rendered = render_annotated_image(image_json, self.cfg)

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w <= 1:
            canvas_w = self.cfg["canvas_width"]
        if canvas_h <= 1:
            canvas_h = self.cfg["canvas_height"]

        new_w, new_h, _ = fit_image_size(
            rendered.width, rendered.height, canvas_w, canvas_h
        )
        rendered = rendered.resize((new_w, new_h), Image.Resampling.NEAREST)

        self.tk_image = ImageTk.PhotoImage(rendered)

        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_w // 2, canvas_h // 2, image=self.tk_image, anchor="center"
        )

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
            if is_entry_flagged(self.index_entries[i]):
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
