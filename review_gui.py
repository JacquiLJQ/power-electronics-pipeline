import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont


# =========================================================
# Config
# =========================================================

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
    # normal bbox style
    "normal_box_color": (0, 255, 0),
    "normal_text_bg": (0, 255, 0),
    "normal_text_fg": (0, 0, 0),
    # overlap-problem style
    "overlap_box_color": (255, 0, 0),
    "overlap_text_bg": (255, 0, 0),
    "overlap_text_fg": (255, 255, 255),
    # disconnected style if later still present
    "disconnected_box_color": (0, 0, 255),
    "disconnected_text_bg": (0, 0, 255),
    "disconnected_text_fg": (255, 255, 255),
    "line_width": 1,
    "font_size": 11,
}


# =========================================================
# Helpers
# =========================================================


def load_json(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_font(size=18):
    # PIL default font fallback
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def sort_index_entries(index_entries):
    """
    Priority:
      1) final_flag=True
      2) cross_class_overlap_flag=True
      3) disconnected_component_flag=True
      4) then by image_name
    """

    def key_fn(x):
        return (
            0 if x.get("final_flag", False) else 1,
            0 if x.get("cross_class_overlap_flag", False) else 1,
            0 if x.get("disconnected_component_flag", False) else 1,
            x.get("image_name", ""),
        )

    return sorted(index_entries, key=key_fn)


def get_overlap_component_ids(image_json):
    overlap_ids = set()
    for pair in image_json.get("cross_class_overlap_pairs", []):
        a = pair.get("component_id_a")
        b = pair.get("component_id_b")
        if a is not None:
            overlap_ids.add(a)
        if b is not None:
            overlap_ids.add(b)
    return overlap_ids


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

    # 半透明背景
    bg_rgba = (*bg_color, alpha)
    draw.rectangle(bg, fill=bg_rgba)

    # 半透明文字（关键！）
    fg_rgba = (*fg_color, 255)
    draw.text((x1, y1), text, fill=fg_rgba, font=font)


def render_annotated_image(image_json, gui_cfg):
    image_path = Path(image_json["image_path"])
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # 🔥 改成 RGBA
    base_img = Image.open(image_path).convert("RGBA")

    # overlay 层（透明）
    overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font = get_font(gui_cfg["font_size"])
    overlap_ids = get_overlap_component_ids(image_json)

    # 🎯 透明度（你可以调这个）
    ALPHA = 120

    for comp in image_json.get("components_current", []):
        comp_id = comp["component_id"]
        cls_id = comp["class_id"]
        x1, y1, x2, y2 = comp["bbox_xyxy"]

        # ===== 颜色选择 =====
        if comp_id in overlap_ids:
            base_color = gui_cfg["overlap_box_color"]
            text_bg = gui_cfg["overlap_text_bg"]
            text_fg = gui_cfg["overlap_text_fg"]
        elif comp.get("flags", {}).get("disconnected", False):
            base_color = gui_cfg["disconnected_box_color"]
            text_bg = gui_cfg["disconnected_text_bg"]
            text_fg = gui_cfg["disconnected_text_fg"]
        else:
            base_color = gui_cfg["normal_box_color"]
            text_bg = gui_cfg["normal_text_bg"]
            text_fg = gui_cfg["normal_text_fg"]

        # ===== 半透明颜色 =====
        rgba_color = (*base_color, ALPHA)

        # ===== 画边框（半透明）=====
        draw.rectangle(
            [x1, y1, x2, y2],
            outline=rgba_color,
            width=gui_cfg["line_width"],
        )

        # ===== label（保持不透明更清晰）=====
        label_text = str(cls_id)
        label_y = max(0, y1 - 22)

        draw_label(draw, x1, label_y, label_text, text_bg, text_fg, font, alpha=120)

    # 🔥 合成
    out = Image.alpha_composite(base_img, overlay)

    return out.convert("RGB")


# =========================================================
# Main GUI
# =========================================================


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

        # left: canvas area
        left_frame = ttk.Frame(outer)
        left_frame.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        # top control bar
        ctrl = ttk.Frame(left_frame)
        ctrl.pack(side="top", fill="x")

        self.prev_btn = ttk.Button(ctrl, text="Prev", command=self.prev_image)
        self.prev_btn.pack(side="left", padx=4)

        self.next_btn = ttk.Button(ctrl, text="Next", command=self.next_image)
        self.next_btn.pack(side="left", padx=4)

        self.goto_flagged_btn = ttk.Button(
            ctrl, text="Next Flagged", command=self.next_flagged_image
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

        # right: info panel
        right_frame = ttk.Frame(outer, width=360)
        right_frame.pack(side="right", fill="y", padx=8, pady=8)

        ttk.Label(right_frame, text="Review Panel", font=("Arial", 14, "bold")).pack(
            anchor="w", pady=(0, 10)
        )

        # current image info
        self.info_text = tk.Text(right_frame, width=42, height=18, wrap="word")
        self.info_text.pack(fill="x", pady=(0, 10))

        # legend
        ttk.Label(right_frame, text="Class Legend", font=("Arial", 12, "bold")).pack(
            anchor="w", pady=(8, 4)
        )
        self.legend_text = tk.Text(right_frame, width=42, height=20, wrap="word")
        self.legend_text.pack(fill="both", expand=True)

        self.fill_legend()

    def fill_legend(self):
        self.legend_text.delete("1.0", tk.END)
        self.legend_text.insert(tk.END, "Box colors:\n")
        self.legend_text.insert(tk.END, "  Green  = normal component\n")
        self.legend_text.insert(tk.END, "  Red    = cross-class overlap problem\n")
        self.legend_text.insert(tk.END, "  Blue   = disconnected (if present)\n\n")

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
        self.position_label.config(text=f"Image {self.current_idx + 1}/{total}")
        self.image_info_label.config(
            text=f"{entry['image_name']} | final_flag={entry.get('final_flag', False)}"
        )

    def update_info_panel(self, entry, image_json):
        self.info_text.config(state="normal")
        self.info_text.delete("1.0", tk.END)

        flags = image_json.get("flags", {})
        summary = image_json.get("summary", {})

        self.info_text.insert(tk.END, f"Image: {image_json.get('image_name', '')}\n")
        self.info_text.insert(tk.END, f"Path: {image_json.get('image_path', '')}\n\n")

        self.info_text.insert(tk.END, "Flags:\n")
        self.info_text.insert(
            tk.END, f"  final_flag: {flags.get('final_flag', False)}\n"
        )
        self.info_text.insert(
            tk.END,
            f"  cross_class_overlap_flag: {flags.get('cross_class_overlap_flag', False)}\n",
        )
        self.info_text.insert(
            tk.END,
            f"  disconnected_component_flag: {flags.get('disconnected_component_flag', False)}\n\n",
        )

        self.info_text.insert(tk.END, "Summary:\n")
        for k, v in summary.items():
            self.info_text.insert(tk.END, f"  {k}: {v}\n")

        overlap_pairs = image_json.get("cross_class_overlap_pairs", [])
        self.info_text.insert(tk.END, f"\nOverlap pairs: {len(overlap_pairs)}\n")
        for i, pair in enumerate(overlap_pairs[:10]):
            self.info_text.insert(
                tk.END,
                f"  {i}: {pair.get('component_id_a')} <-> {pair.get('component_id_b')} "
                f"(cls {pair.get('cls_a')} vs {pair.get('cls_b')})\n",
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
            if self.index_entries[i].get("final_flag", False):
                self.current_idx = i
                self.load_current_image()
                return
        messagebox.showinfo("Info", "No later flagged image found.")


def main():
    root = tk.Tk()
    app = ReviewGUI(root, CONFIG)
    root.mainloop()


if __name__ == "__main__":
    main()
