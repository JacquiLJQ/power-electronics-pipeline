import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
import tkinter as tk
from tkinter import messagebox, ttk


DEFAULT_CLASS_MAP = {
    0: "ac_src",  # positive / negative
    1: "battery",  # positive / negative
    2: "cap",  # terminal 1 / terminal 2
    3: "curr_src",  # positive / negative
    4: "diode",  # anode / cathode
    5: "inductor",  # terminal 1 / terminal 2
    6: "resistor",  # terminal 1 / terminal 2
    7: "swi_ideal",  # terminal 1 / terminal 2
    8: "swi_real",  # MOSFET non-two terminal
    9: "volt_src",  # positive / negative
    10: "xformer",  # non-two terminal
}


@dataclass
class Component:
    component_id: str
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: List[int]
    bbox_xywh_abs: List[float]
    source: str = "yolo"


@dataclass
class AutoNode:
    id: int
    area: int
    bbox_xywh: Tuple[int, int, int, int]
    centroid_xy: Tuple[float, float]


@dataclass
class ManualNode:
    id: int
    x: int
    y: int
    radius: int = 6
    kind: str = "manual_circle"


@dataclass
class NodeGroup:
    id: int
    members_auto: List[int]
    members_manual: List[int]


@dataclass(frozen=True)
class Connection:
    node_id: int
    component_id: str


class IntegratedCircuitGUI:
    def __init__(self, root: tk.Tk, args: argparse.Namespace):
        self.root = root
        self.args = args
        self.root.title("Circuit Component + Node Annotation GUI")
        self.root.geometry("1880x1020")

        self.image_path = Path(args.image).resolve()
        self.label_path = Path(args.labels).resolve()
        self.out_dir = Path(args.out_dir).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.class_id_to_name = self.load_class_map(args.class_map)

        self.canvas_width = args.canvas_width
        self.canvas_height = args.canvas_height
        self.line_width = 2
        self.font_size = 12
        self.click_box_expand = 4
        self.min_box_size = 3

        self.image_bgr = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if self.image_bgr is None:
            raise RuntimeError(f"Failed to read image: {self.image_path}")
        self.image_h, self.image_w = self.image_bgr.shape[:2]

        self.components: List[Component] = self.load_yolo_components(self.label_path)

        self.stage = "component"  # component | node

        # Component-stage state
        self.selected_class_id = tk.IntVar(value=0)
        self.delete_mode = tk.BooleanVar(value=False)
        self.selected_component_ids: set[str] = set()
        self.dragging = False
        self.drag_start_img: Optional[Tuple[float, float]] = None
        self.drag_end_img: Optional[Tuple[float, float]] = None

        # Node-stage state
        self.auto_nodes: List[AutoNode] = []
        self.auto_by_id: Dict[int, AutoNode] = {}
        self.label_map: Optional[np.ndarray] = None
        self.manual_by_id: Dict[int, ManualNode] = {}
        self.group_by_id: Dict[int, NodeGroup] = {}
        self.connections: List[Connection] = []
        self.conn_index: Dict[Tuple[int, str], int] = {}
        self.merge_mode = tk.BooleanVar(value=False)
        self.connect_mode = tk.BooleanVar(value=False)
        self.add_node_mode = tk.BooleanVar(value=False)
        self.show_all_connections = tk.BooleanVar(value=False)
        self.pending_merge_first: Optional[int] = None
        self.current_connect_node: Optional[int] = None
        self.selected_node_group_id: Optional[int] = None
        self.selected_connection_index: Optional[int] = None
        self.undo_stack: List[dict] = []
        self.next_manual_id = 1
        self.node_params = {
            "mask_pad": args.mask_pad,
            "blur": args.blur,
            "open_iter": args.open_iter,
            "close_iter": args.close_iter,
            "min_area": args.min_area,
            "connectivity": args.connectivity,
            "auto_conn_pad": args.auto_conn_pad,
            "auto_conn_min_pixels": args.auto_conn_min_pixels,
        }

        # Display geometry
        self.tk_image = None
        self.display_scale = 1.0
        self.display_offset_x = 0
        self.display_offset_y = 0
        self.display_img_w = 0
        self.display_img_h = 0

        self.build_layout()
        self.refresh_all()

    # ----------------------------
    # Setup / loading
    # ----------------------------
    def load_class_map(self, class_map_path: Optional[str]) -> Dict[int, str]:
        if not class_map_path:
            return dict(DEFAULT_CLASS_MAP)
        p = Path(class_map_path)
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {int(k): str(v) for k, v in data.items()}

    def load_yolo_components(self, label_path: Path) -> List[Component]:
        if not label_path.exists():
            raise FileNotFoundError(f"YOLO label file not found: {label_path}")

        components: List[Component] = []
        with label_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id = int(float(parts[0]))
                cx = float(parts[1]) * self.image_w
                cy = float(parts[2]) * self.image_h
                w = float(parts[3]) * self.image_w
                h = float(parts[4]) * self.image_h
                conf = float(parts[5]) if len(parts) >= 6 else 1.0
                x1 = int(round(cx - w / 2.0))
                y1 = int(round(cy - h / 2.0))
                x2 = int(round(cx + w / 2.0))
                y2 = int(round(cy + h / 2.0))
                x1, y1, x2, y2 = self.clamp_box(x1, y1, x2, y2)
                comp_id = f"comp_{idx:04d}"
                components.append(
                    Component(
                        component_id=comp_id,
                        class_id=cls_id,
                        class_name=self.class_id_to_name.get(cls_id, f"cls_{cls_id}"),
                        confidence=conf,
                        bbox_xyxy=[x1, y1, x2, y2],
                        bbox_xywh_abs=self.xyxy_to_xywh_abs(x1, y1, x2, y2),
                        source="yolo",
                    )
                )
        return components

    # ----------------------------
    # General utils
    # ----------------------------
    def set_status(self, text: str):
        self.status_label.config(text=text)

    def get_font(self, size=12):
        try:
            return ImageFont.truetype("arial.ttf", size)
        except Exception:
            return ImageFont.load_default()

    def xyxy_to_xywh_abs(self, x1, y1, x2, y2):
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        return [cx, cy, w, h]

    def xyxy_to_yolo_line(self, cls_id, x1, y1, x2, y2):
        cx, cy, w, h = self.xyxy_to_xywh_abs(x1, y1, x2, y2)
        return f"{cls_id} {cx/self.image_w:.6f} {cy/self.image_h:.6f} {w/self.image_w:.6f} {h/self.image_h:.6f}"

    def clamp_box(self, x1, y1, x2, y2):
        x1 = max(0, min(self.image_w - 1, x1))
        x2 = max(0, min(self.image_w - 1, x2))
        y1 = max(0, min(self.image_h - 1, y1))
        y2 = max(0, min(self.image_h - 1, y2))
        return x1, y1, x2, y2

    def clamp_image_coords(self, x, y):
        x = max(0, min(self.image_w - 1, x))
        y = max(0, min(self.image_h - 1, y))
        return x, y

    def fit_image_size(self, orig_w, orig_h, target_w, target_h):
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))
        return new_w, new_h, scale

    def canvas_to_image_coords(self, canvas_x, canvas_y):
        ix = (canvas_x - self.display_offset_x) / self.display_scale
        iy = (canvas_y - self.display_offset_y) / self.display_scale
        return self.clamp_image_coords(ix, iy)

    def point_in_xyxy(self, x, y, box, expand=0):
        x1, y1, x2, y2 = box
        return (x1 - expand) <= x <= (x2 + expand) and (y1 - expand) <= y <= (
            y2 + expand
        )

    def box_area_xyxy(self, box):
        x1, y1, x2, y2 = box
        return max(0, x2 - x1) * max(0, y2 - y1)

    def make_new_component_id(self):
        used = {c.component_id for c in self.components}
        k = 0
        while f"manual_{k}" in used:
            k += 1
        return f"manual_{k}"

    def get_component_by_id(self, component_id: str) -> Optional[Component]:
        for c in self.components:
            if c.component_id == component_id:
                return c
        return None

    def get_component_ordered(self) -> List[Component]:
        return list(self.components)

    def rebuild_connection_index(self):
        self.conn_index = {}
        for i, c in enumerate(self.connections):
            self.conn_index[(c.node_id, c.component_id)] = i

    # ----------------------------
    # Layout
    # ----------------------------
    def build_layout(self):
        outer = ttk.Frame(self.root)
        outer.pack(fill="both", expand=True)

        left = ttk.Frame(outer)
        left.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        topbar = ttk.Frame(left)
        topbar.pack(side="top", fill="x")

        self.stage_label = ttk.Label(
            topbar, text="Stage: Component Review", font=("Arial", 12, "bold")
        )
        self.stage_label.pack(side="left", padx=(0, 12))

        ttk.Button(topbar, text="Enter Node Stage", command=self.enter_node_stage).pack(
            side="left", padx=4
        )
        ttk.Button(
            topbar, text="Back To Component", command=self.back_to_component_stage
        ).pack(side="left", padx=4)
        ttk.Button(
            topbar, text="Re-run Auto Nodes", command=self.rerun_auto_nodes
        ).pack(side="left", padx=4)
        ttk.Button(topbar, text="Save Final", command=self.save_all_outputs).pack(
            side="left", padx=12
        )

        self.canvas = tk.Canvas(
            left, width=self.canvas_width, height=self.canvas_height, bg="gray20"
        )
        self.canvas.pack(side="top", fill="both", expand=True)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.canvas.bind("<ButtonPress-1>", self.on_left_press)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)

        right = ttk.Frame(outer, width=500)
        right.pack(side="right", fill="y", padx=8, pady=8)

        self.status_label = ttk.Label(right, text="Ready", foreground="#333")
        self.status_label.pack(anchor="w", pady=(0, 8))

        notebook = ttk.Notebook(right)
        notebook.pack(fill="both", expand=True)

        self.component_tab = ttk.Frame(notebook)
        self.node_tab = ttk.Frame(notebook)
        notebook.add(self.component_tab, text="Component Stage")
        notebook.add(self.node_tab, text="Node Stage")

        self.build_component_panel(self.component_tab)
        self.build_node_panel(self.node_tab)

        self.root.bind("<Delete>", self.on_delete_key)
        self.root.bind("<Escape>", lambda e: self.clear_transient_state())
        self.root.bind("<KeyPress-m>", lambda e: self.toggle_merge_mode())
        self.root.bind("<KeyPress-c>", lambda e: self.toggle_connect_mode())
        self.root.bind("<KeyPress-u>", lambda e: self.undo_node_action())

    def build_component_panel(self, parent):
        ttk.Label(parent, text="Review Panel", font=("Arial", 14, "bold")).pack(
            anchor="w", pady=(4, 8)
        )

        ctrl = ttk.Frame(parent)
        ctrl.pack(fill="x", pady=(0, 8))
        ttk.Button(
            ctrl,
            text="Change Selected Class",
            command=self.change_selected_component_class,
        ).pack(fill="x", pady=2)
        ttk.Button(
            ctrl,
            text="Clear Component Selection",
            command=self.clear_component_selection,
        ).pack(fill="x", pady=2)
        ttk.Checkbutton(ctrl, text="Delete Mode", variable=self.delete_mode).pack(
            anchor="w", pady=4
        )

        ttk.Label(parent, text="Target Class", font=("Arial", 12, "bold")).pack(
            anchor="w", pady=(8, 4)
        )
        class_frame = ttk.Frame(parent)
        class_frame.pack(fill="x", pady=(0, 8))
        for cls_id in sorted(self.class_id_to_name):
            ttk.Radiobutton(
                class_frame,
                text=f"{cls_id} -> {self.class_id_to_name[cls_id]}",
                value=cls_id,
                variable=self.selected_class_id,
            ).pack(anchor="w")

        ttk.Label(parent, text="Info", font=("Arial", 12, "bold")).pack(
            anchor="w", pady=(8, 4)
        )
        self.component_info_text = tk.Text(parent, width=56, height=22, wrap="word")
        self.component_info_text.pack(fill="both", expand=False, pady=(0, 8))

        ttk.Label(parent, text="Instructions", font=("Arial", 12, "bold")).pack(
            anchor="w", pady=(8, 4)
        )
        self.component_legend_text = tk.Text(parent, width=56, height=12, wrap="word")
        self.component_legend_text.pack(fill="both", expand=True)
        self.component_legend_text.insert(
            tk.END,
            "Component stage controls:\n"
            "- Left click on bbox: select / deselect.\n"
            "- Left drag on empty area: add bbox using selected class.\n"
            "- Delete Mode ON + click bbox: delete bbox.\n"
            "- Change Selected Class: change all selected bbox classes.\n"
            "- Enter Node Stage: use the current reviewed bbox set to auto-detect nodes.\n",
        )
        self.component_legend_text.config(state="disabled")

    def build_node_panel(self, parent):
        ttk.Label(parent, text="Node Annotation", font=("Arial", 14, "bold")).pack(
            anchor="w", pady=(4, 8)
        )

        ctrl = ttk.Frame(parent)
        ctrl.pack(fill="x", pady=(0, 8))
        ttk.Checkbutton(
            ctrl,
            text="Merge Mode (m)",
            variable=self.merge_mode,
            command=self.on_toggle_merge,
        ).pack(anchor="w", pady=2)
        ttk.Checkbutton(
            ctrl,
            text="Connect Mode (c)",
            variable=self.connect_mode,
            command=self.on_toggle_connect,
        ).pack(anchor="w", pady=2)
        ttk.Checkbutton(
            ctrl,
            text="Add Node Mode",
            variable=self.add_node_mode,
            command=self.on_toggle_add_node,
        ).pack(anchor="w", pady=2)
        ttk.Checkbutton(
            ctrl,
            text="Show All Connections",
            variable=self.show_all_connections,
            command=self.on_toggle_show_all_connections,
        ).pack(anchor="w", pady=2)
        ttk.Button(
            ctrl, text="Delete Selected Node", command=self.delete_selected_node
        ).pack(fill="x", pady=2)
        ttk.Button(
            ctrl,
            text="Delete Selected Connection",
            command=self.delete_selected_connection,
        ).pack(fill="x", pady=2)
        ttk.Button(
            ctrl, text="Undo Node Action (u)", command=self.undo_node_action
        ).pack(fill="x", pady=2)
        ttk.Button(ctrl, text="Clear Node Focus", command=self.reset_node_focus).pack(
            fill="x", pady=2
        )

        ttk.Label(parent, text="Node Summary", font=("Arial", 12, "bold")).pack(
            anchor="w", pady=(8, 4)
        )
        self.node_info_text = tk.Text(parent, width=56, height=12, wrap="word")
        self.node_info_text.pack(fill="x", pady=(0, 8))

        lists = ttk.Frame(parent)
        lists.pack(fill="both", expand=True)

        ttk.Label(lists, text="Nodes").grid(
            row=0, column=0, padx=4, pady=(0, 4), sticky="w"
        )
        ttk.Label(lists, text="Components").grid(
            row=0, column=1, padx=4, pady=(0, 4), sticky="w"
        )
        ttk.Label(lists, text="Connections").grid(
            row=0, column=2, padx=4, pady=(0, 4), sticky="w"
        )

        self.nodes_list = tk.Listbox(lists, width=18, height=18, exportselection=False)
        self.comps_list = tk.Listbox(lists, width=22, height=18, exportselection=False)
        self.conns_list = tk.Listbox(lists, width=28, height=18, exportselection=False)

        self.nodes_list.grid(row=1, column=0, padx=4, pady=4, sticky="nsew")
        self.comps_list.grid(row=1, column=1, padx=4, pady=4, sticky="nsew")
        self.conns_list.grid(row=1, column=2, padx=4, pady=4, sticky="nsew")

        lists.columnconfigure(0, weight=1)
        lists.columnconfigure(1, weight=1)
        lists.columnconfigure(2, weight=1)

        self.nodes_list.bind("<<ListboxSelect>>", self.on_select_node_from_list)
        self.comps_list.bind("<<ListboxSelect>>", self.on_select_component_from_list)
        self.conns_list.bind("<<ListboxSelect>>", self.on_select_connection_from_list)

    # ----------------------------
    # Component-stage operations
    # ----------------------------
    def find_component_at(self, x, y) -> Optional[str]:
        candidates = []
        for comp in self.components:
            if self.point_in_xyxy(x, y, comp.bbox_xyxy, expand=self.click_box_expand):
                candidates.append(
                    (self.box_area_xyxy(comp.bbox_xyxy), comp.component_id)
                )
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][1]

    def toggle_component_selection(self, component_id: str):
        if component_id in self.selected_component_ids:
            self.selected_component_ids.remove(component_id)
        else:
            self.selected_component_ids.add(component_id)

    def clear_component_selection(self):
        self.selected_component_ids.clear()
        self.refresh_all()

    def add_component(self, cls_id, x1, y1, x2, y2):
        x1, x2 = sorted([int(round(x1)), int(round(x2))])
        y1, y2 = sorted([int(round(y1)), int(round(y2))])
        x1, y1, x2, y2 = self.clamp_box(x1, y1, x2, y2)

        if x2 - x1 < self.min_box_size or y2 - y1 < self.min_box_size:
            return False

        new_id = self.make_new_component_id()
        self.components.append(
            Component(
                component_id=new_id,
                class_id=cls_id,
                class_name=self.class_id_to_name.get(cls_id, f"cls_{cls_id}"),
                confidence=1.0,
                bbox_xyxy=[x1, y1, x2, y2],
                bbox_xywh_abs=self.xyxy_to_xywh_abs(x1, y1, x2, y2),
                source="manual_add",
            )
        )
        self.set_status(f"Added component {new_id}")
        return True

    def delete_component_by_id(self, component_id: str):
        before = len(self.components)
        self.components = [c for c in self.components if c.component_id != component_id]
        if len(self.components) == before:
            return False
        self.selected_component_ids.discard(component_id)

        # Also remove stale connections if component deleted after returning from node stage
        self.connections = [
            c for c in self.connections if c.component_id != component_id
        ]
        self.rebuild_connection_index()
        self.set_status(f"Deleted component {component_id}")
        return True

    def change_selected_component_class(self):
        if not self.selected_component_ids:
            messagebox.showwarning("Warning", "No component bbox selected.")
            return
        new_cls_id = self.selected_class_id.get()
        new_name = self.class_id_to_name.get(new_cls_id, f"cls_{new_cls_id}")
        changed = 0
        for comp in self.components:
            if (
                comp.component_id in self.selected_component_ids
                and comp.class_id != new_cls_id
            ):
                comp.class_id = new_cls_id
                comp.class_name = new_name
                comp.source = "manual_class_change"
                changed += 1
        self.set_status(f"Changed class for {changed} component(s)")
        self.refresh_all()

    # ----------------------------
    # Node auto-detection
    # ----------------------------
    def get_component_bboxes_xywh(self):
        boxes = []
        for comp in self.components:
            x1, y1, x2, y2 = comp.bbox_xyxy
            boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes

    def mask_components_white(self, img_bgr: np.ndarray, bboxes_xywh, pad=2):
        out = img_bgr.copy()
        h, w = out.shape[:2]
        for x, y, bw, bh in bboxes_xywh:
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(w, x + bw + pad)
            y1 = min(h, y + bh + pad)
            out[y0:y1, x0:x1] = 255
        return out

    def make_wire_binary(self, masked_bgr, blur_ksize=3, open_iter=1, close_iter=2):
        gray = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2GRAY)
        if blur_ksize > 0:
            gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        if open_iter > 0:
            bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=open_iter)
        if close_iter > 0:
            bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=close_iter)
        return bw

    def find_auto_nodes_from_bw(self, bw, min_area=80, connectivity=8):
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(
            (bw > 0).astype(np.uint8), connectivity=connectivity
        )
        auto_nodes: List[AutoNode] = []
        label_map = np.zeros_like(labels, dtype=np.int32)
        new_id = 1
        for lab in range(1, num):
            x, y, w, h, area = stats[lab].tolist()
            if area < min_area:
                continue
            cx, cy = centroids[lab].tolist()
            auto_nodes.append(
                AutoNode(new_id, int(area), (x, y, w, h), (float(cx), float(cy)))
            )
            label_map[labels == lab] = new_id
            new_id += 1
        return auto_nodes, label_map

    def component_connection_ring_mask(self, comp: Component, pad: int) -> np.ndarray:
        x1, y1, x2, y2 = comp.bbox_xyxy
        outer_x1 = max(0, x1 - pad)
        outer_y1 = max(0, y1 - pad)
        outer_x2 = min(self.image_w - 1, x2 + pad)
        outer_y2 = min(self.image_h - 1, y2 + pad)

        mask = np.zeros((self.image_h, self.image_w), dtype=np.uint8)
        mask[outer_y1 : outer_y2 + 1, outer_x1 : outer_x2 + 1] = 255
        mask[y1 : y2 + 1, x1 : x2 + 1] = 0
        return mask

    def get_unconnected_node_ids(self):
        connected_ids = {c.node_id for c in self.connections}
        return [gid for gid in sorted(self.group_by_id) if gid not in connected_ids]

    def prune_unconnected_nodes(self, delete_manual=False):
        # 统计每个 node 连了多少个 component
        conn_count = {}
        for c in self.connections:
            conn_count[c.node_id] = conn_count.get(c.node_id, 0) + 1

        removed = []
        for gid in list(sorted(self.group_by_id)):
            # 现在要求：node 至少连接 2 个 component，否则删掉
            if conn_count.get(gid, 0) >= 2:
                continue

            group = self.group_by_id.get(gid)
            if group is None:
                continue

            is_manual_only = len(group.members_auto) == 0
            if is_manual_only and not delete_manual:
                continue

            for mid in group.members_manual:
                self.manual_by_id.pop(mid, None)
            self.group_by_id.pop(gid, None)
            removed.append(gid)

            if self.selected_node_group_id == gid:
                self.selected_node_group_id = None
            if self.pending_merge_first == gid:
                self.pending_merge_first = None
            if self.current_connect_node == gid:
                self.current_connect_node = None

        if removed:
            removed_set = set(removed)
            self.connections = [
                c for c in self.connections if c.node_id not in removed_set
            ]

            self.selected_connection_index = None
            self.rebuild_connection_index()

        return removed

    def auto_detect_component_connections(
        self, pad: int = 3, min_pixels: int = 2
    ) -> List[Connection]:
        if self.label_map is None:
            return []

        auto_connections: List[Connection] = []
        seen = set()
        for comp in self.components:
            ring_mask = self.component_connection_ring_mask(comp, pad)
            ring_node_ids = self.label_map[ring_mask > 0]
            if ring_node_ids.size == 0:
                continue

            node_ids, counts = np.unique(
                ring_node_ids[ring_node_ids > 0], return_counts=True
            )
            for node_id, count in zip(node_ids.tolist(), counts.tolist()):
                if count < min_pixels:
                    continue
                key = (int(node_id), comp.component_id)
                if key in seen or int(node_id) not in self.group_by_id:
                    continue
                auto_connections.append(
                    Connection(node_id=int(node_id), component_id=comp.component_id)
                )
                seen.add(key)
        return auto_connections

    def init_node_state_from_components(self, reset_existing=True):
        masked = self.mask_components_white(
            self.image_bgr,
            self.get_component_bboxes_xywh(),
            pad=self.node_params["mask_pad"],
        )
        bw = self.make_wire_binary(
            masked,
            blur_ksize=self.node_params["blur"],
            open_iter=self.node_params["open_iter"],
            close_iter=self.node_params["close_iter"],
        )
        self.auto_nodes, self.label_map = self.find_auto_nodes_from_bw(
            bw,
            min_area=self.node_params["min_area"],
            connectivity=self.node_params["connectivity"],
        )
        self.auto_by_id = {n.id: n for n in self.auto_nodes}

        if reset_existing:
            self.manual_by_id = {}
            self.connections = []
            self.undo_stack = []
        else:
            # Keep only connections pointing to existing components.
            existing_ids = {c.component_id for c in self.components}
            self.connections = [
                c for c in self.connections if c.component_id in existing_ids
            ]

        self.group_by_id = {
            n.id: NodeGroup(id=n.id, members_auto=[n.id], members_manual=[])
            for n in self.auto_nodes
        }

        if not reset_existing and self.manual_by_id:
            for mid in sorted(self.manual_by_id):
                self.group_by_id[mid] = NodeGroup(
                    id=mid, members_auto=[], members_manual=[mid]
                )

        auto_connections = self.auto_detect_component_connections(
            pad=self.node_params["auto_conn_pad"],
            min_pixels=self.node_params["auto_conn_min_pixels"],
        )
        if reset_existing or not self.connections:
            self.connections = auto_connections
        else:
            self.connections = self.unique_connections(
                self.connections + auto_connections
            )

        removed_unconnected = self.prune_unconnected_nodes(delete_manual=False)
        self.rebuild_connection_index()
        self.selected_node_group_id = None
        self.selected_connection_index = None
        self.pending_merge_first = None
        self.current_connect_node = None
        self.next_manual_id = max([n.id for n in self.auto_nodes], default=0) + 1
        while self.next_manual_id in self.manual_by_id:
            self.next_manual_id += 1
        # return removed_unconnected

    def enter_node_stage(self):
        self.init_node_state_from_components(reset_existing=False)
        self.stage = "node"
        self.set_status(
            f"Entered node stage. Auto-detected {len(self.auto_nodes)} nodes and {len(self.connections)} connections."
        )
        self.refresh_all()

    def back_to_component_stage(self):
        self.stage = "component"
        self.merge_mode.set(False)
        self.connect_mode.set(False)
        self.pending_merge_first = None
        self.current_connect_node = None
        self.selected_node_group_id = None
        self.selected_connection_index = None
        self.set_status("Back to component stage.")
        self.refresh_all()

    def rerun_auto_nodes(self):
        self.init_node_state_from_components(reset_existing=True)
        self.stage = "node"
        self.set_status(
            f"Auto detection re-ran: {len(self.auto_nodes)} nodes, {len(self.connections)} connections."
        )
        self.refresh_all()

    # ----------------------------
    # Node-stage helpers
    # ----------------------------
    def on_toggle_merge(self):
        if self.merge_mode.get():
            self.connect_mode.set(False)
            self.current_connect_node = None
            self.set_status("Merge Mode ON: click/select node A then node B")
        else:
            self.pending_merge_first = None
            self.set_status("Merge Mode OFF")
        self.refresh_all()

    def on_toggle_connect(self):
        if self.connect_mode.get():
            self.merge_mode.set(False)
            self.pending_merge_first = None
            self.set_status(
                "Connect Mode ON: select a node, then select/click a component"
            )
        else:
            self.current_connect_node = None
            self.set_status("Connect Mode OFF")
        self.refresh_all()

    def toggle_merge_mode(self):
        if self.stage != "node":
            return
        self.merge_mode.set(not self.merge_mode.get())
        self.on_toggle_merge()

    def toggle_connect_mode(self):
        if self.stage != "node":
            return
        self.connect_mode.set(not self.connect_mode.get())
        self.on_toggle_connect()

    def on_toggle_add_node(self):
        if self.add_node_mode.get():
            self.set_status(
                "Add Node Mode ON: click empty canvas to add manual node. Existing nodes are color-highlighted."
            )
            self.selected_node_group_id = None
            self.selected_connection_index = None
        else:
            self.set_status(
                "Add Node Mode OFF: empty canvas click now clears node focus."
            )
        self.refresh_all()

    def on_toggle_show_all_connections(self):
        if self.show_all_connections.get():
            self.set_status("Show All Connections ON")
        else:
            self.set_status("Show All Connections OFF")
        self.refresh_all()

    def reset_node_focus(self):
        self.selected_node_group_id = None
        self.selected_connection_index = None
        self.pending_merge_first = None
        if not self.connect_mode.get():
            self.current_connect_node = None
        self.nodes_list.selection_clear(0, tk.END)
        self.comps_list.selection_clear(0, tk.END)
        self.conns_list.selection_clear(0, tk.END)
        self.set_status("Node focus cleared")
        self.refresh_all()

    def clear_transient_state(self):
        self.dragging = False
        self.drag_start_img = None
        self.drag_end_img = None
        if self.stage == "node":
            self.reset_node_focus()
        else:
            self.clear_component_selection()

    def group_union_mask(self, group: NodeGroup) -> np.ndarray:
        mask = np.zeros((self.image_h, self.image_w), dtype=np.uint8)
        if self.label_map is not None:
            for aid in group.members_auto:
                mask[self.label_map == aid] = 255
        for mid in group.members_manual:
            m = self.manual_by_id.get(mid)
            if m is not None:
                cv2.circle(mask, (m.x, m.y), m.radius, 255, -1)
        return mask

    def find_group_at(self, x: int, y: int) -> Optional[int]:
        candidates = []
        for gid, group in self.group_by_id.items():
            mask = self.group_union_mask(group)
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] > 0:
                area = int(np.count_nonzero(mask))
                candidates.append((area, gid))
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][1]

    def add_manual_node(self, x: int, y: int):
        mid = self.next_manual_id
        self.next_manual_id += 1
        self.manual_by_id[mid] = ManualNode(id=mid, x=x, y=y)
        self.group_by_id[mid] = NodeGroup(id=mid, members_auto=[], members_manual=[mid])
        self.undo_stack.append({"op": "add_manual", "manual_id": mid})
        self.selected_node_group_id = mid
        self.set_status(f"Added manual node node{mid}")
        self.refresh_all()

    def delete_group(self, gid: int):
        group = self.group_by_id.get(gid)
        if group is None:
            return
        snapshot = {
            "op": "delete_group",
            "group": asdict(group),
            "manual_nodes": [
                asdict(self.manual_by_id[mid])
                for mid in group.members_manual
                if mid in self.manual_by_id
            ],
            "connections_removed": [
                asdict(c) for c in self.connections if c.node_id == gid
            ],
        }
        self.undo_stack.append(snapshot)
        for mid in group.members_manual:
            self.manual_by_id.pop(mid, None)
        self.group_by_id.pop(gid, None)
        self.connections = [c for c in self.connections if c.node_id != gid]
        self.rebuild_connection_index()
        if self.selected_node_group_id == gid:
            self.selected_node_group_id = None
        if self.pending_merge_first == gid:
            self.pending_merge_first = None
        if self.current_connect_node == gid:
            self.current_connect_node = None
        self.set_status(f"Deleted node{gid}")
        self.refresh_all()

    def merge_groups(self, gid_a: int, gid_b: int):
        if gid_a == gid_b:
            return
        ga = self.group_by_id.get(gid_a)
        gb = self.group_by_id.get(gid_b)
        if ga is None or gb is None:
            return
        before_a = asdict(ga)
        before_b = asdict(gb)
        moved = [asdict(c) for c in self.connections if c.node_id == gid_b]

        ga.members_auto = sorted(set(ga.members_auto + gb.members_auto))
        ga.members_manual = sorted(set(ga.members_manual + gb.members_manual))
        self.group_by_id.pop(gid_b, None)

        new_conns = []
        seen = set()
        for c in self.connections:
            nid = gid_a if c.node_id == gid_b else c.node_id
            key = (nid, c.component_id)
            if key not in seen:
                new_conns.append(Connection(node_id=nid, component_id=c.component_id))
                seen.add(key)
        self.connections = new_conns
        self.rebuild_connection_index()
        self.undo_stack.append(
            {
                "op": "merge_groups",
                "a_before": before_a,
                "b_before": before_b,
                "moved": moved,
            }
        )
        self.pending_merge_first = None
        self.selected_node_group_id = gid_a
        if self.current_connect_node == gid_b:
            self.current_connect_node = gid_a
        self.set_status(f"Merged node{gid_b} into node{gid_a}")
        self.refresh_all()

    def add_connection(self, node_id: int, component_id: str):
        if node_id not in self.group_by_id:
            return
        if self.get_component_by_id(component_id) is None:
            return
        key = (node_id, component_id)
        if key in self.conn_index:
            self.set_status(
                f"Connection already exists: node{node_id} -> {component_id}"
            )
            return
        conn = Connection(node_id=node_id, component_id=component_id)
        self.connections.append(conn)
        self.rebuild_connection_index()
        self.undo_stack.append({"op": "add_connection", "conn": asdict(conn)})
        self.selected_node_group_id = node_id
        self.set_status(f"Added connection node{node_id} -> {component_id}")
        self.refresh_all()

    def delete_selected_connection(self):
        if self.selected_connection_index is None:
            return
        idx = self.selected_connection_index
        if not (0 <= idx < len(self.connections)):
            return
        conn = self.connections[idx]
        self.undo_stack.append({"op": "delete_connection", "conn": asdict(conn)})
        del self.connections[idx]
        self.rebuild_connection_index()
        self.selected_connection_index = None
        self.set_status("Deleted selected connection")
        self.refresh_all()

    def delete_selected_node(self):
        if self.selected_node_group_id is None:
            return
        self.delete_group(self.selected_node_group_id)

    def undo_node_action(self):
        if self.stage != "node":
            return
        if not self.undo_stack:
            self.set_status("Undo stack empty")
            return
        act = self.undo_stack.pop()
        op = act.get("op")

        if op == "add_manual":
            mid = act["manual_id"]
            self.group_by_id.pop(mid, None)
            self.manual_by_id.pop(mid, None)

        elif op == "delete_group":
            g = NodeGroup(**act["group"])
            self.group_by_id[g.id] = g
            for md in act.get("manual_nodes", []):
                self.manual_by_id[md["id"]] = ManualNode(**md)
            for cd in act.get("connections_removed", []):
                self.connections.append(Connection(**cd))
            self.connections = self.unique_connections(self.connections)
            self.rebuild_connection_index()

        elif op == "merge_groups":
            a = NodeGroup(**act["a_before"])
            b = NodeGroup(**act["b_before"])
            self.group_by_id[a.id] = a
            self.group_by_id[b.id] = b
            moved_comp_ids = {m["component_id"] for m in act.get("moved", [])}
            self.connections = [
                c
                for c in self.connections
                if not (c.node_id == a.id and c.component_id in moved_comp_ids)
            ]
            for moved in act.get("moved", []):
                self.connections.append(
                    Connection(node_id=b.id, component_id=moved["component_id"])
                )
            self.connections = self.unique_connections(self.connections)
            self.rebuild_connection_index()

        elif op == "add_connection":
            cd = act["conn"]
            key = (cd["node_id"], cd["component_id"])
            if key in self.conn_index:
                del self.connections[self.conn_index[key]]
                self.rebuild_connection_index()

        elif op == "delete_connection":
            self.connections.append(Connection(**act["conn"]))
            self.connections = self.unique_connections(self.connections)
            self.rebuild_connection_index()

        self.selected_node_group_id = None
        self.selected_connection_index = None
        self.set_status("Undo completed")
        self.refresh_all()

    def unique_connections(self, conns: List[Connection]) -> List[Connection]:
        out = []
        seen = set()
        for c in conns:
            key = (c.node_id, c.component_id)
            if key not in seen:
                out.append(c)
                seen.add(key)
        return out

    # ----------------------------
    # Listbox handlers
    # ----------------------------
    def on_select_node_from_list(self, _event=None):
        if self.stage != "node":
            return
        sel = self.nodes_list.curselection()
        if not sel:
            return
        txt = self.nodes_list.get(sel[0])
        gid = int(txt.replace("node", ""))
        self.handle_node_selection(gid)

    def on_select_component_from_list(self, _event=None):
        if self.stage != "node":
            return
        sel = self.comps_list.curselection()
        if not sel:
            return
        comp = self.get_component_ordered()[sel[0]]
        self.handle_component_selection_in_node_stage(comp.component_id)

    def on_select_connection_from_list(self, _event=None):
        if self.stage != "node":
            return
        sel = self.conns_list.curselection()
        if not sel:
            return
        idx = sel[0]
        if 0 <= idx < len(self.connections):
            self.selected_connection_index = idx
            self.selected_node_group_id = self.connections[idx].node_id
            self.set_status(f"Selected connection {idx}")
            self.refresh_all()

    def handle_node_selection(self, gid: int):
        if self.merge_mode.get():
            if self.pending_merge_first is None:
                self.pending_merge_first = gid
                self.selected_node_group_id = gid
                self.set_status(f"Merge first node = node{gid}; now choose second node")
                self.refresh_all()
                return
            if gid == self.pending_merge_first:
                self.pending_merge_first = None
                self.set_status("Merge cancelled")
                self.refresh_all()
                return
            self.merge_groups(self.pending_merge_first, gid)
            return

        if self.connect_mode.get():
            self.current_connect_node = gid
            self.selected_node_group_id = gid
            self.set_status(f"Connect current node = node{gid}. Now pick a component.")
            self.refresh_all()
            return

        self.selected_node_group_id = gid
        self.selected_connection_index = None
        self.set_status(f"Selected node{gid}")
        self.refresh_all()

    def handle_component_selection_in_node_stage(self, component_id: str):
        if self.connect_mode.get():
            if self.current_connect_node is None:
                self.set_status("Connect Mode: select a node first")
                return
            self.add_connection(self.current_connect_node, component_id)
            return
        self.selected_component_ids = {component_id}
        self.set_status(f"Selected component {component_id} in node stage")
        self.refresh_all()

    # ----------------------------
    # Mouse / keyboard
    # ----------------------------
    def on_left_press(self, event):
        x, y = self.canvas_to_image_coords(event.x, event.y)

        if self.stage == "component":
            hit_id = self.find_component_at(int(x), int(y))
            if self.delete_mode.get():
                if hit_id is not None:
                    self.delete_component_by_id(hit_id)
                    self.refresh_all()
                return
            if hit_id is not None:
                self.toggle_component_selection(hit_id)
                self.set_status(f"Toggled component {hit_id}")
                self.refresh_all()
                return
            self.dragging = True
            self.drag_start_img = (x, y)
            self.drag_end_img = (x, y)
            return

        # Node stage
        hit_group = self.find_group_at(int(x), int(y))
        hit_comp = self.find_component_at(int(x), int(y))

        if hit_group is not None:
            self.handle_node_selection(hit_group)
            return

        if hit_comp is not None:
            self.handle_component_selection_in_node_stage(hit_comp)
            return

        if self.connect_mode.get():
            self.set_status(
                "Connect Mode ON: select node and component, empty click ignored"
            )
            return

        if self.add_node_mode.get():
            self.add_manual_node(int(x), int(y))
            return

        self.reset_node_focus()

    def on_left_drag(self, event):
        if self.stage != "component":
            return
        if not self.dragging:
            return
        x, y = self.canvas_to_image_coords(event.x, event.y)
        self.drag_end_img = (x, y)
        x1, y1 = self.drag_start_img
        x2, y2 = self.drag_end_img
        self.update_canvas(draft_box=[x1, y1, x2, y2])

    def on_left_release(self, event):
        if self.stage != "component":
            return
        if not self.dragging:
            return
        x, y = self.canvas_to_image_coords(event.x, event.y)
        self.drag_end_img = (x, y)
        x1, y1 = self.drag_start_img
        x2, y2 = self.drag_end_img
        added = self.add_component(self.selected_class_id.get(), x1, y1, x2, y2)
        self.dragging = False
        self.drag_start_img = None
        self.drag_end_img = None
        if added:
            self.refresh_all()
        else:
            self.update_canvas()

    def on_delete_key(self, _event=None):
        if self.stage == "component":
            for cid in list(self.selected_component_ids):
                self.delete_component_by_id(cid)
            self.refresh_all()
            return
        if self.selected_connection_index is not None:
            self.delete_selected_connection()
        else:
            self.delete_selected_node()

    # ----------------------------
    # Rendering
    # ----------------------------
    def darken(self, img, alpha=0.25):
        return (img.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)

    def draw_label_pil(self, draw, x1, y1, text, bg_color, fg_color, font, alpha=120):
        bbox = draw.textbbox((x1, y1), text, font=font)
        tx1, ty1, tx2, ty2 = bbox
        pad_x, pad_y = 4, 2
        bg = [tx1 - pad_x, ty1 - pad_y, tx2 + pad_x, ty2 + pad_y]
        draw.rectangle(bg, fill=(*bg_color, alpha))
        draw.text((x1, y1), text, fill=(*fg_color, 255), font=font)

    def draw_components_overlay_pil(self, pil_img: Image.Image, draft_box=None):
        overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        font = self.get_font(self.font_size)
        for comp in self.components:
            x1, y1, x2, y2 = comp.bbox_xyxy
            if comp.component_id in self.selected_component_ids:
                outline = (255, 255, 0, 220)
                width = self.line_width + 2
            else:
                outline = (0, 255, 0, 140)
                width = self.line_width
            draw.rectangle([x1, y1, x2, y2], outline=outline, width=width)
            text = f"{comp.class_id}:{comp.class_name}"
            self.draw_label_pil(
                draw, x1, max(0, y1 - 20), text, (0, 255, 0), (0, 0, 0), font, alpha=120
            )
        if draft_box is not None:
            x1, y1, x2, y2 = draft_box
            draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 0, 180), width=2)
        return Image.alpha_composite(pil_img.convert("RGBA"), overlay).convert("RGB")

    def draw_group_outline_and_id(
        self, vis, group: NodeGroup, color=(0, 255, 0), thickness=1
    ):
        mask = self.group_union_mask(group)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
        cv2.drawContours(vis, contours, -1, color, thickness)
        x, y, w, h = cv2.boundingRect(np.vstack(contours))
        cv2.putText(
            vis,
            str(group.id),
            (x + 2, max(12, y + 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )

    def draw_component_node_mode(
        self, vis, comp: Component, draw_bbox=False, color=(80, 80, 80)
    ):
        x1, y1, x2, y2 = comp.bbox_xyxy
        if draw_bbox:
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
        tx = x1 + 2
        ty = max(12, y1 - 4)
        cv2.putText(
            vis,
            comp.component_id,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            color,
            1,
            cv2.LINE_AA,
        )

    def get_group_anchor_point(self, group: NodeGroup) -> Tuple[int, int]:
        if group.members_auto:
            pts = []
            for aid in group.members_auto:
                node = self.auto_by_id.get(aid)
                if node is not None:
                    pts.append(node.centroid_xy)
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                return int(round(sum(xs) / len(xs))), int(round(sum(ys) / len(ys)))
        if group.members_manual:
            pts = []
            for mid in group.members_manual:
                m = self.manual_by_id.get(mid)
                if m is not None:
                    pts.append((m.x, m.y))
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                return int(round(sum(xs) / len(xs))), int(round(sum(ys) / len(ys)))
        mask = self.group_union_mask(group)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return 0, 0
        return int(round(xs.mean())), int(round(ys.mean()))

    def get_component_marker_positions(
        self, component_to_nodes: Dict[str, List[int]]
    ) -> Dict[Tuple[str, int], Tuple[int, int]]:
        positions = {}
        step = 14
        for comp in self.components:
            node_ids = component_to_nodes.get(comp.component_id, [])
            if not node_ids:
                continue
            x1, y1, x2, y2 = comp.bbox_xyxy
            base_x = min(self.image_w - 10, x2 + 10)
            base_y = max(10, y1 + 10)
            for i, nid in enumerate(node_ids):
                cy = min(self.image_h - 10, base_y + i * step)
                positions[(comp.component_id, nid)] = (base_x, cy)
        return positions

    def get_node_color_bgr(self, node_id: int) -> Tuple[int, int, int]:
        palette = [
            (80, 80, 255),
            (80, 200, 80),
            (255, 170, 40),
            (220, 80, 220),
            (255, 255, 80),
            (80, 220, 220),
            (160, 120, 255),
            (120, 255, 160),
            (255, 120, 120),
            (120, 180, 255),
        ]
        return palette[(max(1, node_id) - 1) % len(palette)]

    def build_connection_maps(self):
        node_to_components: Dict[int, List[str]] = {}
        component_to_nodes: Dict[str, List[int]] = {}
        for conn in self.connections:
            node_to_components.setdefault(conn.node_id, []).append(conn.component_id)
            component_to_nodes.setdefault(conn.component_id, []).append(conn.node_id)
        for k in node_to_components:
            node_to_components[k] = sorted(set(node_to_components[k]))
        for k in component_to_nodes:
            component_to_nodes[k] = sorted(set(component_to_nodes[k]))
        return node_to_components, component_to_nodes

    def draw_connection_markers(
        self,
        vis,
        component_to_nodes: Dict[str, List[int]],
        selected_node_id: Optional[int] = None,
    ):
        marker_r = 5
        positions = self.get_component_marker_positions(component_to_nodes)
        for comp in self.components:
            node_ids = component_to_nodes.get(comp.component_id, [])
            if not node_ids:
                continue
            if selected_node_id is not None:
                node_ids = [nid for nid in node_ids if nid == selected_node_id]
                if not node_ids:
                    continue
            for nid in node_ids:
                cx, cy = positions[(comp.component_id, nid)]
                color = self.get_node_color_bgr(nid)
                cv2.circle(vis, (cx, cy), marker_r, color, -1)
                cv2.circle(vis, (cx, cy), marker_r + 1, (0, 0, 0), 1)
                cv2.putText(
                    vis,
                    str(nid),
                    (cx + 8, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.32,
                    color,
                    1,
                    cv2.LINE_AA,
                )

    def draw_connection_lines(
        self,
        vis,
        component_to_nodes: Dict[str, List[int]],
        selected_node_id: Optional[int] = None,
    ):
        positions = self.get_component_marker_positions(component_to_nodes)
        for comp in self.components:
            node_ids = component_to_nodes.get(comp.component_id, [])
            if selected_node_id is not None:
                node_ids = [nid for nid in node_ids if nid == selected_node_id]
            for nid in node_ids:
                group = self.group_by_id.get(nid)
                if group is None:
                    continue
                start = self.get_group_anchor_point(group)
                end = positions.get((comp.component_id, nid))
                if end is None:
                    continue
                color = self.get_node_color_bgr(nid)
                cv2.line(vis, start, end, color, 1, cv2.LINE_AA)

    def render_component_stage(self, draft_box=None):
        pil = Image.fromarray(cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB))
        return self.draw_components_overlay_pil(pil, draft_box=draft_box)

    def render_node_stage(self):
        node_to_components, component_to_nodes = self.build_connection_maps()

        highlight_all_nodes = (
            self.add_node_mode.get() or self.show_all_connections.get()
        )
        has_focus = (
            highlight_all_nodes
            or self.selected_node_group_id is not None
            or bool(self.selected_component_ids)
        )
        vis = (
            self.darken(self.image_bgr, alpha=0.25)
            if has_focus
            else self.image_bgr.copy()
        )
        base_text_color = (155, 155, 155) if has_focus else (90, 90, 90)

        for comp in self.components:
            self.draw_component_node_mode(
                vis, comp, draw_bbox=False, color=base_text_color
            )

        for gid in sorted(self.group_by_id):
            if highlight_all_nodes:
                color = self.get_node_color_bgr(gid)
            else:
                color = (
                    self.get_node_color_bgr(gid)
                    if node_to_components.get(gid)
                    else (0, 255, 0)
                )
            thickness = 2 if gid == self.selected_node_group_id else 1
            self.draw_group_outline_and_id(
                vis, self.group_by_id[gid], color=color, thickness=thickness
            )

        if self.show_all_connections.get():
            self.draw_connection_lines(vis, component_to_nodes)
            self.draw_connection_markers(vis, component_to_nodes)
        elif self.selected_node_group_id is not None:
            self.draw_connection_lines(
                vis, component_to_nodes, selected_node_id=self.selected_node_group_id
            )
            self.draw_connection_markers(
                vis, component_to_nodes, selected_node_id=self.selected_node_group_id
            )
        elif not has_focus:
            self.draw_connection_markers(vis, component_to_nodes)

        if (
            self.selected_node_group_id is not None
            and self.selected_node_group_id in self.group_by_id
        ):
            node_color = self.get_node_color_bgr(self.selected_node_group_id)
            self.draw_group_outline_and_id(
                vis,
                self.group_by_id[self.selected_node_group_id],
                color=node_color,
                thickness=2,
            )
            for comp_id in node_to_components.get(self.selected_node_group_id, []):
                comp = self.get_component_by_id(comp_id)
                if comp is not None:
                    self.draw_component_node_mode(
                        vis, comp, draw_bbox=True, color=node_color
                    )

        for cid in self.selected_component_ids:
            comp = self.get_component_by_id(cid)
            if comp is not None:
                self.draw_component_node_mode(
                    vis, comp, draw_bbox=True, color=(0, 255, 255)
                )
                for nid in component_to_nodes.get(cid, []):
                    color = self.get_node_color_bgr(nid)
                    positions = self.get_component_marker_positions(component_to_nodes)
                    if (cid, nid) in positions:
                        cx, cy = positions[(cid, nid)]
                        cv2.circle(vis, (cx, cy), 5, color, -1)
                        cv2.circle(vis, (cx, cy), 6, (0, 0, 0), 1)

        return Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

    def update_canvas(self, draft_box=None):
        rendered = (
            self.render_component_stage(draft_box)
            if self.stage == "component"
            else self.render_node_stage()
        )

        self.root.update_idletasks()
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w <= 1:
            canvas_w = self.canvas_width
        if canvas_h <= 1:
            canvas_h = self.canvas_height

        new_w, new_h, scale = self.fit_image_size(
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
            self.display_offset_x,
            self.display_offset_y,
            image=self.tk_image,
            anchor="nw",
        )

    def on_canvas_configure(self, event):
        if self.tk_image is not None:
            self.update_canvas()

    # ----------------------------
    # Info panel refresh
    # ----------------------------
    def refresh_component_info(self):
        txt = self.component_info_text
        txt.config(state="normal")
        txt.delete("1.0", tk.END)
        txt.insert(tk.END, f"Image: {self.image_path.name}\n")
        txt.insert(tk.END, f"Path: {self.image_path}\n")
        txt.insert(tk.END, f"Size: {self.image_w} x {self.image_h}\n\n")
        txt.insert(tk.END, f"Stage: {self.stage}\n")
        txt.insert(tk.END, f"Component count: {len(self.components)}\n")
        txt.insert(tk.END, f"Selected count: {len(self.selected_component_ids)}\n\n")

        class_counts: Dict[str, int] = {}
        for c in self.components:
            class_counts[c.class_name] = class_counts.get(c.class_name, 0) + 1
        txt.insert(tk.END, "Class histogram:\n")
        for name in sorted(class_counts):
            txt.insert(tk.END, f"  {name}: {class_counts[name]}\n")

        if self.selected_component_ids:
            txt.insert(tk.END, "\nSelected components:\n")
            for cid in sorted(self.selected_component_ids):
                comp = self.get_component_by_id(cid)
                if comp is not None:
                    txt.insert(
                        tk.END,
                        f"  {cid} | {comp.class_id}:{comp.class_name} | bbox={comp.bbox_xyxy}\n",
                    )
        txt.config(state="disabled")

    def refresh_node_info(self):
        txt = self.node_info_text
        txt.config(state="normal")
        txt.delete("1.0", tk.END)
        txt.insert(tk.END, f"Stage: {self.stage}\n")
        txt.insert(tk.END, f"Auto nodes: {len(self.auto_nodes)}\n")
        txt.insert(tk.END, f"Manual nodes: {len(self.manual_by_id)}\n")
        txt.insert(tk.END, f"Node groups: {len(self.group_by_id)}\n")
        txt.insert(tk.END, f"Connections: {len(self.connections)}\n")
        txt.insert(tk.END, f"Undo stack size: {len(self.undo_stack)}\n\n")
        txt.insert(tk.END, "Node detection params:\n")
        for k, v in self.node_params.items():
            txt.insert(tk.END, f"  {k}: {v}\n")

        if (
            self.selected_node_group_id is not None
            and self.selected_node_group_id in self.group_by_id
        ):
            g = self.group_by_id[self.selected_node_group_id]
            txt.insert(tk.END, f"\nSelected node: node{g.id}\n")
            txt.insert(tk.END, f"  members_auto: {g.members_auto}\n")
            txt.insert(tk.END, f"  members_manual: {g.members_manual}\n")
            attached = [c.component_id for c in self.connections if c.node_id == g.id]
            txt.insert(tk.END, f"  attached components: {attached}\n")

        if self.connect_mode.get():
            txt.insert(
                tk.END, f"\nConnect mode current node: {self.current_connect_node}\n"
            )
        if self.merge_mode.get():
            txt.insert(tk.END, f"Merge mode first node: {self.pending_merge_first}\n")
        txt.config(state="disabled")

    def refresh_node_lists(self):
        self.nodes_list.delete(0, tk.END)
        for gid in sorted(self.group_by_id):
            self.nodes_list.insert(tk.END, f"node{gid}")

        self.comps_list.delete(0, tk.END)
        for comp in self.get_component_ordered():
            self.comps_list.insert(tk.END, f"{comp.component_id} | {comp.class_name}")

        self.conns_list.delete(0, tk.END)
        for conn in self.connections:
            self.conns_list.insert(tk.END, f"node{conn.node_id} -> {conn.component_id}")

    def refresh_all(self):
        self.stage_label.config(
            text=f"Stage: {'Component Review' if self.stage == 'component' else 'Node Annotation'}"
        )
        self.refresh_component_info()
        self.refresh_node_info()
        self.refresh_node_lists()
        self.update_canvas()

    # ----------------------------
    # Saving
    # ----------------------------
    def components_payload(self):
        return {
            "image_name": self.image_path.name,
            "image_path": str(self.image_path),
            "image_width": self.image_w,
            "image_height": self.image_h,
            "components": [asdict(c) for c in self.components],
        }

    def node_payload(self):
        return {
            "image_name": self.image_path.name,
            "image_path": str(self.image_path),
            "params": self.node_params,
            "auto_nodes": [asdict(n) for n in self.auto_nodes],
            "manual_nodes": [asdict(m) for m in self.manual_by_id.values()],
            "node_groups": [
                {
                    "id": gid,
                    "members_auto": group.members_auto,
                    "members_manual": group.members_manual,
                }
                for gid, group in sorted(self.group_by_id.items(), key=lambda x: x[0])
            ],
            "connections": [asdict(c) for c in self.connections],
        }

    def save_updated_yolo_labels(self, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for comp in self.components:
                x1, y1, x2, y2 = comp.bbox_xyxy
                f.write(self.xyxy_to_yolo_line(comp.class_id, x1, y1, x2, y2) + "\n")

    def save_incidence_matrix_csv(self, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        node_ids = sorted(self.group_by_id)
        comps = self.get_component_ordered()
        conn_set = {(c.node_id, c.component_id) for c in self.connections}
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            header = ["node_id"] + [c.component_id for c in comps]
            writer.writerow(header)
            for nid in node_ids:
                row = [f"node{nid}"]
                for comp in comps:
                    row.append(1 if (nid, comp.component_id) in conn_set else 0)
                writer.writerow(row)

    def save_all_outputs(self):
        unconnected_node_ids = self.get_unconnected_node_ids()
        if unconnected_node_ids:
            node_text = ", ".join([f"node{nid}" for nid in unconnected_node_ids[:12]])
            if len(unconnected_node_ids) > 12:
                node_text += ", ..."
            proceed = messagebox.askyesno(
                "Warning: Unconnected Nodes",
                "There are node(s) on the canvas with no component connection:\n"
                f"{node_text}\n\n"
                "These nodes will still be saved unless you go back and remove or connect them. Continue saving?",
            )
            if not proceed:
                self.set_status("Save cancelled: unconnected nodes still exist.")
                return
        components_json = (
            self.out_dir / f"{self.image_path.stem}_reviewed_components.json"
        )
        node_json = self.out_dir / f"{self.image_path.stem}_node_annotation.json"
        incidence_csv = self.out_dir / f"{self.image_path.stem}_incidence_matrix.csv"
        updated_yolo = self.out_dir / f"{self.image_path.stem}_reviewed.txt"

        with components_json.open("w", encoding="utf-8") as f:
            json.dump(self.components_payload(), f, indent=2, ensure_ascii=False)

        with node_json.open("w", encoding="utf-8") as f:
            json.dump(self.node_payload(), f, indent=2, ensure_ascii=False)

        self.save_updated_yolo_labels(updated_yolo)
        self.save_incidence_matrix_csv(incidence_csv)

        self.set_status(f"Saved outputs to {self.out_dir}")
        messagebox.showinfo(
            "Saved",
            "Saved:\n"
            f"- {components_json}\n"
            f"- {node_json}\n"
            f"- {incidence_csv}\n"
            f"- {updated_yolo}",
        )


def main():
    ap = argparse.ArgumentParser(
        description="Integrated component-review + node-annotation GUI"
    )
    ap.add_argument("--image", required=True, help="Path to the input image")
    ap.add_argument("--labels", required=True, help="Path to the YOLO txt label file")
    ap.add_argument(
        "--out_dir", required=True, help="Directory to save reviewed outputs"
    )
    ap.add_argument(
        "--class_map",
        default=None,
        help="Optional JSON file mapping class id to class name",
    )
    ap.add_argument("--canvas_width", type=int, default=1100)
    ap.add_argument("--canvas_height", type=int, default=760)
    ap.add_argument("--mask_pad", type=int, default=2)
    ap.add_argument("--blur", type=int, default=3)
    ap.add_argument("--open_iter", type=int, default=1)
    ap.add_argument("--close_iter", type=int, default=2)
    ap.add_argument("--min_area", type=int, default=80)
    ap.add_argument("--connectivity", type=int, default=8)
    ap.add_argument("--auto_conn_pad", type=int, default=3)
    ap.add_argument("--auto_conn_min_pixels", type=int, default=2)
    args = ap.parse_args()

    root = tk.Tk()
    app = IntegratedCircuitGUI(root, args)
    app.refresh_all()
    root.mainloop()


if __name__ == "__main__":
    main()
