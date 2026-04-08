import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk


PIN_SCHEMA: Dict[str, List[str]] = {
    "cap": ["terminal_1", "terminal_2"],
    "inductor": ["terminal_1", "terminal_2"],
    "resistor": ["terminal_1", "terminal_2"],
    "swi_ideal": ["terminal_1", "terminal_2"],
    "diode": ["anode", "cathode"],
    "battery": ["positive", "negative"],
    "ac_src": ["positive", "negative"],
    "volt_src": ["positive", "negative"],
    "curr_src": ["positive", "negative"],
}

AUTO_PIN_CLASSES = {"cap", "inductor", "resistor", "swi_ideal"}


@dataclass
class Component:
    component_id: str
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: List[int]
    bbox_xywh_abs: List[float]
    source: str = "loaded"


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


@dataclass(frozen=True)
class PinConnection:
    node_id: int
    component_id: str
    pin_name: str
    source: str = "auto"


class PinAnnotationGUI:
    def __init__(self, root: tk.Tk, args: argparse.Namespace):
        self.root = root
        self.args = args
        self.root.title("Circuit Pin Annotation GUI")
        self.root.geometry("1880x1020")

        self.image_path = Path(args.image).resolve()
        self.components_json = Path(args.components_json).resolve()
        self.node_json = Path(args.node_json).resolve()
        self.out_dir = Path(args.out_dir).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.image_bgr = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if self.image_bgr is None:
            raise RuntimeError(f"Failed to read image: {self.image_path}")
        self.image_h, self.image_w = self.image_bgr.shape[:2]

        self.canvas_width = args.canvas_width
        self.canvas_height = args.canvas_height
        self.click_box_expand = 4

        self.components = self.load_components_json(self.components_json)
        (
            self.auto_nodes,
            self.auto_by_id,
            self.manual_by_id,
            self.group_by_id,
            self.connections,
        ) = self.load_node_json(self.node_json)

        self.pin_connections: List[PinConnection] = []
        self.pin_conn_index: Dict[Tuple[int, str, str], int] = {}

        self.selected_component_ids: set[str] = set()
        self.selected_node_group_id: Optional[int] = None
        self.selected_pin_connection_index: Optional[int] = None
        self.current_assign_node: Optional[int] = None
        self.current_assign_component_id: Optional[str] = None

        self.pin_name_var = tk.StringVar(value="")
        self.show_all_connections = tk.BooleanVar(value=True)
        self.show_all_pins = tk.BooleanVar(value=True)

        self.tk_image = None
        self.display_scale = 1.0
        self.display_offset_x = 0
        self.display_offset_y = 0
        self.display_img_w = 0
        self.display_img_h = 0

        self.auto_generate_pin_connections(reset_existing=True)
        self.build_layout()
        self.refresh_all()

    # ----------------------------
    # Loading
    # ----------------------------
    def load_components_json(self, path: Path) -> List[Component]:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return [Component(**c) for c in data.get("components", [])]

    def load_node_json(self, path: Path):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        auto_nodes = [
            AutoNode(
                id=int(n["id"]),
                area=int(n["area"]),
                bbox_xywh=tuple(n["bbox_xywh"]),
                centroid_xy=tuple(n["centroid_xy"]),
            )
            for n in data.get("auto_nodes", [])
        ]
        auto_by_id = {n.id: n for n in auto_nodes}
        manual_by_id = {int(m["id"]): ManualNode(**m) for m in data.get("manual_nodes", [])}
        group_by_id = {
            int(g["id"]): NodeGroup(
                id=int(g["id"]),
                members_auto=list(g.get("members_auto", [])),
                members_manual=list(g.get("members_manual", [])),
            )
            for g in data.get("node_groups", [])
        }
        connections = [Connection(**c) for c in data.get("connections", [])]
        return auto_nodes, auto_by_id, manual_by_id, group_by_id, connections

    # ----------------------------
    # Utility
    # ----------------------------
    def set_status(self, text: str):
        self.status_label.config(text=text)

    def fit_image_size(self, orig_w, orig_h, target_w, target_h):
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))
        return new_w, new_h, scale

    def clamp_image_coords(self, x, y):
        x = max(0, min(self.image_w - 1, x))
        y = max(0, min(self.image_h - 1, y))
        return x, y

    def canvas_to_image_coords(self, canvas_x, canvas_y):
        ix = (canvas_x - self.display_offset_x) / self.display_scale
        iy = (canvas_y - self.display_offset_y) / self.display_scale
        return self.clamp_image_coords(ix, iy)

    def point_in_xyxy(self, x, y, box, expand=0):
        x1, y1, x2, y2 = box
        return (x1 - expand) <= x <= (x2 + expand) and (y1 - expand) <= y <= (y2 + expand)

    def box_area_xyxy(self, box):
        x1, y1, x2, y2 = box
        return max(0, x2 - x1) * max(0, y2 - y1)

    def euclidean(self, p1, p2):
        return float(((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5)

    def rebuild_pin_connection_index(self):
        self.pin_conn_index = {}
        for i, pc in enumerate(self.pin_connections):
            self.pin_conn_index[(pc.node_id, pc.component_id, pc.pin_name)] = i

    def unique_pin_connections(self, pin_conns: List[PinConnection]) -> List[PinConnection]:
        out = []
        seen = set()
        for pc in pin_conns:
            key = (pc.node_id, pc.component_id, pc.pin_name)
            if key not in seen:
                out.append(pc)
                seen.add(key)
        return out

    def get_pin_schema(self, class_name: str) -> List[str]:
        return PIN_SCHEMA.get(class_name, ["pin_1", "pin_2"])

    def supports_auto_terminal_generation(self, comp: Component) -> bool:
        return comp.class_name in AUTO_PIN_CLASSES

    def get_component_by_id(self, component_id: str) -> Optional[Component]:
        for c in self.components:
            if c.component_id == component_id:
                return c
        return None

    def get_component_ordered(self) -> List[Component]:
        return list(self.components)

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

    def get_group_anchor_point(self, group: NodeGroup) -> Tuple[int, int]:
        pts = []
        for aid in group.members_auto:
            node = self.auto_by_id.get(aid)
            if node is not None:
                pts.append(node.centroid_xy)
        for mid in group.members_manual:
            node = self.manual_by_id.get(mid)
            if node is not None:
                pts.append((node.x, node.y))
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            return int(round(sum(xs) / len(xs))), int(round(sum(ys) / len(ys)))
        return 0, 0

    def find_group_at(self, x: int, y: int) -> Optional[int]:
        candidates = []
        for gid, group in self.group_by_id.items():
            gx, gy = self.get_group_anchor_point(group)
            if self.euclidean((x, y), (gx, gy)) <= 14:
                candidates.append((0.0, gid))
            for aid in group.members_auto:
                node = self.auto_by_id.get(aid)
                if node is None:
                    continue
                bx, by, bw, bh = node.bbox_xywh
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    candidates.append((0.1, gid))
            for mid in group.members_manual:
                node = self.manual_by_id.get(mid)
                if node is None:
                    continue
                if self.euclidean((x, y), (node.x, node.y)) <= node.radius + 4:
                    candidates.append((0.0, gid))
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][1]

    def find_component_at(self, x, y) -> Optional[str]:
        candidates = []
        for comp in self.components:
            if self.point_in_xyxy(x, y, comp.bbox_xyxy, expand=self.click_box_expand):
                candidates.append((self.box_area_xyxy(comp.bbox_xyxy), comp.component_id))
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][1]

    # ----------------------------
    # Pin generation / editing
    # ----------------------------
    def infer_two_terminal_pin_positions(self, comp: Component):
        x1, y1, x2, y2 = comp.bbox_xyxy
        w = x2 - x1
        h = y2 - y1
        if w >= h:
            return {
                "terminal_1": (x1, (y1 + y2) / 2.0),
                "terminal_2": (x2, (y1 + y2) / 2.0),
            }
        return {
            "terminal_1": ((x1 + x2) / 2.0, y1),
            "terminal_2": ((x1 + x2) / 2.0, y2),
        }

    def auto_generate_pin_connections(self, reset_existing: bool = True):
        _, component_to_nodes = self.build_connection_maps()
        generated: List[PinConnection] = []

        for comp in self.components:
            if not self.supports_auto_terminal_generation(comp):
                continue
            node_ids = component_to_nodes.get(comp.component_id, [])
            if len(node_ids) != 2:
                continue

            pin_pos = self.infer_two_terminal_pin_positions(comp)
            n1, n2 = node_ids
            g1 = self.group_by_id.get(n1)
            g2 = self.group_by_id.get(n2)
            if g1 is None or g2 is None:
                continue
            p1 = self.get_group_anchor_point(g1)
            p2 = self.get_group_anchor_point(g2)
            t1 = pin_pos["terminal_1"]
            t2 = pin_pos["terminal_2"]

            assign_a = self.euclidean(p1, t1) + self.euclidean(p2, t2)
            assign_b = self.euclidean(p1, t2) + self.euclidean(p2, t1)

            if assign_a <= assign_b:
                generated.append(PinConnection(n1, comp.component_id, "terminal_1", "auto"))
                generated.append(PinConnection(n2, comp.component_id, "terminal_2", "auto"))
            else:
                generated.append(PinConnection(n1, comp.component_id, "terminal_2", "auto"))
                generated.append(PinConnection(n2, comp.component_id, "terminal_1", "auto"))

        self.pin_connections = self.unique_pin_connections(generated if reset_existing else self.pin_connections + generated)
        self.rebuild_pin_connection_index()

    def get_component_pin_positions(self, comp: Component) -> Dict[str, Tuple[int, int]]:
        x1, y1, x2, y2 = comp.bbox_xyxy
        schema = self.get_pin_schema(comp.class_name)
        if len(schema) == 2:
            w = x2 - x1
            h = y2 - y1
            if w >= h:
                return {
                    schema[0]: (int(round(x1)), int(round((y1 + y2) / 2.0))),
                    schema[1]: (int(round(x2)), int(round((y1 + y2) / 2.0))),
                }
            return {
                schema[0]: (int(round((x1 + x2) / 2.0)), int(round(y1))),
                schema[1]: (int(round((x1 + x2) / 2.0)), int(round(y2))),
            }
        cx = int(round((x1 + x2) / 2.0))
        cy = int(round((y1 + y2) / 2.0))
        offsets = [(-14, 0), (14, 0), (0, -14), (0, 14), (-14, -14), (14, 14)]
        return {pin: (cx + offsets[i % len(offsets)][0], cy + offsets[i % len(offsets)][1]) for i, pin in enumerate(schema)}

    def add_or_update_pin_connection(self, node_id: int, component_id: str, pin_name: str, source: str = "manual"):
        comp = self.get_component_by_id(component_id)
        if comp is None:
            self.set_status("Component not found")
            return
        valid_pins = self.get_pin_schema(comp.class_name)
        if pin_name not in valid_pins:
            self.set_status(f"Invalid pin '{pin_name}' for class {comp.class_name}")
            return
        self.pin_connections = [pc for pc in self.pin_connections if not (pc.component_id == component_id and pc.pin_name == pin_name)]
        self.pin_connections.append(PinConnection(node_id, component_id, pin_name, source))
        self.pin_connections = self.unique_pin_connections(self.pin_connections)
        self.rebuild_pin_connection_index()
        self.set_status(f"Assigned {component_id}.{pin_name} -> node{node_id}")
        self.refresh_all()

    def delete_selected_pin_connection(self):
        idx = self.selected_pin_connection_index
        if idx is None or not (0 <= idx < len(self.pin_connections)):
            return
        del self.pin_connections[idx]
        self.selected_pin_connection_index = None
        self.rebuild_pin_connection_index()
        self.refresh_all()

    def clear_component_pin_connections(self, component_id: str):
        self.pin_connections = [pc for pc in self.pin_connections if pc.component_id != component_id]
        self.rebuild_pin_connection_index()
        self.refresh_all()

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
        ttk.Label(topbar, text="Pin Annotation Stage", font=("Arial", 12, "bold")).pack(side="left", padx=(0, 12))
        ttk.Button(topbar, text="Auto Generate Pins", command=lambda: self.auto_generate_pin_connections(True) or self.refresh_all()).pack(side="left", padx=4)
        ttk.Button(topbar, text="Save Pin Outputs", command=self.save_all_outputs).pack(side="left", padx=4)

        self.canvas = tk.Canvas(left, width=self.canvas_width, height=self.canvas_height, bg="gray20")
        self.canvas.pack(side="top", fill="both", expand=True)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.canvas.bind("<ButtonPress-1>", self.on_left_press)

        right = ttk.Frame(outer, width=560)
        right.pack(side="right", fill="y", padx=8, pady=8)

        self.status_label = ttk.Label(right, text="Ready", foreground="#333")
        self.status_label.pack(anchor="w", pady=(0, 8))

        ctrl = ttk.LabelFrame(right, text="Manual Pin Assignment")
        ctrl.pack(fill="x", pady=(0, 8))
        ttk.Checkbutton(ctrl, text="Show Component-Node Connections", variable=self.show_all_connections, command=self.refresh_all).pack(anchor="w", pady=2)
        ttk.Checkbutton(ctrl, text="Show Pin Connections", variable=self.show_all_pins, command=self.refresh_all).pack(anchor="w", pady=2)
        ttk.Label(ctrl, text="Pin name for selected component").pack(anchor="w", pady=(8, 2))
        self.pin_name_combo = ttk.Combobox(ctrl, textvariable=self.pin_name_var, state="readonly", values=[])
        self.pin_name_combo.pack(fill="x", pady=2)
        ttk.Button(ctrl, text="Assign Selected Node + Component + Pin", command=self.assign_current_selection).pack(fill="x", pady=2)
        ttk.Button(ctrl, text="Delete Selected Pin Connection", command=self.delete_selected_pin_connection).pack(fill="x", pady=2)
        ttk.Button(ctrl, text="Clear Selected Component Pins", command=self.clear_selected_component_pins).pack(fill="x", pady=2)
        ttk.Label(ctrl, text="Canvas:\n- click node to select node\n- click component to select component\n- then choose pin and assign", justify="left").pack(anchor="w", pady=(8, 4))

        ttk.Label(right, text="Summary", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 4))
        self.info_text = tk.Text(right, width=64, height=11, wrap="word")
        self.info_text.pack(fill="x", pady=(0, 8))

        lists = ttk.Frame(right)
        lists.pack(fill="both", expand=True)
        ttk.Label(lists, text="Nodes").grid(row=0, column=0, sticky="w", padx=4)
        ttk.Label(lists, text="Components").grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(lists, text="Pin Connections").grid(row=0, column=2, sticky="w", padx=4)
        self.nodes_list = tk.Listbox(lists, width=18, height=22, exportselection=False)
        self.comps_list = tk.Listbox(lists, width=26, height=22, exportselection=False)
        self.pin_conns_list = tk.Listbox(lists, width=34, height=22, exportselection=False)
        self.nodes_list.grid(row=1, column=0, padx=4, pady=4, sticky="nsew")
        self.comps_list.grid(row=1, column=1, padx=4, pady=4, sticky="nsew")
        self.pin_conns_list.grid(row=1, column=2, padx=4, pady=4, sticky="nsew")
        lists.columnconfigure(0, weight=1)
        lists.columnconfigure(1, weight=1)
        lists.columnconfigure(2, weight=1)
        self.nodes_list.bind("<<ListboxSelect>>", self.on_select_node_from_list)
        self.comps_list.bind("<<ListboxSelect>>", self.on_select_component_from_list)
        self.pin_conns_list.bind("<<ListboxSelect>>", self.on_select_pin_connection_from_list)

    # ----------------------------
    # Interaction
    # ----------------------------
    def update_pin_name_choices(self, comp: Component):
        values = self.get_pin_schema(comp.class_name)
        self.pin_name_combo["values"] = values
        self.pin_name_var.set(values[0] if values else "")

    def on_select_node_from_list(self, _event=None):
        sel = self.nodes_list.curselection()
        if not sel:
            return
        gid = int(self.nodes_list.get(sel[0]).replace("node", "").split()[0])
        self.selected_node_group_id = gid
        self.current_assign_node = gid
        self.refresh_all()

    def on_select_component_from_list(self, _event=None):
        sel = self.comps_list.curselection()
        if not sel:
            return
        comp = self.get_component_ordered()[sel[0]]
        self.selected_component_ids = {comp.component_id}
        self.current_assign_component_id = comp.component_id
        self.update_pin_name_choices(comp)
        self.refresh_all()

    def on_select_pin_connection_from_list(self, _event=None):
        sel = self.pin_conns_list.curselection()
        if not sel:
            return
        idx = sel[0]
        if 0 <= idx < len(self.pin_connections):
            pc = self.pin_connections[idx]
            self.selected_pin_connection_index = idx
            self.selected_node_group_id = pc.node_id
            self.current_assign_node = pc.node_id
            self.selected_component_ids = {pc.component_id}
            self.current_assign_component_id = pc.component_id
            self.pin_name_var.set(pc.pin_name)
            self.refresh_all()

    def on_left_press(self, event):
        x, y = self.canvas_to_image_coords(event.x, event.y)
        gid = self.find_group_at(int(x), int(y))
        if gid is not None:
            self.selected_node_group_id = gid
            self.current_assign_node = gid
            self.selected_pin_connection_index = None
            self.refresh_all()
            return
        cid = self.find_component_at(int(x), int(y))
        if cid is not None:
            self.selected_component_ids = {cid}
            self.current_assign_component_id = cid
            comp = self.get_component_by_id(cid)
            if comp is not None:
                self.update_pin_name_choices(comp)
            self.selected_pin_connection_index = None
            self.refresh_all()
            return
        self.selected_node_group_id = None
        self.selected_component_ids.clear()
        self.current_assign_node = None
        self.current_assign_component_id = None
        self.selected_pin_connection_index = None
        self.refresh_all()

    def assign_current_selection(self):
        if self.current_assign_node is None:
            messagebox.showwarning("Warning", "Select a node first.")
            return
        if self.current_assign_component_id is None:
            messagebox.showwarning("Warning", "Select a component first.")
            return
        pin_name = self.pin_name_var.get().strip()
        if not pin_name:
            messagebox.showwarning("Warning", "Choose a pin name.")
            return
        self.add_or_update_pin_connection(self.current_assign_node, self.current_assign_component_id, pin_name, source="manual")

    def clear_selected_component_pins(self):
        if not self.selected_component_ids:
            messagebox.showwarning("Warning", "Select a component first.")
            return
        self.clear_component_pin_connections(sorted(self.selected_component_ids)[0])

    # ----------------------------
    # Rendering
    # ----------------------------
    def get_node_color_bgr(self, node_id: int) -> Tuple[int, int, int]:
        palette = [
            (80, 80, 255), (80, 200, 80), (255, 170, 40), (220, 80, 220),
            (255, 255, 80), (80, 220, 220), (160, 120, 255), (120, 255, 160),
            (255, 120, 120), (120, 180, 255),
        ]
        return palette[(max(1, node_id) - 1) % len(palette)]

    def draw_nodes(self, vis):
        for gid, group in sorted(self.group_by_id.items()):
            color = self.get_node_color_bgr(gid)
            thickness = 2 if gid == self.selected_node_group_id else 1
            for aid in group.members_auto:
                node = self.auto_by_id.get(aid)
                if node is None:
                    continue
                x, y, w, h = node.bbox_xywh
                cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
            for mid in group.members_manual:
                node = self.manual_by_id.get(mid)
                if node is None:
                    continue
                cv2.circle(vis, (node.x, node.y), node.radius, color, thickness)
            ax, ay = self.get_group_anchor_point(group)
            cv2.circle(vis, (ax, ay), 4, color, -1)
            cv2.putText(vis, f"node{gid}", (ax + 5, max(12, ay - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    def draw_components(self, vis):
        for comp in self.components:
            x1, y1, x2, y2 = comp.bbox_xyxy
            selected = comp.component_id in self.selected_component_ids
            color = (0, 255, 255) if selected else (160, 160, 160)
            thickness = 2 if selected else 1
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(vis, f"{comp.component_id}:{comp.class_name}", (x1 + 2, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)

    def draw_component_node_connections(self, vis):
        if not self.show_all_connections.get():
            return
        for conn in self.connections:
            comp = self.get_component_by_id(conn.component_id)
            group = self.group_by_id.get(conn.node_id)
            if comp is None or group is None:
                continue
            ax, ay = self.get_group_anchor_point(group)
            x1, y1, x2, y2 = comp.bbox_xyxy
            cx = int(round((x1 + x2) / 2.0))
            cy = int(round((y1 + y2) / 2.0))
            color = self.get_node_color_bgr(conn.node_id)
            cv2.line(vis, (ax, ay), (cx, cy), color, 1, cv2.LINE_AA)

    def draw_pin_connections(self, vis):
        if not self.show_all_pins.get():
            return
        for pc in self.pin_connections:
            comp = self.get_component_by_id(pc.component_id)
            group = self.group_by_id.get(pc.node_id)
            if comp is None or group is None:
                continue
            pin_positions = self.get_component_pin_positions(comp)
            pin_pt = pin_positions.get(pc.pin_name)
            if pin_pt is None:
                continue
            node_pt = self.get_group_anchor_point(group)
            color = self.get_node_color_bgr(pc.node_id)
            cv2.line(vis, node_pt, pin_pt, color, 2, cv2.LINE_AA)
            cv2.circle(vis, pin_pt, 5, color, -1)
            cv2.circle(vis, pin_pt, 6, (0, 0, 0), 1)
            label = pc.pin_name.replace("terminal_", "T")
            cv2.putText(vis, label, (pin_pt[0] + 6, pin_pt[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)

        for cid in self.selected_component_ids:
            comp = self.get_component_by_id(cid)
            if comp is None:
                continue
            pin_positions = self.get_component_pin_positions(comp)
            assigned = {pc.pin_name for pc in self.pin_connections if pc.component_id == cid}
            for pin_name, pt in pin_positions.items():
                if pin_name in assigned:
                    continue
                cv2.circle(vis, pt, 4, (180, 180, 180), -1)
                cv2.circle(vis, pt, 5, (20, 20, 20), 1)
                cv2.putText(vis, pin_name.replace("terminal_", "T"), (pt[0] + 6, pt[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 200), 1, cv2.LINE_AA)

    def render(self):
        vis = (self.image_bgr.astype("float32") * 0.28).clip(0, 255).astype("uint8")
        self.draw_components(vis)
        self.draw_nodes(vis)
        self.draw_component_node_connections(vis)
        self.draw_pin_connections(vis)
        return Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

    def update_canvas(self):
        rendered = self.render()
        self.root.update_idletasks()
        canvas_w = self.canvas.winfo_width() or self.canvas_width
        canvas_h = self.canvas.winfo_height() or self.canvas_height
        new_w, new_h, scale = self.fit_image_size(rendered.width, rendered.height, canvas_w, canvas_h)
        rendered = rendered.resize((new_w, new_h), Image.Resampling.NEAREST)
        self.display_scale = scale
        self.display_img_w = new_w
        self.display_img_h = new_h
        self.display_offset_x = (canvas_w - new_w) // 2
        self.display_offset_y = (canvas_h - new_h) // 2
        self.tk_image = ImageTk.PhotoImage(rendered)
        self.canvas.delete("all")
        self.canvas.create_image(self.display_offset_x, self.display_offset_y, image=self.tk_image, anchor="nw")

    def on_canvas_configure(self, _event=None):
        if self.tk_image is not None:
            self.update_canvas()

    # ----------------------------
    # Refresh
    # ----------------------------
    def refresh_info(self):
        txt = self.info_text
        txt.config(state="normal")
        txt.delete("1.0", tk.END)
        txt.insert(tk.END, f"Image: {self.image_path.name}\n")
        txt.insert(tk.END, f"Components: {len(self.components)}\n")
        txt.insert(tk.END, f"Node groups: {len(self.group_by_id)}\n")
        txt.insert(tk.END, f"Component-node connections: {len(self.connections)}\n")
        txt.insert(tk.END, f"Pin connections: {len(self.pin_connections)}\n\n")
        txt.insert(tk.END, f"Auto pin classes: {sorted(AUTO_PIN_CLASSES)}\n")
        if self.selected_node_group_id is not None:
            txt.insert(tk.END, f"Selected node: node{self.selected_node_group_id}\n")
        if self.selected_component_ids:
            cid = sorted(self.selected_component_ids)[0]
            comp = self.get_component_by_id(cid)
            if comp is not None:
                current = [(pc.pin_name, pc.node_id, pc.source) for pc in self.pin_connections if pc.component_id == cid]
                txt.insert(tk.END, f"Selected component: {cid} ({comp.class_name})\n")
                txt.insert(tk.END, f"Available pins: {self.get_pin_schema(comp.class_name)}\n")
                txt.insert(tk.END, f"Current pin assignments: {current}\n")
        txt.config(state="disabled")

    def refresh_lists(self):
        self.nodes_list.delete(0, tk.END)
        for gid in sorted(self.group_by_id):
            attached = len([c for c in self.connections if c.node_id == gid])
            self.nodes_list.insert(tk.END, f"node{gid} ({attached} comps)")

        self.comps_list.delete(0, tk.END)
        for comp in self.get_component_ordered():
            num_nodes = len([c for c in self.connections if c.component_id == comp.component_id])
            num_pins = len([pc for pc in self.pin_connections if pc.component_id == comp.component_id])
            self.comps_list.insert(tk.END, f"{comp.component_id} | {comp.class_name} | nodes={num_nodes} pins={num_pins}")

        self.pin_conns_list.delete(0, tk.END)
        for pc in self.pin_connections:
            self.pin_conns_list.insert(tk.END, f"{pc.component_id}.{pc.pin_name} -> node{pc.node_id} [{pc.source}]")

    def refresh_all(self):
        self.refresh_info()
        self.refresh_lists()
        self.update_canvas()

    # ----------------------------
    # Saving
    # ----------------------------
    def pin_payload(self):
        return {
            "image_name": self.image_path.name,
            "image_path": str(self.image_path),
            "components_json": str(self.components_json),
            "node_json": str(self.node_json),
            "pin_schema": PIN_SCHEMA,
            "pin_connections": [asdict(pc) for pc in self.pin_connections],
        }

    def save_pin_csv(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["component_id", "class_name", "pin_name", "node_id", "source"])
            for pc in self.pin_connections:
                comp = self.get_component_by_id(pc.component_id)
                writer.writerow([pc.component_id, comp.class_name if comp else "", pc.pin_name, pc.node_id, pc.source])

    def save_all_outputs(self):
        pin_json = self.out_dir / f"{self.image_path.stem}_pin_annotation.json"
        pin_csv = self.out_dir / f"{self.image_path.stem}_pin_connections.csv"
        with pin_json.open("w", encoding="utf-8") as f:
            json.dump(self.pin_payload(), f, indent=2, ensure_ascii=False)
        self.save_pin_csv(pin_csv)
        self.set_status(f"Saved pin outputs to {self.out_dir}")
        messagebox.showinfo("Saved", f"Saved:\n- {pin_json}\n- {pin_csv}")


def main():
    ap = argparse.ArgumentParser(description="Pin annotation GUI from reviewed component/node files")
    ap.add_argument("--image", required=True, help="Path to the source image")
    ap.add_argument("--components_json", required=True, help="Path to reviewed_components.json")
    ap.add_argument("--node_json", required=True, help="Path to node_annotation.json")
    ap.add_argument("--out_dir", required=True, help="Output directory for pin annotation results")
    ap.add_argument("--canvas_width", type=int, default=1180)
    ap.add_argument("--canvas_height", type=int, default=800)
    args = ap.parse_args()

    root = tk.Tk()
    app = PinAnnotationGUI(root, args)
    app.refresh_all()
    root.mainloop()


if __name__ == "__main__":
    main()
