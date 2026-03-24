import json
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox


# ----------------------------
# Data
# ----------------------------
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
    radius: int
    kind: str = "manual_circle"


@dataclass
class ComponentItem:
    id: int
    category_id: int
    category_name: str
    instance_name: str
    bbox_xywh: Tuple[int, int, int, int]


@dataclass
class NodeGroup:
    id: int
    members_auto: List[int]
    members_manual: List[int]


@dataclass(frozen=True)
class Connection:
    node_id: int  # node group id
    component_id: int  # component instance id


# ----------------------------
# COCO per-image helpers
# ----------------------------
def load_per_image_coco(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_category_map(data: dict) -> Dict[int, str]:
    mp = {}
    for c in data.get("categories", []):
        cid = c.get("id")
        name = c.get("name")
        if cid is not None and name:
            mp[int(cid)] = str(name)
    return mp


def build_components(data: dict) -> List[ComponentItem]:
    cat_map = build_category_map(data)
    counter: Dict[str, int] = {}
    comps: List[ComponentItem] = []
    inst_id = 1
    for ann in data.get("annotations", []):
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x, y, w, h = bbox
        cat_id = int(ann.get("category_id", -1))
        cat_name = cat_map.get(cat_id, f"cat{cat_id}")
        counter.setdefault(cat_name, 0)
        counter[cat_name] += 1
        inst_name = f"{cat_name}_{counter[cat_name]}"
        comps.append(
            ComponentItem(
                id=inst_id,
                category_id=cat_id,
                category_name=cat_name,
                instance_name=inst_name,
                bbox_xywh=(int(round(x)), int(round(y)), int(round(w)), int(round(h))),
            )
        )
        inst_id += 1
    return comps


def load_component_bboxes(data: dict) -> List[Tuple[int, int, int, int]]:
    bboxes = []
    for ann in data.get("annotations", []):
        bbox = ann.get("bbox", None)
        if bbox and len(bbox) == 4:
            x, y, w, h = bbox
            bboxes.append((int(round(x)), int(round(y)), int(round(w)), int(round(h))))
    return bboxes


# ----------------------------
# Wire blobs (auto nodes)
# ----------------------------
def mask_components_white(
    img_bgr: np.ndarray, bboxes_xywh: List[Tuple[int, int, int, int]], pad: int = 2
) -> np.ndarray:
    out = img_bgr.copy()
    H, W = out.shape[:2]
    for x, y, w, h in bboxes_xywh:
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(W, x + w + pad)
        y1 = min(H, y + h + pad)
        out[y0:y1, x0:x1] = 255
    return out


def make_wire_binary(
    masked_bgr: np.ndarray, blur_ksize: int = 3, open_iter: int = 1, close_iter: int = 2
) -> np.ndarray:
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


def find_auto_nodes_from_bw(bw: np.ndarray, min_area: int = 80, connectivity: int = 8):
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


# ----------------------------
# Tk GUI
# ----------------------------
class App:
    def __init__(
        self,
        root,
        img_bgr,
        auto_nodes,
        label_map,
        components,
        out_path,
        image_path,
        comp_json_path,
        params,
    ):
        self.root = root
        self.root.title("Wire Nodes Annotator (Tkinter)")

        self.img_bgr = img_bgr
        self.H, self.W = img_bgr.shape[:2]

        self.auto_nodes = auto_nodes
        self.auto_by_id = {n.id: n for n in auto_nodes}
        self.label_map = label_map

        self.components = components
        self.comp_by_id = {c.id: c for c in components}

        # Manual nodes by id
        self.manual_by_id: Dict[int, ManualNode] = {}
        self.next_manual_id = (max([n.id for n in auto_nodes]) + 1) if auto_nodes else 1

        # Node groups in UI
        self.group_by_id: Dict[int, NodeGroup] = {
            n.id: NodeGroup(id=n.id, members_auto=[n.id], members_manual=[])
            for n in auto_nodes
        }

        # Selection: ("node", group_id) or ("component", component_id)
        self.selected: Optional[Tuple[str, int]] = None

        # Modes (mutually exclusive)
        self.merge_mode = tk.BooleanVar(value=False)
        self.connect_mode = tk.BooleanVar(value=False)

        # Merge state
        self.pending_merge_first: Optional[int] = None

        # Connect state: current node to connect from
        self.current_connect_node: Optional[int] = None

        # Connections (unique)
        self.connections: List[Connection] = []
        self.conn_index: Dict[Tuple[int, int], int] = (
            {}
        )  # (node_id, comp_id) -> idx in list

        # Undo stack (basic)
        self.undo_stack: List[dict] = []

        self.out_path = out_path
        self.image_path = image_path
        self.comp_json_path = comp_json_path
        self.params = params

        # --- layout ---
        self.main = ttk.Frame(root)
        self.main.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(self.main, width=self.W, height=self.H, bg="white")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        right = ttk.Frame(self.main)
        right.grid(row=0, column=1, sticky="ns")
        self.main.columnconfigure(0, weight=1)
        self.main.rowconfigure(0, weight=1)

        # Titles
        ttk.Label(right, text="Nodes").grid(row=0, column=0, padx=6, pady=(6, 0))
        ttk.Label(right, text="Components").grid(row=0, column=1, padx=6, pady=(6, 0))
        ttk.Label(right, text="Connections").grid(row=0, column=2, padx=6, pady=(6, 0))

        # Lists
        self.nodes_list = tk.Listbox(right, width=18, height=26, exportselection=False)
        self.comps_list = tk.Listbox(right, width=26, height=26, exportselection=False)
        self.conns_list = tk.Listbox(right, width=32, height=26, exportselection=False)

        self.nodes_list.grid(row=1, column=0, padx=6, pady=6)
        self.comps_list.grid(row=1, column=1, padx=6, pady=6)
        self.conns_list.grid(row=1, column=2, padx=6, pady=6)

        # Buttons row
        btns = ttk.Frame(right)
        btns.grid(row=2, column=0, columnspan=3, pady=(0, 6))
        ttk.Button(btns, text="Save", command=self.on_save).grid(
            row=0, column=0, padx=6
        )
        ttk.Button(btns, text="Undo", command=self.on_undo).grid(
            row=0, column=1, padx=6
        )
        ttk.Button(btns, text="Reset", command=self.on_reset).grid(
            row=0, column=2, padx=6
        )

        # Delete / Modes / Connection delete
        btns2 = ttk.Frame(right)
        btns2.grid(row=3, column=0, columnspan=3, pady=(0, 6))
        ttk.Button(btns2, text="Delete Node", command=self.delete_selected_node).grid(
            row=0, column=0, padx=6
        )
        ttk.Checkbutton(
            btns2,
            text="Merge Mode (m)",
            variable=self.merge_mode,
            command=self.on_toggle_merge,
        ).grid(row=0, column=1, padx=6)
        ttk.Checkbutton(
            btns2,
            text="Connect Mode (c)",
            variable=self.connect_mode,
            command=self.on_toggle_connect,
        ).grid(row=0, column=2, padx=6)

        btns3 = ttk.Frame(right)
        btns3.grid(row=4, column=0, columnspan=3, pady=(0, 6))
        ttk.Button(
            btns3, text="Delete Connection", command=self.delete_selected_connection
        ).grid(row=0, column=0, padx=6)

        # Status line
        self.status = ttk.Label(right, text="Ready", foreground="#444")
        self.status.grid(row=5, column=0, columnspan=3, padx=6, pady=(0, 6), sticky="w")

        # Bind events
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.nodes_list.bind("<<ListboxSelect>>", self.on_select_node)
        self.comps_list.bind("<<ListboxSelect>>", self.on_select_comp)
        self.conns_list.bind("<<ListboxSelect>>", self.on_select_conn)

        self.root.bind("<KeyPress-u>", lambda e: self.on_undo())
        self.root.bind("<KeyPress-s>", lambda e: self.on_save())
        self.root.bind("<Escape>", lambda e: self.on_reset())
        self.root.bind("<Delete>", lambda e: self.on_delete_key())
        self.root.bind("<KeyPress-m>", lambda e: self.toggle_merge_mode())
        self.root.bind("<KeyPress-c>", lambda e: self.toggle_connect_mode())

        self.populate_lists()
        self.redraw()

    # ----------------------------
    # Modes
    # ----------------------------
    def set_status(self, msg: str):
        self.status.configure(text=msg)

    def on_toggle_merge(self):
        if self.merge_mode.get():
            # mutually exclusive
            self.connect_mode.set(False)
            self.current_connect_node = None
            self.set_status("Merge Mode ON: click nodeA then nodeB")
        else:
            self.pending_merge_first = None
            self.set_status("Merge Mode OFF")

    def on_toggle_connect(self):
        if self.connect_mode.get():
            # mutually exclusive
            self.merge_mode.set(False)
            self.pending_merge_first = None
            self.set_status(
                "Connect Mode ON: select node then select component to connect"
            )
        else:
            self.current_connect_node = None
            self.set_status("Connect Mode OFF")

    def toggle_merge_mode(self):
        self.merge_mode.set(not self.merge_mode.get())
        self.on_toggle_merge()

    def toggle_connect_mode(self):
        self.connect_mode.set(not self.connect_mode.get())
        self.on_toggle_connect()

    # ----------------------------
    # Lists
    # ----------------------------
    def populate_lists(self):
        self.nodes_list.delete(0, tk.END)
        for gid in sorted(self.group_by_id.keys()):
            self.nodes_list.insert(tk.END, f"node{gid}")

        self.comps_list.delete(0, tk.END)
        for c in self.components:
            self.comps_list.insert(tk.END, c.instance_name)

        self.refresh_connections_list()

    def refresh_connections_list(self):
        self.conns_list.delete(0, tk.END)
        for conn in self.connections:
            cname = (
                self.comp_by_id[conn.component_id].instance_name
                if conn.component_id in self.comp_by_id
                else f"comp{conn.component_id}"
            )
            self.conns_list.insert(tk.END, f"node{conn.node_id}  ->  {cname}")

    def get_group(self, gid: int) -> Optional[NodeGroup]:
        return self.group_by_id.get(gid)

    # ----------------------------
    # Drawing helpers
    # ----------------------------
    def darken(self, img, alpha=0.25):
        return (img.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)

    def group_union_mask(self, group: NodeGroup) -> np.ndarray:
        mask = np.zeros((self.H, self.W), dtype=np.uint8)
        for aid in group.members_auto:
            mask[self.label_map == aid] = 255
        for mid in group.members_manual:
            m = self.manual_by_id.get(mid)
            if m is None:
                continue
            cv2.circle(mask, (m.x, m.y), m.radius, 255, -1)
        return mask

    def draw_group_outline_and_id(self, vis, group: NodeGroup, color=(0, 255, 0)):
        mask = self.group_union_mask(group)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
        cv2.drawContours(vis, contours, -1, color, 1)
        x, y, w, h = cv2.boundingRect(np.vstack(contours))
        cv2.putText(
            vis,
            str(group.id),
            (x + 2, max(10, y + 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28,
            color,
            1,
            cv2.LINE_AA,
        )

    def draw_component(self, vis, comp: ComponentItem, color=(255, 0, 255)):
        x, y, w, h = comp.bbox_xywh
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 1)
        cv2.putText(
            vis,
            comp.instance_name,
            (x + 2, max(10, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28,
            color,
            1,
            cv2.LINE_AA,
        )

    def draw_component_label_only(self, vis, comp: ComponentItem, color=(80, 80, 80)):
        # 只画文字，不画bbox框
        x, y, w, h = comp.bbox_xywh
        # 放在 bbox 左上角上方一点，避免压住元件本体（你可按喜好改位置）
        tx = x + 2
        ty = max(12, y - 4)
        cv2.putText(
            vis,
            comp.instance_name,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28,
            color,
            1,
            cv2.LINE_AA,
        )

    # ----------------------------
    # Render
    # ----------------------------
    def render(self) -> np.ndarray:
        if self.selected is None:
            vis = self.img_bgr.copy()
            for c in self.components:
                self.draw_component_label_only(vis, c, color=(80, 80, 80))
            for gid in sorted(self.group_by_id.keys()):
                self.draw_group_outline_and_id(
                    vis, self.group_by_id[gid], color=(0, 255, 0)
                )
            return vis

        kind, sid = self.selected
        vis = self.darken(self.img_bgr, alpha=0.25)
        for c in self.components:
            self.draw_component_label_only(vis, c, color=(150, 150, 150))

        if kind == "node":
            g = self.get_group(sid)
            if g is not None:
                self.draw_group_outline_and_id(vis, g, color=(0, 255, 0))

            # In connect mode, also highlight components already connected to this node
            if self.connect_mode.get():
                for conn in self.connections:
                    if conn.node_id == sid and conn.component_id in self.comp_by_id:
                        self.draw_component(
                            vis, self.comp_by_id[conn.component_id], color=(255, 0, 255)
                        )

        elif kind == "component":
            comp = self.comp_by_id.get(sid)
            if comp is not None:
                self.draw_component(vis, comp, color=(255, 0, 255))

        return vis

    def redraw(self):
        vis_bgr = self.render()
        vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(vis_rgb)
        self.tk_img = ImageTk.PhotoImage(pil)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

    # ----------------------------
    # Node operations
    # ----------------------------
    def add_manual_node(self, x: int, y: int):
        mid = self.next_manual_id
        self.next_manual_id += 1

        self.manual_by_id[mid] = ManualNode(id=mid, x=x, y=y, radius=6)
        self.group_by_id[mid] = NodeGroup(id=mid, members_auto=[], members_manual=[mid])

        self.undo_stack.append({"op": "add_manual", "manual_id": mid})
        self.populate_lists()
        self.redraw()

    def delete_group(self, gid: int):
        g = self.group_by_id.get(gid)
        if g is None:
            return

        # snapshot for undo
        snap = {
            "op": "delete_group",
            "group": asdict(g),
            "manual_nodes": [
                asdict(self.manual_by_id[mid])
                for mid in g.members_manual
                if mid in self.manual_by_id
            ],
            "connections_removed": [
                asdict(c) for c in self.connections if c.node_id == gid
            ],
        }
        self.undo_stack.append(snap)

        # remove manual nodes
        for mid in g.members_manual:
            self.manual_by_id.pop(mid, None)

        # remove group
        del self.group_by_id[gid]

        # remove connections that reference this node
        self.connections = [c for c in self.connections if c.node_id != gid]
        self.rebuild_conn_index()

        # selection cleanup
        if self.selected and self.selected[0] == "node" and self.selected[1] == gid:
            self.selected = None
        if self.pending_merge_first == gid:
            self.pending_merge_first = None
        if self.current_connect_node == gid:
            self.current_connect_node = None

        self.populate_lists()
        self.redraw()

    def merge_groups(self, gid_a: int, gid_b: int):
        if gid_a == gid_b:
            return
        ga = self.group_by_id.get(gid_a)
        gb = self.group_by_id.get(gid_b)
        if ga is None or gb is None:
            return

        before_a = asdict(ga)
        before_b = asdict(gb)

        ga.members_auto = sorted(list(set(ga.members_auto + gb.members_auto)))
        ga.members_manual = sorted(list(set(ga.members_manual + gb.members_manual)))
        del self.group_by_id[gid_b]

        # migrate connections of nodeB to nodeA
        moved = []
        for c in self.connections:
            if c.node_id == gid_b:
                moved.append(asdict(c))
        new_conns = []
        for c in self.connections:
            if c.node_id == gid_b:
                new_conns.append(Connection(node_id=gid_a, component_id=c.component_id))
            else:
                new_conns.append(c)
        # dedupe
        uniq = []
        seen = set()
        for c in new_conns:
            key = (c.node_id, c.component_id)
            if key not in seen:
                uniq.append(c)
                seen.add(key)
        self.connections = uniq
        self.rebuild_conn_index()

        self.undo_stack.append(
            {
                "op": "merge_groups",
                "a_before": before_a,
                "b_before": before_b,
                "moved": moved,
            }
        )

        self.selected = ("node", gid_a)
        self.pending_merge_first = None
        if self.current_connect_node == gid_b:
            self.current_connect_node = gid_a

        self.populate_lists()
        self.redraw()

    # ----------------------------
    # Connections
    # ----------------------------
    def rebuild_conn_index(self):
        self.conn_index = {}
        for i, c in enumerate(self.connections):
            self.conn_index[(c.node_id, c.component_id)] = i

    def add_connection(self, node_id: int, comp_id: int):
        if node_id not in self.group_by_id:
            return
        if comp_id not in self.comp_by_id:
            return
        key = (node_id, comp_id)
        if key in self.conn_index:
            self.set_status(
                f"Connection already exists: node{node_id} -> {self.comp_by_id[comp_id].instance_name}"
            )
            return

        conn = Connection(node_id=node_id, component_id=comp_id)
        self.connections.append(conn)
        self.rebuild_conn_index()
        self.undo_stack.append({"op": "add_connection", "conn": asdict(conn)})

        self.refresh_connections_list()
        self.set_status(
            f"Added: node{node_id} -> {self.comp_by_id[comp_id].instance_name}"
        )
        self.redraw()

    def delete_selected_connection(self):
        sel = self.conns_list.curselection()
        if not sel:
            return
        idx = sel[0]
        if not (0 <= idx < len(self.connections)):
            return

        conn = self.connections[idx]
        self.undo_stack.append({"op": "delete_connection", "conn": asdict(conn)})

        del self.connections[idx]
        self.rebuild_conn_index()
        self.refresh_connections_list()
        self.set_status("Deleted connection")
        self.redraw()

    # ----------------------------
    # Event handlers
    # ----------------------------
    def on_canvas_click(self, event):
        # Click canvas: if focused -> reset focus; else add manual node
        if self.selected is not None:
            self.on_reset()
            return
        # Don't allow adding manual node while connect mode is ON (optional safety)
        # If you DO want to allow it, delete the next 2 lines.
        if self.connect_mode.get():
            self.set_status(
                "Connect Mode ON: select node/component on the right to connect"
            )
            return

        self.add_manual_node(int(event.x), int(event.y))

    def on_select_node(self, event):
        sel = self.nodes_list.curselection()
        if not sel:
            return
        gid = int(self.nodes_list.get(sel[0]).replace("node", ""))

        if self.merge_mode.get():
            if self.pending_merge_first is None:
                self.pending_merge_first = gid
                self.selected = ("node", gid)
                self.set_status(f"Merge: first = node{gid}, now pick second node")
                self.redraw()
                return
            else:
                first = self.pending_merge_first
                if gid != first:
                    self.merge_groups(first, gid)
                    self.set_status(f"Merged node{gid} into node{first}")
                else:
                    self.pending_merge_first = None
                    self.set_status("Merge cancelled")
                return

        # Connect mode: set current connect node
        if self.connect_mode.get():
            self.current_connect_node = gid
            self.selected = ("node", gid)
            self.set_status(
                f"Connect: current node = node{gid}. Now click a component to attach."
            )
            self.redraw()
            return

        # Normal selection
        self.selected = ("node", gid)
        self.pending_merge_first = None
        self.current_connect_node = None
        self.redraw()

    def on_select_comp(self, event):
        sel = self.comps_list.curselection()
        if not sel:
            return
        comp = self.components[sel[0]]

        if self.connect_mode.get():
            if self.current_connect_node is None:
                self.set_status("Connect Mode: please select a node first.")
                return
            self.add_connection(self.current_connect_node, comp.id)
            # keep focus on node for faster multiple connections
            self.selected = ("node", self.current_connect_node)
            self.redraw()
            return

        # Normal: focus component
        self.selected = ("component", comp.id)
        self.pending_merge_first = None
        self.current_connect_node = None
        self.redraw()

    def on_select_conn(self, event):
        # optional: when selecting a connection, highlight it
        sel = self.conns_list.curselection()
        if not sel:
            return
        idx = sel[0]
        if 0 <= idx < len(self.connections):
            conn = self.connections[idx]
            # focus the node to show connected comps
            self.selected = ("node", conn.node_id)
            self.redraw()

    def on_delete_key(self):
        # If a connection row selected -> delete connection; else if node selected -> delete node
        if self.conns_list.curselection():
            self.delete_selected_connection()
            return
        self.delete_selected_node()

    def on_reset(self):
        self.selected = None
        self.pending_merge_first = None
        if not self.connect_mode.get():
            self.current_connect_node = None
        self.nodes_list.selection_clear(0, tk.END)
        self.comps_list.selection_clear(0, tk.END)
        self.conns_list.selection_clear(0, tk.END)
        self.set_status("Ready")
        self.redraw()

    def delete_selected_node(self):
        if self.selected is None or self.selected[0] != "node":
            return
        gid = self.selected[1]
        self.delete_group(gid)

    def on_undo(self):
        if not self.undo_stack:
            return
        act = self.undo_stack.pop()
        op = act.get("op")

        if op == "add_manual":
            mid = act["manual_id"]
            self.group_by_id.pop(mid, None)
            self.manual_by_id.pop(mid, None)

        elif op == "delete_group":
            gdict = act["group"]
            g = NodeGroup(**gdict)
            self.group_by_id[g.id] = g
            for md in act.get("manual_nodes", []):
                m = ManualNode(**md)
                self.manual_by_id[m.id] = m
            # restore removed connections
            for cd in act.get("connections_removed", []):
                c = Connection(**cd)
                self.connections.append(c)
            # dedupe
            uniq = []
            seen = set()
            for c in self.connections:
                key = (c.node_id, c.component_id)
                if key not in seen:
                    uniq.append(c)
                    seen.add(key)
            self.connections = uniq
            self.rebuild_conn_index()

        elif op == "merge_groups":
            a = act["a_before"]
            b = act["b_before"]
            self.group_by_id[int(a["id"])] = NodeGroup(**a)
            self.group_by_id[int(b["id"])] = NodeGroup(**b)
            # remove any connections with node=a that came from b, then restore b's moved
            # simplest: recompute by removing all conns involving b->a migration isn't tracked perfectly;
            # we'll do robust route: clear and rebuild from current list + moved reversed.
            moved = act.get("moved", [])
            # remove all conns where node_id == a and component_id matches moved components (those were from b)
            moved_comp_ids = set([m["component_id"] for m in moved])
            self.connections = [
                c
                for c in self.connections
                if not (c.node_id == int(a["id"]) and c.component_id in moved_comp_ids)
            ]
            # restore b connections
            for m in moved:
                self.connections.append(
                    Connection(
                        node_id=int(b["node_id"]), component_id=int(m["component_id"])
                    )
                )
            # dedupe
            uniq = []
            seen = set()
            for c in self.connections:
                key = (c.node_id, c.component_id)
                if key not in seen:
                    uniq.append(c)
                    seen.add(key)
            self.connections = uniq
            self.rebuild_conn_index()

        elif op == "add_connection":
            cd = act["conn"]
            key = (cd["node_id"], cd["component_id"])
            if key in self.conn_index:
                idx = self.conn_index[key]
                del self.connections[idx]
                self.rebuild_conn_index()

        elif op == "delete_connection":
            cd = act["conn"]
            self.connections.append(Connection(**cd))
            # dedupe
            uniq = []
            seen = set()
            for c in self.connections:
                key = (c.node_id, c.component_id)
                if key not in seen:
                    uniq.append(c)
                    seen.add(key)
            self.connections = uniq
            self.rebuild_conn_index()

        self.populate_lists()
        self.selected = None
        self.pending_merge_first = None
        if not self.connect_mode.get():
            self.current_connect_node = None
        self.set_status("Undo")
        self.redraw()

    def on_save(self):
        payload = {
            "image": self.image_path.name,
            "source_component_json": self.comp_json_path.name,
            "params": self.params,
            "components": [asdict(c) for c in self.components],
            "node_groups": [
                {
                    "id": gid,
                    "members_auto": g.members_auto,
                    "members_manual": g.members_manual,
                }
                for gid, g in sorted(self.group_by_id.items(), key=lambda x: x[0])
            ],
            "manual_nodes": [asdict(m) for m in self.manual_by_id.values()],
            "connections": [asdict(c) for c in self.connections],
        }

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        messagebox.showinfo("Saved", f"Saved to:\n{self.out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mask_pad", type=int, default=2)
    ap.add_argument("--blur", type=int, default=3)
    ap.add_argument("--open_iter", type=int, default=1)
    ap.add_argument("--close_iter", type=int, default=2)
    ap.add_argument("--min_area", type=int, default=80)
    ap.add_argument("--connectivity", type=int, default=8)
    args = ap.parse_args()

    image_path = Path(args.image)
    comp_json_path = Path(args.json)
    out_path = (
        Path(args.out)
        if args.out
        else comp_json_path.with_name(f"{image_path.stem}_nodes.json")
    )

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    data = load_per_image_coco(comp_json_path)
    comps = build_components(data)
    comp_bboxes = load_component_bboxes(data)

    masked = mask_components_white(img, comp_bboxes, pad=args.mask_pad)
    bw = make_wire_binary(
        masked,
        blur_ksize=args.blur,
        open_iter=args.open_iter,
        close_iter=args.close_iter,
    )
    auto_nodes, label_map = find_auto_nodes_from_bw(
        bw, min_area=args.min_area, connectivity=args.connectivity
    )

    params = {
        "mask_pad": args.mask_pad,
        "blur": args.blur,
        "open_iter": args.open_iter,
        "close_iter": args.close_iter,
        "min_area": args.min_area,
        "connectivity": args.connectivity,
    }

    root = tk.Tk()
    App(
        root,
        img,
        auto_nodes,
        label_map,
        comps,
        out_path,
        image_path,
        comp_json_path,
        params,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
