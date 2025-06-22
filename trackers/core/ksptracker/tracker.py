from dataclasses import dataclass
from typing import Any, Dict, List

import networkx as nx
import numpy as np
import supervision as sv

from trackers.core.base import BaseTracker

import matplotlib.pyplot as plt
import networkx as nx

import cv2
import itertools
from copy import deepcopy
from pyvis.network import Network
from trackers.core.ksptracker.InteractivePMapViewer import InteractivePMapViewer


def visualize_tracking_graph_debug(G: nx.DiGraph, max_edges=500):
    import matplotlib.pyplot as plt
    import networkx as nx

    plt.figure(figsize=(18, 8))

    # Collect all TrackNode nodes
    track_nodes = [n for n in G.nodes if isinstance(n, TrackNode)]
    frames = sorted(set(n.frame_id for n in track_nodes))

    frame_to_x = {f: i for i, f in enumerate(frames)}

    # Group nodes by frame and sort by det_idx (or confidence)
    nodes_by_frame = {}
    for node in track_nodes:
        nodes_by_frame.setdefault(node.frame_id, []).append(node)
    for frame in nodes_by_frame:
        nodes_by_frame[frame].sort(key=lambda n: n.det_idx)  # or key=lambda n: -n.confidence for sorting by confidence

    pos = {}
    for node in G.nodes:
        if node == "source":
            pos[node] = (-1, 0)
        elif node == "sink":
            pos[node] = (len(frames), 0)
        elif isinstance(node, TrackNode):
            x = frame_to_x[node.frame_id]
            # vertical spacing: spread nodes evenly in y-axis
            idx = nodes_by_frame[node.frame_id].index(node)
            total = len(nodes_by_frame[node.frame_id])
            # spread vertically between 0 and total, centered around 0
            y = idx - total / 2
            pos[node] = (x, y)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color="lightblue", alpha=0.9)

    # Labels
    labels = {}
    for node in G.nodes:
        if node == "source":
            labels[node] = "SRC"
        elif node == "sink":
            labels[node] = "SNK"
        elif isinstance(node, TrackNode):
            labels[node] = f"F{node.frame_id},D{node.det_idx}"
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    # Edges (limit number)
    edges_to_draw = list(G.edges(data=True))[:max_edges]
    edge_list = [(u, v) for u, v, _ in edges_to_draw]
    nx.draw_networkx_edges(G, pos, edgelist=edge_list, arrowstyle='->', arrowsize=15, edge_color='gray')

    # Edge weights
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges_to_draw}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, label_pos=0.5)

    plt.title("Tracking Graph - Directed Timeline Layout with Vertical Spacing")
    plt.xlabel("Frame Index")
    plt.ylabel("Detection Vertical Position (Jittered)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_all_shortest_bellman_ford_paths(G: nx.DiGraph):
    try:
        all_paths = list(nx.all_shortest_paths(G, source="source", target="sink", weight="weight", method="bellman-ford"))
        if not all_paths:
            print("No path found from source to sink.")
            return
        shortest_path_length = sum(G[u][v]['weight'] for u, v in zip(all_paths[0][:-1], all_paths[0][1:]))
        print(f"Number of shortest paths from source to sink: {len(all_paths)}")
        print(f"Each shortest path has cost: {shortest_path_length:.2f}")

    except nx.NetworkXUnbounded:
        print("Negative weight cycle detected.")
        return

    plt.figure(figsize=(18, 8))

    # Layout
    track_nodes = [n for n in G.nodes if isinstance(n, TrackNode)]
    frames = sorted(set(n.frame_id for n in track_nodes))
    frame_to_x = {f: i for i, f in enumerate(frames)}

    nodes_by_frame = {}
    for node in track_nodes:
        nodes_by_frame.setdefault(node.frame_id, []).append(node)
    for frame in nodes_by_frame:
        nodes_by_frame[frame].sort(key=lambda n: n.det_idx)

    pos = {}
    for node in G.nodes:
        if node == "source":
            pos[node] = (-1, 0)
        elif node == "sink":
            pos[node] = (len(frames), 0)
        elif isinstance(node, TrackNode):
            x = frame_to_x[node.frame_id]
            idx = nodes_by_frame[node.frame_id].index(node)
            total = len(nodes_by_frame[node.frame_id])
            y = idx - total / 2
            pos[node] = (x, y)

    # Draw all nodes and labels
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color="lightgray", alpha=0.7)
    labels = {
        node: (
            "SRC" if node == "source"
            else "SNK" if node == "sink"
            else f"F{node.frame_id},D{node.det_idx}"
        )
        for node in G.nodes
    }
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    # Draw all edges (light gray)
    nx.draw_networkx_edges(G, pos, edge_color="lightgray", alpha=0.3)

    # Draw edge weights for all edges
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

    # Collect all edges in all shortest paths
    edges_in_paths = set()
    nodes_in_paths = set()
    for path in all_paths:
        nodes_in_paths.update(path)
        edges_in_paths.update(zip(path[:-1], path[1:]))

    # Draw all nodes in shortest paths highlighted (orange)
    nx.draw_networkx_nodes(G, pos, nodelist=nodes_in_paths, node_color="orange")

    # Draw all edges in shortest paths highlighted (red, thicker)
    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges_in_paths,
        edge_color="red",
        width=2.5,
        arrowstyle="->",
        arrowsize=15,
        label="Shortest Paths"
    )

    plt.title("Bellman-Ford All Shortest Paths Visualization")
    plt.xlabel("Frame Index")
    plt.ylabel("Detection Index Offset")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_path(G, path, pos, path_num, color="red"):
    plt.figure(figsize=(12, 6))

    # Draw all nodes and edges lightly
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color="lightgray", alpha=0.7)
    nx.draw_networkx_edges(G, pos, edge_color="lightgray", alpha=0.3)
    labels = {
        node: (
            "SRC" if node == "source"
            else "SNK" if node == "sink"
            else f"F{node.frame_id},D{node.det_idx}"
        )
        for node in G.nodes
    }
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    # Highlight the path nodes and edges
    nx.draw_networkx_nodes(G, pos, nodelist=path, node_color=color)
    path_edges = list(zip(path[:-1], path[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color=color, width=3, arrowstyle="->", arrowsize=15)

    plt.title(f"Node-Disjoint Shortest Path #{path_num}")
    plt.xlabel("Frame Index")
    plt.ylabel("Detection Index Offset")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def path_to_detections(path) -> sv.Detections:
    track_nodes = [node for node in path if hasattr(node, "frame_id") and hasattr(node, "det_idx")]

    xyxys = []
    confidences = []
    class_ids = []

    for node in track_nodes:
        det_idx = node.det_idx
        dets = node.dets  # sv.Detections

        bbox = dets.xyxy[det_idx]
        if bbox is None or len(bbox) != 4:
            raise ValueError(f"Invalid bbox at node {node}: {bbox}")

        xyxys.append(bbox)
        confidences.append(dets.confidence[det_idx])

        if hasattr(dets, "class_id") and len(dets.class_id) > det_idx:
            class_ids.append(dets.class_id[det_idx])
        else:
            class_ids.append(0)

    xyxys = np.array(xyxys)
    confidences = np.array(confidences)
    class_ids = np.array(class_ids)

    if xyxys.ndim != 2 or xyxys.shape[1] != 4:
        raise ValueError(f"xyxy must be 2D array with shape (_,4), got shape {xyxys.shape}")

    return sv.Detections(xyxy=xyxys, confidence=confidences, class_id=class_ids)


def find_and_visualize_disjoint_paths(G_orig, source="source", sink="sink", weight="weight") -> List[sv.Detections]:
    G = G_orig.copy()

    # Compute layout positions once for consistency
    track_nodes = [n for n in G.nodes if hasattr(n, "frame_id") and hasattr(n, "det_idx")]
    frames = sorted(set(n.frame_id for n in track_nodes))
    frame_to_x = {f: i for i, f in enumerate(frames)}
    nodes_by_frame = {}
    for node in track_nodes:
        nodes_by_frame.setdefault(node.frame_id, []).append(node)
    for frame in nodes_by_frame:
        nodes_by_frame[frame].sort(key=lambda n: n.det_idx)
    pos = {}
    for node in G.nodes:
        if node == source:
            pos[node] = (-1, 0)
        elif node == sink:
            pos[node] = (len(frames), 0)
        elif hasattr(node, "frame_id") and hasattr(node, "det_idx"):
            x = frame_to_x[node.frame_id]
            idx = nodes_by_frame[node.frame_id].index(node)
            total = len(nodes_by_frame[node.frame_id])
            y = idx - total / 2
            pos[node] = (x, y)
        else:
            pos[node] = (0, 0)  # fallback

    all_detections = []
    colors = itertools.cycle(["red", "blue", "green", "orange", "purple", "brown", "cyan", "magenta"])

    while True:
        try:
            length, paths = nx.single_source_bellman_ford(G, source=source, weight=weight)
            if sink not in paths:
                print("No more paths found.")
                break

            shortest_path = paths[sink]
            cost = length[sink]
            color = next(colors)
            print(f"Found path with cost {cost:.2f}: {shortest_path}")

            visualize_path(G_orig, shortest_path, pos, len(all_detections) + 1, color=color)

            dets = path_to_detections(shortest_path)
            all_detections.append(dets)

            # Remove intermediate nodes to enforce node-disjointness
            intermediate_nodes = shortest_path[1:-1]
            G.remove_nodes_from(intermediate_nodes)

        except nx.NetworkXNoPath:
            print("No more paths found.")
            break
        except nx.NetworkXUnbounded:
            print("Negative weight cycle detected.")
            break

    print(f"Total node-disjoint shortest paths found: {len(all_detections)}")
    return all_detections

def visualize_tracking_graph_with_path(G: nx.DiGraph, path: list, title: str = "Tracking Graph with Path") -> None:
    """
    Visualize the graph with:
    - Nodes arranged by frame and position
    - Edge costs displayed for all edges
    - Highlight a specific path with thicker, colored edges and nodes

    Args:
        G (nx.DiGraph): The tracking graph.
        path (list): A list of nodes forming the path.
        title (str): Plot title.
    """
    pos = {}
    spacing_x = 5
    spacing_y = 5

    # Determine max frame for sink placement
    max_frame = max((n.frame_id for n in G.nodes if isinstance(n, TrackNode)), default=0)

    # Position nodes: x by frame_id, y by position y_bin (grid cell)
    for node in G.nodes():
        if node == "source":
            pos[node] = (-spacing_x, 0)
        elif node == "sink":
            pos[node] = (spacing_x * (max_frame + 1), 0)
        elif isinstance(node, TrackNode):
            pos[node] = (spacing_x * node.frame_id, spacing_y * node.position[1])

    path_edges = set(zip(path, path[1:]))

    node_colors = []
    for node in G.nodes():
        if node == "source":
            node_colors.append("green")
        elif node == "sink":
            node_colors.append("red")
        elif node in path:
            node_colors.append("orange")
        else:
            node_colors.append("white")

    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        if (u, v) in path_edges:
            edge_colors.append("orange")
            edge_widths.append(2.5)
        else:
            edge_colors.append("gray")
            edge_widths.append(.5)

    plt.figure(figsize=(16, 8))
    nx.draw(
        G,
        pos,
        with_labels=False,
        node_color=node_colors,
        edge_color=edge_colors,
        width=edge_widths,
        node_size=250,
        arrows=True,
    )

    # Show edge weights for all edges
    edge_labels = {
        (u, v): f"{G[u][v]['weight']:.2f}"
        for u, v in G.edges()
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', font_size=7)

    # Node labels (frame:det_idx)
    node_labels = {
        node: f"{node.frame_id}:{node.det_idx}" if isinstance(node, TrackNode) else node
        for node in G.nodes()
    }
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=6)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def visualize_tracking_graph_with_path_pyvis(G: nx.DiGraph, path: list, output_file="graph.html"):
    net = Network(height="800px", width="100%", directed=True)
    net.force_atlas_2based()

    spacing_x = 300
    spacing_y = 50
    path_edges = set(zip(path, path[1:]))
    max_frame = max((n.frame_id for n in G.nodes if isinstance(n, TrackNode)), default=0)

    for node in G.nodes():
        if node == "source":
            x, y = -spacing_x, 0
        elif node == "sink":
            x, y = spacing_x * (max_frame + 1), 0
        elif isinstance(node, TrackNode):
            x = spacing_x * node.frame_id
            y = spacing_y * node.position[1]
        else:
            x, y = 0, 0

        node_id = str(node)
        label = f"{node}" if not isinstance(node, TrackNode) else f"{node.frame_id}:{node.det_idx}"
        color = "green" if node == "source" else "red" if node == "sink" else "orange" if node in path else "lightgray"

        net.add_node(node_id, label=label, color=color, x=x, y=y, physics=False)

    for u, v, data in G.edges(data=True):
        u_id, v_id = str(u), str(v)
        color = "orange" if (u, v) in path_edges else "gray"
        width = 3 if (u, v) in path_edges else 1
        label = f"{data['weight']:.2f}"
        net.add_edge(u_id, v_id, color=color, width=width, label=label)

    net.set_options("""
    var options = {
      "nodes": {
        "font": {
          "size": 14
        }
      },
      "edges": {
        "font": {
          "size": 10,
          "align": "middle"
        },
        "smooth": false
      },
      "physics": {
        "enabled": false
      }
    }
    """)
    net.show(output_file, notebook=False)

def visualize_p_map_on_image(frame, p_map, grid_size, alpha=0.4, show_text=True, cmap='Greens'):
    """
    Overlay probability map as a grid on the image using Matplotlib.

    Args:
        frame (np.ndarray): Original image (H x W x 3).
        p_map (np.ndarray): Probability map (grid_h x grid_w).
        grid_size (int): Grid cell size in pixels.
        alpha (float): Transparency of the overlay.
        show_text (bool): Whether to draw probability values in each cell.
        cmap (str): Matplotlib colormap name.
    """
    h, w = frame.shape[:2]
    grid_h, grid_w = p_map.shape

    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(frame)

    # Normalize p_map to [0,1] for colormap
    normed = np.clip(p_map, 0.0, 1.0)

    # Draw each cell with its color and value
    for gy in range(grid_h):
        for gx in range(grid_w):
            prob = normed[gy, gx]
            if prob > 0:
                x = gx * grid_size
                y = gy * grid_size
                rect = plt.Rectangle((x, y), grid_size, grid_size,
                                     color=plt.cm.get_cmap(cmap)(prob), alpha=alpha)
                ax.add_patch(rect)

                if show_text:
                    ax.text(x + grid_size / 2, y + grid_size / 2,
                            f"{prob:.2f}",
                            ha='center', va='center',
                            fontsize=6, color='black')

    # Draw grid lines
    for gx in range(0, w, grid_size):
        ax.axvline(gx, color='white', lw=0.5, alpha=0.5)
    for gy in range(0, h, grid_size):
        ax.axhline(gy, color='white', lw=0.5, alpha=0.5)

    ax.set_xlim([0, w])
    ax.set_ylim([h, 0])
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_multiple_p_maps(frames, p_maps, grid_size, alpha=0.4, cols=4, show_text=False, cmap='Greens'):
    """
    Render multiple frames with overlaid probability maps in a grid layout.

    Args:
        frames (List[np.ndarray]): List of images (H x W x 3).
        p_maps (List[np.ndarray]): List of corresponding 2D probability maps.
        grid_size (int): Size of each grid cell in pixels.
        alpha (float): Transparency of overlay.
        cols (int): Number of columns in the subplot grid.
        show_text (bool): Show probability values.
        cmap (str): Matplotlib colormap.
    """
    assert len(frames) == len(p_maps), "Each frame must have a corresponding p_map"

    num_frames = len(frames)
    rows = (num_frames + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = np.reshape(axes, (rows, cols))

    for idx, (frame, p_map) in enumerate(zip(frames, p_maps)):
        ax = axes[idx // cols, idx % cols]
        h, w = frame.shape[:2]
        ax.imshow(frame)

        normed = np.clip(p_map, 0.0, 1.0)
        grid_h, grid_w = p_map.shape

        for gy in range(grid_h):
            for gx in range(grid_w):
                prob = normed[gy, gx]
                if prob > 0:
                    x = gx * grid_size
                    y = gy * grid_size
                    rect = plt.Rectangle((x, y), grid_size, grid_size,
                                         color=plt.cm.get_cmap(cmap)(prob), alpha=alpha)
                    ax.add_patch(rect)
                    if show_text:
                        ax.text(x + grid_size / 2, y + grid_size / 2,
                                f"{prob:.2f}", ha='center', va='center',
                                fontsize=5, color='black')

        for gx in range(0, w, grid_size):
            ax.axvline(gx, color='white', lw=0.3, alpha=0.5)
        for gy in range(0, h, grid_size):
            ax.axhline(gy, color='white', lw=0.3, alpha=0.5)

        ax.set_title(f"Frame {idx}")
        ax.set_xlim([0, w])
        ax.set_ylim([h, 0])
        ax.axis('off')

    # Hide unused axes
    for i in range(num_frames, rows * cols):
        axes[i // cols, i % cols].axis('off')

    plt.tight_layout()
    plt.show()

@dataclass(frozen=True)
class TrackNode:
    """
    Represents a detection node in the tracking graph.

    Attributes:
        frame_id (int): Frame index where detection occurred.
        grid_cell_id (int): Discretized grid cell ID of detection center.
        position (tuple): Grid coordinates (x_bin, y_bin).
        confidence (float): Detection confidence score.
    """

    frame_id: int
    grid_cell_id: int
    det_idx: int
    position: tuple
    confidence: float
    dets: Any

    def __hash__(self) -> int:
        """Generate hash using frame and grid cell."""
        return hash((self.frame_id, self.grid_cell_id))

    def __eq__(self, other: Any) -> bool:
        """Compare nodes by frame and grid cell ID."""
        if not isinstance(other, TrackNode):
            return False
        return (self.frame_id, self.grid_cell_id) == (other.frame_id, other.grid_cell_id)
    
    def __str__(self):
        return str(self.frame_id) + " " + str(self.det_idx)


class KSPTracker(BaseTracker):
    """
    Offline tracker using K-Shortest Paths (KSP) algorithm.

    Attributes:
        grid_size (int): Size of each grid cell (pixels).
        min_confidence (float): Minimum detection confidence to consider.
        max_distance (float): Maximum spatial distance between nodes to connect.
        max_paths (int): Maximum number of paths (tracks) to find.
    """

    def __init__(
        self,
        grid_size: int = 25,
        max_paths: int = 20,
        min_confidence: float = 0.3,
        max_distance: float = 0.3,
        img_dim: tuple = (512, 512),
    ) -> None:
        """
        Initialize the KSP tracker.

        Args:
            grid_size (int): Pixel size of grid cells.
            max_paths (int): Max number of paths to find.
            min_confidence (float): Minimum confidence to keep detection.
            max_distance (float): Max allowed spatial distance between nodes.
        """
        self.grid_size = grid_size
        self.min_confidence = min_confidence
        self.max_distance = max_distance
        self.max_paths = max_paths
        self.img_dim = img_dim
        self.frames = []
        self.G = nx.DiGraph()
        self.reset()

    def set_image_dim(self, dim: tuple):
        self.img_dim = dim

    def reset(self) -> None:
        """Reset the detection buffer."""
        self.detection_buffer: List[sv.Detections] = []

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Append new detections to the buffer.

        Args:
            detections (sv.Detections): Detections for the current frame.

        Returns:
            sv.Detections: The same detections passed in.
        """
        self.detection_buffer.append(detections)
        return detections

    def _discretized_grid_cell_id(self, bbox: np.ndarray) -> tuple:
        """
        Compute discretized grid cell coordinates from bbox center.

        Args:
            bbox (np.ndarray): Bounding box [x1, y1, x2, y2].

        Returns:
            tuple: (grid_x, grid_y) discretized coordinates.
        """
        x_center = (bbox[2] + bbox[0]) / 2
        y_center = (bbox[3] + bbox[1]) / 2
        grid_x = int(x_center // self.grid_size)
        grid_y = int(y_center // self.grid_size)
        return (grid_x, grid_y)
    
    def get_overlapped_cells(self, bbox: np.ndarray) -> list:
        """
        Return all grid cells overlapped by the bounding box.

        Args:
            bbox (np.ndarray): [x1, y1, x2, y2]

        Returns:
            List of tuples [(grid_x, grid_y), ...]
        """
        x1, y1, x2, y2 = bbox
        grid_x1 = int(x1 // self.grid_size)
        grid_y1 = int(y1 // self.grid_size)
        grid_x2 = int(x2 // self.grid_size)
        grid_y2 = int(y2 // self.grid_size)

        overlapped_cells = []
        for gx in range(grid_x1, grid_x2 + 1):
            for gy in range(grid_y1, grid_y2 + 1):
                overlapped_cells.append((gx, gy))
        return overlapped_cells
    
    def get_node_probability(self, node: TrackNode) -> float:
        """
        Retrieve the probability from p_map for the given TrackNode.

        Args:
            node (TrackNode): The node, with a .position attribute as (grid_x, grid_y).

        Returns:
            float: Probability value at the node's grid cell.
        """
        frame_id = node.frame_id
        grid_x, grid_y = node.position

        if frame_id < 0 or frame_id >= len(self.p_maps):
            return 0.0  # Invalid frame index

        p_map = self.p_maps[frame_id]
        grid_height, grid_width = p_map.shape

        if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
            return float(p_map[grid_y, grid_x])  # Note row=y, col=x in numpy indexing
        else:
            return 0.0  # Out of bounds

    def _edge_cost(self, node: TrackNode, confidence: float, dist: float) -> float:
        """
        Compute edge cost from detection confidence.

        Args:
            confidence (float): Detection confidence score.

        Returns:
            float: Edge cost for KSP (non-negative after transform).
        """
        pu = self.get_node_probability(node)
        print(pu)
        return -np.log10(pu / (1 - pu))

    def build_probability_maps(self, all_detections):
        """
        Build a list of probability maps, one per frame.

        Args:
            all_detections (list of list of dict):
                Each element is a list of detections for a frame,
                each detection dict has:
                    'xyxy': [x1, y1, x2, y2]
                    'confidence': float between 0 and 1

        Returns:
            list of np.ndarray: Each element is a 2D probability map for that frame.
        """

        img_width, img_height = self.img_dim

        grid_width = (img_width + self.grid_size - 1) // self.grid_size
        grid_height = (img_height + self.grid_size - 1) // self.grid_size

        all_p_maps = []

        for dets in all_detections:
            p_map = np.zeros((grid_height, grid_width), dtype=np.float32)

            # If no detections, just append zero map
            if dets.is_empty():
                all_p_maps.append(p_map)
                continue

            for i in range(len(dets)):
                bbox = dets.xyxy[i]      # tensor or numpy array [x1,y1,x2,y2]
                conf = dets.confidence[i]  # tensor or numpy scalar

                # Convert bbox to numpy if needed
                if hasattr(bbox, 'cpu'):
                    bbox = bbox.cpu().numpy()
                if hasattr(conf, 'item'):
                    conf = conf.item()

                x1, y1, x2, y2 = bbox

                grid_x1 = int(x1 // self.grid_size)
                grid_y1 = int(y1 // self.grid_size)
                grid_x2 = int(x2 // self.grid_size)
                grid_y2 = int(y2 // self.grid_size)

                grid_x1 = max(0, min(grid_x1, grid_width - 1))
                grid_x2 = max(0, min(grid_x2, grid_width - 1))
                grid_y1 = max(0, min(grid_y1, grid_height - 1))
                grid_y2 = max(0, min(grid_y2, grid_height - 1))

                for gx in range(grid_x1, grid_x2 + 1):
                    for gy in range(grid_y1, grid_y2 + 1):
                        p_map[gy, gx] = max(p_map[gy, gx], conf)

            all_p_maps.append(p_map)

        self.p_maps = all_p_maps

    def _build_graph(self, all_detections: List[sv.Detections]) -> None:
        self.build_probability_maps(all_detections=all_detections)

        self.G = nx.DiGraph()
        self.G.add_node("source")
        self.G.add_node("sink")

        # self.node_to_detection: Dict[TrackNode, tuple] = {}
        node_dict: Dict[int, List[TrackNode]] = {}

        for frame_idx, dets in enumerate(all_detections):
            node_dict[frame_idx] = []

            # Sort detections by (x1, y1) top-left corner of bbox
            if len(dets) > 0:
                # Get an array of [x1, y1]
                coords = np.array([[bbox[0], bbox[1]] for bbox in dets.xyxy])
                sorted_indices = np.lexsort((coords[:, 1], coords[:, 0]))  # sort by x then y
            else:
                sorted_indices = []

            # Build nodes in sorted order for stable det_idx
            for new_det_idx, original_det_idx in enumerate(sorted_indices):
                if dets.confidence[original_det_idx] < self.min_confidence:
                    continue

                pos = self._discretized_grid_cell_id(np.array(dets.xyxy[original_det_idx]))
                cell_id = hash(pos)

                node = TrackNode(
                    frame_id=frame_idx,
                    grid_cell_id=cell_id,
                    det_idx=new_det_idx,  # use stable new_det_idx
                    position=pos,
                    dets=dets,
                    confidence=dets.confidence[original_det_idx],
                )

                self.G.add_node(node)
                node_dict[frame_idx].append(node)
                # self.node_to_detection[node] = (frame_idx, original_det_idx)  # map to original det_idx

                if frame_idx == 0:
                    self.G.add_edge("source", node, weight=0)
                if frame_idx == len(all_detections) - 1:
                    self.G.add_edge(node, "sink", weight=0)

        # Add edges as before
        for i in range(len(all_detections) - 1):
            for node in node_dict[i]:
                for node_next in node_dict[i + 1]:
                    dist = np.linalg.norm(np.array(node.position) - np.array(node_next.position))
                    w = self._edge_cost(node, confidence=node.confidence, dist=dist)
                    print(w)
                    self.G.add_edge(
                        node,
                        node_next,
                        weight=w,
                    )

                
    def _update_detections_with_tracks(
        self, assignments: List[List[TrackNode]]
    ) -> List[sv.Detections]:
        """
        Assign track IDs to detections based on spatially consistent paths,
        with debug output per frame.

        Args:
            assignments (List[List[TrackNode]]): List of detection paths.

        Returns:
            List[sv.Detections]: Detections with assigned tracker IDs.
        """
        print(len(assignments))
        all_detections = []

        assigned = set()
        assignment_map = {}

        # Map (frame_id, grid_cell_id) to track ID
        for track_id, path in enumerate(assignments, start=1):
            for node in path:
                if node in {"source", "sink"}:
                    continue
                
                key = (node.frame_id, node.grid_cell_id)
                if key not in assigned:
                    assignment_map[key] = track_id
                    assigned.add(key)
        
        import pprint
        p = pprint.PrettyPrinter(4)
        p.pprint(assignment_map)

        for frame_idx, dets in enumerate(self.detection_buffer):
            frame_tracker_ids = [-1] * len(dets)
            det_pos_to_idx = {}

            for det_idx in range(len(dets)):
                pos = self._discretized_grid_cell_id(np.array(dets.xyxy[det_idx]))
                det_pos_to_idx[pos] = det_idx

            for det_idx in range(len(dets)):
                pos = self._discretized_grid_cell_id(np.array(dets.xyxy[det_idx]))
                key = (frame_idx, hash(pos))
                print( "Frame: "+ str(frame_idx), key, det_pos_to_idx[pos], assignment_map[key] if key in assignment_map else "NIL")
                if key in assignment_map:
                    frame_tracker_ids[det_pos_to_idx[pos]] = assignment_map[key]

            dets.tracker_id = np.array(frame_tracker_ids)
            all_detections.append(dets)

            # Debug output for this frame
            num_tracked = sum(tid != -1 for tid in frame_tracker_ids)
            print(f"[Frame {frame_idx}] Total Detections: {len(frame_tracker_ids)} | Tracked: {num_tracked}")
            for det_idx, tid in enumerate(frame_tracker_ids):
                bbox = dets.xyxy[det_idx]
                conf = dets.confidence[det_idx]
                if tid == -1:
                    print(f"  ⛔ Untracked Detection {det_idx}: BBox={bbox}, Conf={conf:.2f}")
                else:
                    print(f"  ✅ Track {tid} assigned to Detection {det_idx}: BBox={bbox}, Conf={conf:.2f}")

        frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in self.frames]
        InteractivePMapViewer(frames_rgb, self.p_maps, grid_size=self.grid_size, show_text=True)
        return all_detections

    def _shortest_path(self) -> tuple:
        """
        Compute shortest path from 'source' to 'sink' using Bellman-Ford.

        Returns:
            tuple: (path, total_cost, lengths)

        Raises:
            RuntimeError: If negative cycle detected.
            KeyError: If sink unreachable.
        """
        try:
            lengths, paths = nx.single_source_bellman_ford(self.G, "source", weight="weight")
            if "sink" not in paths:
                raise KeyError("No path found from source to sink.")
            return paths["sink"], lengths["sink"], lengths
        except nx.NetworkXUnbounded:
            raise RuntimeError("Graph contains a negative weight cycle.")

    def _extend_graph(self, paths: list[list]):
        """
        Extend the graph as per Berclaz et al. (2011), Table 4:
        - Split each used node (except source/sink) into `v_in` and `v_out`
        - Add a zero-cost edge v_in → v_out
        - Redirect edges accordingly
        - Reverse and negate all path edges

        Args:
            paths (List[List[TrackNode]]): Previously found paths.

        Returns:
            nx.DiGraph: Extended graph with node splits and reversed edges.
        """
        G_ext = self.G.copy()
        split_map = {}

        # Step 1: Split internal nodes (not source/sink)
        for path in paths:
            for node in path:
                if node in {"source", "sink"}:
                    continue
                if node in split_map:
                    continue  # Already split

                # Deepcopy node to keep all attributes, add split_tag for identification
                v_in = deepcopy(node)
                v_out = deepcopy(node)
                object.__setattr__(v_in, "split_tag", "in")
                object.__setattr__(v_out, "split_tag", "out")
                split_map[node] = (v_in, v_out)

                # Add new nodes
                G_ext.add_node(v_in)
                G_ext.add_node(v_out)

                # Add auxiliary zero-cost edge v_in -> v_out
                G_ext.add_edge(v_in, v_out, weight=0)

                # Redirect incoming edges to v_in
                for u, _, data in list(G_ext.in_edges(node, data=True)):
                    G_ext.add_edge(u, v_in, **data)
                    G_ext.remove_edge(u, node)

                # Redirect outgoing edges from v_out
                for _, v, data in list(G_ext.out_edges(node, data=True)):
                    G_ext.add_edge(v_out, v, **data)
                    G_ext.remove_edge(node, v)

                # Remove original node
                G_ext.remove_node(node)

        # Step 2: Reverse and negate all edges in used paths
        for path in paths:
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]

                # Use split nodes if they exist
                u_out = split_map[u][1] if u in split_map else u
                v_in = split_map[v][0] if v in split_map else v

                if G_ext.has_edge(u_out, v_in):
                    cost = G_ext[u_out][v_in]["weight"]
                    G_ext.remove_edge(u_out, v_in)
                    G_ext.add_edge(v_in, u_out, weight=-cost)

        return G_ext

    def _transform_edge_cost(
        self, G: nx.DiGraph, shortest_costs: Dict[Any, float]
    ) -> nx.DiGraph:
        """
        Apply cost transformation to ensure non-negative edge weights.

        Args:
            G (nx.DiGraph): Graph with possibly negative weights.
            shortest_costs (dict): Shortest path distances from source.

        Returns:
            nx.DiGraph: Cost-transformed graph.
        """
        Gc = nx.DiGraph()
        for u, v, data in G.edges(data=True):
            if u not in shortest_costs or v not in shortest_costs:
                continue
            original = data["weight"]
            transformed = abs(original + shortest_costs[u] - shortest_costs[v])
            Gc.add_edge(u, v, weight=transformed)
        return Gc

    def _interlace_paths(
        self, current_paths: List[List[TrackNode]], new_path: List[TrackNode]
    ) -> List[TrackNode]:
        """
        Remove nodes from new_path that conflict with current_paths.

        Args:
            current_paths (List[List[TrackNode]]): Existing disjoint paths.
            new_path (List[TrackNode]): New candidate path.

        Returns:
            List[TrackNode]: Interlaced path without conflicts.
        """
        used_nodes = set()
        for path in current_paths:
            for node in path:
                if isinstance(node, TrackNode):
                    used_nodes.add((node.frame_id, node.grid_cell_id))

        interlaced = []
        for node in new_path:
            if isinstance(node, TrackNode):
                key = (node.frame_id, node.grid_cell_id)
                if key not in used_nodes:
                    interlaced.append(node)

        return interlaced

    def ksp(self) -> List[List[TrackNode]]:
        """
        Compute k disjoint shortest paths using KSP algorithm.

        Returns:
            List[List[TrackNode]]: List of disjoint detection paths.
        """
        
        path, cost, lengths = self._shortest_path()
        P = [path]
        cost_P = []
        print("Cost: " + str(cost))
        visualize_tracking_graph_with_path_pyvis(self.G, P[-1], "graph1.html")

        for l in range(1, self.max_paths):
            if l > 2 and cost_P[-1] >= cost_P[-2]:
                return P  # early termination

            Gl = self._extend_graph(P)
            Gc_l = self._transform_edge_cost(Gl, lengths)

            try:
                lengths, paths_dict = nx.single_source_dijkstra(Gc_l, "source", weight="weight")

                if "sink" not in paths_dict:
                    break

                new_path = paths_dict["sink"]
                cost_P.append(lengths["sink"])

                interlaced_path = self._interlace_paths(P, new_path)
                P.append(interlaced_path)
                visualize_tracking_graph_with_path_pyvis(Gc_l, P[-1], "graph" + str(len(P)) + ".html")
            except nx.NetworkXNoPath:
                break
        print(len(P))
        return P

    def process_tracks(self) -> List[sv.Detections]:
        """
        Run the tracking algorithm and assign track IDs to detections.

        Returns:
            List[sv.Detections]: Detections updated with tracker IDs.
        """
        self._build_graph(self.detection_buffer)
        # visualize_tracking_graph_debug(self.G)
        # detections_list  = find_and_visualize_disjoint_paths(self.G)
        # return detections_list 
        disjoint_paths = self.ksp()
        return self._update_detections_with_tracks(assignments=disjoint_paths)
