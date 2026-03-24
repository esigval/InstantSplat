import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial import ConvexHull


SUPPORTED_MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")


def slugify_label(label_name: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in label_name.strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "mask"


def resolve_mask_dir(source_path: str, override_dir: str = "") -> str:
    if override_dir:
        return os.path.abspath(override_dir)
    return os.path.join(os.path.abspath(source_path), "images", ".mask")


def find_mask_path(mask_dir: str, image_name: str) -> str | None:
    stem = Path(image_name).stem
    for extension in SUPPORTED_MASK_EXTENSIONS:
        candidate = os.path.join(mask_dir, stem + extension)
        if os.path.exists(candidate):
            return candidate
    return None


def load_mask_tensor(mask_path: str, width: int, height: int, device: torch.device) -> torch.Tensor:
    with Image.open(mask_path) as image:
        mask = image.convert("L")
        if mask.size != (width, height):
            mask = mask.resize((width, height), resample=Image.NEAREST)
        mask_array = np.array(mask, dtype=np.uint8, copy=True)
    return torch.from_numpy(mask_array).to(device=device)


def compute_semantic_score(positive_views: np.ndarray, visible_views: np.ndarray) -> np.ndarray:
    score = np.zeros(positive_views.shape[0], dtype=np.float32)
    has_visible = visible_views > 0
    score[has_visible] = positive_views[has_visible] / visible_views[has_visible]
    return score


def build_selected_mask(
    positive_views: np.ndarray,
    visible_views: np.ndarray,
    min_visible_views: int,
    min_positive_views: int,
    min_score: float,
) -> tuple[np.ndarray, np.ndarray]:
    score = compute_semantic_score(positive_views, visible_views)
    selected_mask = (
        (visible_views >= int(min_visible_views))
        & (positive_views >= int(min_positive_views))
        & (score >= float(min_score))
    )
    return selected_mask, score


def append_semantic_fields(vertex_data, selected_mask, positive_views, visible_views, score):
    new_dtype = vertex_data.dtype.descr + [
        ("semantic_label", "u1"),
        ("semantic_positive_views", "i4"),
        ("semantic_visible_views", "i4"),
        ("semantic_score", "f4"),
    ]
    labeled = np.empty(vertex_data.shape, dtype=new_dtype)
    for name in vertex_data.dtype.names:
        labeled[name] = vertex_data[name]
    labeled["semantic_label"] = selected_mask.astype(np.uint8)
    labeled["semantic_positive_views"] = positive_views.astype(np.int32)
    labeled["semantic_visible_views"] = visible_views.astype(np.int32)
    labeled["semantic_score"] = score.astype(np.float32)
    return labeled


def write_filtered_ply(path: str, vertex_data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    PlyData([PlyElement.describe(vertex_data, "vertex")], text=False).write(path)


def _convex_hull_indices(points_2d: np.ndarray) -> np.ndarray | None:
    if points_2d.shape[0] < 3:
        return None
    deduped, unique_indices = np.unique(np.round(points_2d, decimals=6), axis=0, return_index=True)
    if deduped.shape[0] < 3:
        return None
    hull = ConvexHull(deduped)
    return unique_indices[hull.vertices]


def estimate_plane_outline(xyz: np.ndarray, max_points: int = 10000):
    if xyz.shape[0] < 3:
        return None

    if xyz.shape[0] > max_points:
        step = int(np.ceil(xyz.shape[0] / max_points))
        xyz = xyz[::step]

    origin = xyz.mean(axis=0)
    centered = xyz - origin
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    basis_u = vh[0]
    basis_v = vh[1]
    normal = vh[2]
    projected_2d = np.stack([centered @ basis_u, centered @ basis_v], axis=1)

    hull_indices = _convex_hull_indices(projected_2d)
    if hull_indices is None:
        return None

    hull_2d = projected_2d[hull_indices]
    hull_3d = origin + np.outer(hull_2d[:, 0], basis_u) + np.outer(hull_2d[:, 1], basis_v)

    return {
        "point_count": int(xyz.shape[0]),
        "plane_origin": origin.tolist(),
        "plane_normal": normal.tolist(),
        "basis_u": basis_u.tolist(),
        "basis_v": basis_v.tolist(),
        "singular_values": singular_values.tolist(),
        "outline_2d": hull_2d.tolist(),
        "outline_3d": hull_3d.tolist(),
    }


def write_outline_json(path: str, outline_payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(outline_payload, handle, indent=2)
