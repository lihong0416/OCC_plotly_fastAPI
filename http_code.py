from __future__ import annotations

import os
import re
import io
import threading
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import trimesh

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None


# =========================
# 路径：相对本文件，避免 uvicorn 工作目录变化导致找不到文件
# =========================
BASE_DIR = Path(__file__).resolve().parent

STEP_PATH = BASE_DIR / "DPUB-501020596-XAX_03.002.stp"
GLB_PATH  = BASE_DIR / "exports_actual_catias/DPUB-501020596-XAX_03.002.glb"
OUT_DIR   = BASE_DIR / "exports_pv_parts"

# STEP 读取编码（你之前 66 个焊点能读出来通常用 latin-1 最稳）
STEP_ENCODING = "latin-1"

# GLB 合并 chunk
MERGE_CHUNK = 300

# 最终输出图尺寸
VIEW_IMG_SIZE = (2200, 1400)  # (w,h)
VIEW_ZOOM = 1.25

# 选视角打分的低分辨率（更快）
SCORE_IMG_SIZE = (900, 600)
SCORE_ZOOM = 1.15

# 焊点球体参数
WELD_SPHERE_RADIUS_MM = None      # None=自动估计
WELD_SPHERE_RES = 18

# 异常焊点更大（确保“红点看得见”）
HIGHLIGHT_SCALE = 2.0

# 配色
BODY_COLOR = "#B8B8B8"
WELD_NORMAL_COLOR = "blue"
WELD_BAD_COLOR = "red"

# 引线
CALLOUT_LINE_COLOR = (220, 0, 0)   # 红色
CALLOUT_LINE_WIDTH = 5

# ID 框样式
ID_BG = (255, 255, 255, 235)
ID_BORDER = (40, 40, 40, 200)
ID_SHADOW = (0, 0, 0, 60)
ID_TEXT_COLOR = (15, 15, 15, 255)

# 徽标（封样编号）独立圆框：绿底红边
BADGE_FILL = (120, 200, 120, 235)
BADGE_BORDER = (220, 0, 0, 235)
BADGE_TEXT = (0, 0, 0, 255)


# =============================================================================
# value(封样编号) 强制显示为整数（87.6 -> 87）
# =============================================================================
def _format_badge_value(v: Optional[str]) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    if s == "":
        return ""
    try:
        return str(int(float(s)))
    except Exception:
        return s


# =============================================================================
# STEP 提取焊点：只抓名字包含 RSW 的 CARTESIAN_POINT
# =============================================================================
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def extract_rsw_points(step_file: Path, encoding: str = "latin-1") -> pd.DataFrame:
    text = step_file.read_text(encoding=encoding, errors="ignore")
    pattern = re.compile(
        r"#\d+\s*=\s*CARTESIAN_POINT\(\s*'([^']*RSW[^']*)'\s*,\s*\(\s*([^)]+?)\s*\)\s*\)\s*;",
        re.IGNORECASE | re.DOTALL,
    )
    rows = []
    for name, coords_str in pattern.findall(text):
        nums = _NUM_RE.findall(coords_str)
        if len(nums) < 3:
            continue
        try:
            x, y, z = float(nums[0]), float(nums[1]), float(nums[2])
        except Exception:
            continue
        rows.append({"name": str(name).strip(), "x": x, "y": y, "z": z})
    return pd.DataFrame(rows)


# =============================================================================
# GLB -> trimesh（保留 transform、过滤点云、chunk 合并）
# =============================================================================
def _scene_meshes_with_transform(scene: trimesh.Scene):
    meshes = []
    nodes_geo = scene.graph.nodes_geometry
    if isinstance(nodes_geo, dict):
        node_iter = [(node, geom_name) for node, geom_name in nodes_geo.items()]
    else:
        node_iter = [(node, None) for node in list(nodes_geo)]

    for node_name, geom_name in node_iter:
        try:
            T, gname_from_graph = scene.graph.get(node_name)
        except Exception:
            continue

        geom_name = geom_name or gname_from_graph
        if geom_name is None:
            continue

        geom = scene.geometry.get(geom_name, None)
        if geom is None:
            continue
        if isinstance(geom, trimesh.PointCloud):
            continue
        if not isinstance(geom, trimesh.Trimesh):
            continue
        if geom.faces is None or len(geom.faces) == 0:
            continue

        m = geom.copy()
        if T is not None:
            try:
                m.apply_transform(T)
            except Exception:
                pass

        meshes.append(m)
    return meshes


def _chunked_concatenate(meshes, chunk: int = 300) -> trimesh.Trimesh:
    if len(meshes) == 0:
        raise ValueError("No mesh geometries found in GLB.")
    if len(meshes) == 1:
        return meshes[0]

    chunks = []
    total = len(meshes)
    n_chunks = (total + chunk - 1) // chunk
    for ci, i in enumerate(range(0, total, chunk), start=1):
        part = trimesh.util.concatenate(meshes[i: i + chunk])
        chunks.append(part)
        if ci == 1 or ci % 10 == 0 or ci == n_chunks:
            print(f"  merged chunk {ci}/{n_chunks} (parts {i}~{min(i+chunk, total)-1})")

    if len(chunks) == 1:
        return chunks[0]
    return trimesh.util.concatenate(chunks)


def load_glb_as_single_mesh(glb_path: Path, chunk: int = 300) -> trimesh.Trimesh:
    print(f"\n[GLB] Loading: {glb_path}")
    obj = trimesh.load(glb_path, force="scene")

    if isinstance(obj, trimesh.Scene):
        print(f"  Scene geometries: {len(obj.geometry)}")
        meshes = _scene_meshes_with_transform(obj)
        print(f"  Mesh parts kept: {len(meshes)}")
        face_counts = sorted([len(m.faces) for m in meshes])
        if face_counts:
            print(f"  faces min/median/max: {face_counts[0]} {face_counts[len(face_counts)//2]} {face_counts[-1]}")
    elif isinstance(obj, trimesh.Trimesh):
        meshes = [obj]
        print("  Loaded a single Trimesh")
    else:
        raise TypeError(f"Unsupported loaded object type: {type(obj)}")

    print("[GLB] Merging parts (chunked).")
    mesh_body = _chunked_concatenate(meshes, chunk=chunk)

    print("[GLB] Cleaning mesh.")
    for fn in ("remove_degenerate_faces", "merge_vertices", "fix_normals"):
        try:
            getattr(mesh_body, fn)()
        except Exception:
            pass

    print(f"✓ Final mesh: verts={len(mesh_body.vertices)}, faces={len(mesh_body.faces)}")
    print(f"  Bounds: {mesh_body.bounds}")
    return mesh_body


# =============================================================================
# trimesh -> pyvista
# =============================================================================
def trimesh_to_pyvista(mesh_body: trimesh.Trimesh):
    import pyvista as pv
    V = np.asarray(mesh_body.vertices, dtype=float)
    F = np.asarray(mesh_body.faces, dtype=np.int64)
    faces_pv = np.hstack([np.full((F.shape[0], 1), 3, dtype=np.int64), F]).ravel()
    mesh_pv = pv.PolyData(V, faces_pv)

    try:
        mesh_pv = mesh_pv.compute_normals(
            cell_normals=True,
            point_normals=True,
            consistent_normals=True,
            auto_orient_normals=True,
        )
    except Exception:
        pass

    return mesh_pv


def ensure_headless_pyvista():
    import pyvista as pv
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        try:
            pv.start_xvfb()
        except Exception:
            pass


def _estimate_weld_radius(mesh_body: trimesh.Trimesh) -> float:
    ext = np.asarray(mesh_body.extents, dtype=float)
    return max(2.0, float(ext.max()) * 0.004)


def build_weld_spheres(weld_points: np.ndarray, mesh_body: trimesh.Trimesh):
    import pyvista as pv
    weld_points = np.asarray(weld_points, dtype=float)
    cloud = pv.PolyData(weld_points)

    if WELD_SPHERE_RADIUS_MM is None:
        r = _estimate_weld_radius(mesh_body)
    else:
        r = float(WELD_SPHERE_RADIUS_MM)

    sphere = pv.Sphere(radius=r, theta_resolution=WELD_SPHERE_RES, phi_resolution=WELD_SPHERE_RES)
    spheres = cloud.glyph(geom=sphere, scale=False, orient=False)

    print(f"[WELD] Sphere radius = {r:.3f} (model units)")
    return spheres, r


def build_single_sphere(center_xyz: np.ndarray, radius: float):
    import pyvista as pv
    c = np.asarray(center_xyz, dtype=float).reshape(3,)
    return pv.Sphere(
        radius=float(radius),
        center=tuple(c),
        theta_resolution=WELD_SPHERE_RES,
        phi_resolution=WELD_SPHERE_RES,
    )


# =============================================================================
# 视角（6 个正交面）
# 注意：view_vec 是“相机->焦点”的方向向量
# =============================================================================
def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3,)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return v / n


def _choose_viewup(view_vec: np.ndarray) -> np.ndarray:
    v = _normalize(view_vec)
    if abs(v[2]) > 0.85:  # 接近 z 轴
        return np.array([0.0, 1.0, 0.0], dtype=float)
    return np.array([0.0, 0.0, 1.0], dtype=float)


# 六面候选
CANDIDATE_VIEWS = {
    "front":  np.array([0.0,  1.0, 0.0]),  # +Y
    "back":   np.array([0.0, -1.0, 0.0]),  # -Y
    "right":  np.array([1.0,  0.0, 0.0]),  # +X
    "left":   np.array([-1.0, 0.0, 0.0]),  # -X
    "top":    np.array([0.0,  0.0, 1.0]),  # +Z
    "bottom": np.array([0.0,  0.0,-1.0]),  # -Z
}
VIEW_PRIORITY = ["front", "back", "right", "left", "top", "bottom"]


# =============================================================================
# 世界坐标 -> 屏幕坐标（用于判断“在不在画面中”、以及绘制标注）
# =============================================================================
def _world_to_display_xyz(plotter, world_xyz: np.ndarray):
    try:
        ren = plotter.renderer
        x, y, z = float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2])
        ren.SetWorldPoint(x, y, z, 1.0)
        ren.WorldToDisplay()
        dx, dy, dz = ren.GetDisplayPoint()
        return float(dx), float(dy), float(dz)
    except Exception:
        return None


def _mesh_bottom_in_pil(plotter, mesh_pv) -> Optional[float]:
    try:
        w, h = plotter.window_size
    except Exception:
        w, h = VIEW_IMG_SIZE

    b = mesh_pv.bounds
    xmin, xmax, ymin, ymax, zmin, zmax = map(float, b)
    corners = np.array([
        [xmin, ymin, zmin], [xmin, ymin, zmax],
        [xmin, ymax, zmin], [xmin, ymax, zmax],
        [xmax, ymin, zmin], [xmax, ymin, zmax],
        [xmax, ymax, zmin], [xmax, ymax, zmax],
    ], dtype=float)

    ys = []
    for c in corners:
        d = _world_to_display_xyz(plotter, c)
        if d is None:
            continue
        _dx, dy, _dz = d
        ys.append(float(h - dy))  # PIL y向下
    if not ys:
        return None
    return float(max(ys))


# =============================================================================
# 2D 标注：折线 + ID 框 + 徽标（徽标不和 ID 挤一起）
# =============================================================================
def _draw_callout_id_and_badge(
    pil_img: Image.Image,
    px: float,
    py: float,
    weld_id: str,
    badge_text: Optional[str],
    min_elbow_y: Optional[float],
):
    if ImageDraw is None:
        return

    w, h = pil_img.size
    draw = ImageDraw.Draw(pil_img, "RGBA")

    try:
        font_id = ImageFont.truetype("DejaVuSans.ttf", 26)
        font_badge = ImageFont.truetype("DejaVuSans.ttf", 28)
    except Exception:
        font_id = ImageFont.load_default()
        font_badge = ImageFont.load_default()

    # 折线尽量往下（并强制到模型底部以下）
    base_down = 620
    horiz = 520
    margin = 22

    elbow_y = py + base_down
    if min_elbow_y is not None:
        elbow_y = max(elbow_y, float(min_elbow_y) + 90)  # ✅ 永远落在模型底部以下
    elbow_y = min(h - 180, elbow_y)
    elbow_y = max(margin, elbow_y)

    # 文本块左右自动放置
    if px < w * 0.55:
        box_anchor_x = px + horiz
    else:
        box_anchor_x = px - horiz
    box_anchor_x = max(margin, min(box_anchor_x, w - margin))

    p1 = (int(px), int(py))
    p2 = (int(px), int(elbow_y))
    p3 = (int(box_anchor_x), int(elbow_y))

    draw.line([p1, p2, p3], fill=CALLOUT_LINE_COLOR + (235,), width=CALLOUT_LINE_WIDTH)

    # ID 框
    text = weld_id
    bb = draw.textbbox((0, 0), text, font=font_id)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]

    pad_x, pad_y = 16, 12
    radius = 12

    if p3[0] >= p2[0]:
        tx = p3[0] + 18
    else:
        tx = p3[0] - 18 - (tw + 2 * pad_x)
    ty = p3[1] - int(th * 0.5)

    tx = max(margin, min(tx, w - margin - (tw + 2 * pad_x)))
    ty = max(margin, min(ty, h - margin - (th + 2 * pad_y)))

    x0 = tx - pad_x
    y0 = ty - pad_y
    x1 = tx + tw + pad_x
    y1 = ty + th + pad_y

    sx, sy = 4, 4
    draw.rounded_rectangle([x0 + sx, y0 + sy, x1 + sx, y1 + sy], radius=radius, fill=ID_SHADOW)
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=ID_BG, outline=ID_BORDER, width=2)
    draw.text((tx, ty), text, fill=ID_TEXT_COLOR, font=font_id)

    # 徽标（封样编号）在 ID 框旁边独立圆框
    if not badge_text:
        return
    badge_text = str(badge_text).strip()

    bb2 = draw.textbbox((0, 0), badge_text, font=font_badge)
    bw, bh = bb2[2] - bb2[0], bb2[3] - bb2[1]
    R = int(max(22, (max(bw, bh) * 0.75) + 14))
    gap = 14

    cy = int((y0 + y1) * 0.5)
    cx_left = int(x0 - gap - R)
    cx_right = int(x1 + gap + R)

    if cx_left - R >= margin:
        cx = cx_left
    elif cx_right + R <= w - margin:
        cx = cx_right
    else:
        cx = margin + R

    draw.ellipse([cx - R + 3, cy - R + 3, cx + R + 3, cy + R + 3], fill=ID_SHADOW)
    draw.ellipse([cx - R, cy - R, cx + R, cy + R], fill=BADGE_FILL, outline=BADGE_BORDER, width=6)

    tx2 = cx - bw / 2
    ty2 = cy - bh / 2
    draw.text((tx2, ty2), badge_text, fill=BADGE_TEXT, font=font_badge)


# =============================================================================
# 视角打分：只要求“红点在画面里”（靠边没关系）
# 在画面内时，以红像素数量作为得分（越大越容易看见红点）
# =============================================================================
def _score_red_visibility(img_rgb: np.ndarray) -> int:
    r = img_rgb[..., 0].astype(np.int16)
    g = img_rgb[..., 1].astype(np.int16)
    b = img_rgb[..., 2].astype(np.int16)
    mask = (r > 170) & (g < 90) & (b < 90)
    return int(mask.sum())


def _render_score_for_view(mesh_pv, bad_xyz, bad_radius, view_name: str, view_vec: np.ndarray) -> Tuple[float, Dict]:
    import pyvista as pv
    ensure_headless_pyvista()

    p = pv.Plotter(off_screen=True, window_size=SCORE_IMG_SIZE)
    p.set_background("white")
    try:
        p.enable_anti_aliasing()
    except Exception:
        pass

    p.add_mesh(mesh_pv, color=BODY_COLOR, smooth_shading=True, show_edges=False, opacity=1.0)

    bad_sphere = build_single_sphere(bad_xyz, bad_radius)
    p.add_mesh(bad_sphere, color=(1.0, 0.0, 0.0), smooth_shading=False, show_edges=False, lighting=False)

    p.reset_camera()
    dist = float(getattr(p.camera, "distance", 1.0))

    focal = np.asarray(bad_xyz, dtype=float)
    v = _normalize(view_vec)
    up = _choose_viewup(v)

    p.camera.focal_point = tuple(focal)
    p.camera.position = tuple(focal - v * dist)
    p.camera.up = tuple(up)

    try:
        p.camera.zoom(SCORE_ZOOM)
    except Exception:
        pass

    try:
        p.render()
    except Exception:
        pass

    # 先判断焊点投影是否在画面内（只要在就行）
    disp = _world_to_display_xyz(p, bad_xyz)
    w, h = SCORE_IMG_SIZE
    if disp is None:
        p.close()
        return -1e9, {"view": view_name, "reason": "no_display_point"}

    dx, dy, _ = disp
    px = float(dx)
    py = float(h - dy)  # 转为 PIL 坐标系（y向下）

    margin = 2.0
    if not (-margin <= px <= w + margin and -margin <= py <= h + margin):
        p.close()
        return -1e9, {"view": view_name, "reason": "off_screen", "px": px, "py": py}

    img = p.screenshot(return_img=True)
    p.close()

    if img is None:
        return -1e9, {"view": view_name, "reason": "no_image"}

    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    img = img.astype(np.uint8)

    red_cnt = _score_red_visibility(img)
    return float(red_cnt), {"view": view_name, "red_cnt": red_cnt, "px": px, "py": py}


def _select_best_view(mesh_pv, bad_xyz: np.ndarray, bad_radius: float) -> Tuple[str, np.ndarray, np.ndarray]:
    best_name = "front"
    best_score = -1e18
    best_vec = _normalize(CANDIDATE_VIEWS["front"])
    best_up = _choose_viewup(best_vec)

    for name in VIEW_PRIORITY:
        vec = CANDIDATE_VIEWS[name]
        s, _meta = _render_score_for_view(mesh_pv, bad_xyz, bad_radius, name, vec)
        if s > best_score:
            best_score = s
            best_name = name
            best_vec = _normalize(vec)
            best_up = _choose_viewup(best_vec)

    return best_name, best_vec, _normalize(best_up)


# =============================================================================
# 渲染最终图：模型+蓝点+红点 + 标注（base64）
# =============================================================================
_ASSETS_LOCK = threading.Lock()
_RENDER_LOCK = threading.Lock()

_ASSETS: Dict[str, object] = {
    "loaded": False,
    "mesh_pv": None,
    "weld_id_to_xyz": None,
    "weld_spheres": None,
    "weld_radius": None,
}

# 缓存：key=(weld_id, badge, best_view)
_PNG_CACHE: Dict[Tuple[str, str, str], Tuple[float, bytes]] = {}
_PNG_CACHE_MAX = 256

# 焊点 -> 最佳视角缓存
_BEST_VIEW_CACHE: Dict[str, str] = {}


def _set_png_cache(weld_id: str, badge: str, view_name: str, png: bytes):
    key = (weld_id, badge, view_name)
    _PNG_CACHE[key] = (datetime.now().timestamp(), png)
    if len(_PNG_CACHE) > _PNG_CACHE_MAX:
        oldest = sorted(_PNG_CACHE.items(), key=lambda kv: kv[1][0])[: max(1, len(_PNG_CACHE) - _PNG_CACHE_MAX)]
        for k, _ in oldest:
            _PNG_CACHE.pop(k, None)


def _get_png_cache(weld_id: str, badge: str, view_name: str) -> Optional[bytes]:
    key = (weld_id, badge, view_name)
    return _PNG_CACHE.get(key, (None, None))[1]


def _png_to_data_url(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return "data:image/png;base64," + b64


def load_assets_once():
    with _ASSETS_LOCK:
        if _ASSETS["loaded"]:
            return

        if not STEP_PATH.exists():
            raise FileNotFoundError(f"STEP not found: {STEP_PATH}")
        if not GLB_PATH.exists():
            raise FileNotFoundError(f"GLB not found: {GLB_PATH}")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[OUT] Output dir: {OUT_DIR.resolve()}")

        print(f"\n[STEP] Loading weld points from: {STEP_PATH}")
        weld_df = extract_rsw_points(STEP_PATH, encoding=STEP_ENCODING)
        print(f"✓ Total weld points: {len(weld_df)}")
        if len(weld_df) == 0:
            raise RuntimeError("No weld points extracted from STEP. Check STEP content/encoding/RSW naming.")

        weld_csv = OUT_DIR / "weld_points.csv"
        weld_df.to_csv(weld_csv, index=False, encoding="utf-8-sig")
        print(f"✅ Saved weld CSV: {weld_csv.resolve()}")

        weld_id_to_xyz = {
            str(r["name"]).strip(): np.array([r["x"], r["y"], r["z"]], dtype=float)
            for _, r in weld_df.iterrows()
        }
        weld_points = weld_df[["x", "y", "z"]].to_numpy(dtype=float)

        mesh_body = load_glb_as_single_mesh(GLB_PATH, chunk=MERGE_CHUNK)
        print("\n[PyVista] Converting mesh.")
        mesh_pv = trimesh_to_pyvista(mesh_body)
        print(f"✓ PyVista mesh: points={mesh_pv.n_points}, cells={mesh_pv.n_cells}")

        weld_spheres, weld_radius = build_weld_spheres(weld_points, mesh_body)

        _ASSETS.update({
            "loaded": True,
            "mesh_pv": mesh_pv,
            "weld_id_to_xyz": weld_id_to_xyz,
            "weld_spheres": weld_spheres,
            "weld_radius": float(weld_radius),
        })


def _render_final_png(mesh_pv, weld_spheres, bad_xyz, weld_radius, weld_id: str, badge_text: Optional[str], view_vec, view_up) -> bytes:
    if Image is None:
        raise RuntimeError("Pillow required: pip install pillow")

    import pyvista as pv
    ensure_headless_pyvista()

    p = pv.Plotter(off_screen=True, window_size=VIEW_IMG_SIZE)
    p.set_background("white")
    try:
        p.enable_anti_aliasing()
    except Exception:
        pass

    # 模型
    p.add_mesh(mesh_pv, color=BODY_COLOR, smooth_shading=True, show_edges=False, opacity=1.0)

    # 正常焊点：蓝色
    p.add_mesh(weld_spheres, color=WELD_NORMAL_COLOR, smooth_shading=True)

    # 异常焊点：红色 + 更大
    bad_sphere = build_single_sphere(bad_xyz, weld_radius * float(HIGHLIGHT_SCALE))
    p.add_mesh(bad_sphere, color=WELD_BAD_COLOR, smooth_shading=True, show_edges=False)

    p.reset_camera()
    dist = float(getattr(p.camera, "distance", 1.0))

    focal = np.asarray(bad_xyz, dtype=float)
    v = _normalize(view_vec)
    up = _normalize(view_up)

    p.camera.focal_point = tuple(focal)
    p.camera.position = tuple(focal - v * dist)
    p.camera.up = tuple(up)

    try:
        p.camera.zoom(VIEW_ZOOM)
    except Exception:
        pass

    try:
        p.render()
    except Exception:
        pass

    model_bottom = _mesh_bottom_in_pil(p, mesh_pv)
    disp = _world_to_display_xyz(p, bad_xyz)

    img = p.screenshot(return_img=True)
    p.close()

    if img is None:
        raise RuntimeError("PyVista screenshot returned None.")
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    pil = Image.fromarray(img.astype(np.uint8)).convert("RGBA")

    # 有 badge 才画标注（否则只高亮焊点）
    if disp is not None and badge_text:
        dx, dy, _ = disp
        ww, hh = pil.size
        px = float(dx)
        py = float(hh - dy)
        _draw_callout_id_and_badge(
            pil, px, py,
            weld_id=weld_id,
            badge_text=badge_text,
            min_elbow_y=model_bottom,
        )

    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def get_views_base64(weld_id: str, value: Optional[str]) -> Dict:
    load_assets_once()

    weld_id = str(weld_id).strip()
    weld_map: Dict[str, np.ndarray] = _ASSETS["weld_id_to_xyz"]  # type: ignore
    if weld_id not in weld_map:
        sample = list(weld_map.keys())[:10]
        raise KeyError(f"Unknown weld_id: {weld_id}. Sample IDs: {sample}")

    badge = _format_badge_value(value)
    badge_text = badge if badge != "" else None

    bad_xyz = weld_map[weld_id]
    mesh_pv = _ASSETS["mesh_pv"]
    weld_spheres = _ASSETS["weld_spheres"]
    weld_radius = float(_ASSETS["weld_radius"])

    # 选最佳视角（只在 6 个面里）
    if weld_id in _BEST_VIEW_CACHE:
        best_name = _BEST_VIEW_CACHE[weld_id]
        best_vec = _normalize(CANDIDATE_VIEWS[best_name])
        best_up = _choose_viewup(best_vec)
    else:
        best_name, best_vec, best_up = _select_best_view(mesh_pv, bad_xyz, weld_radius * float(HIGHLIGHT_SCALE))
        _BEST_VIEW_CACHE[weld_id] = best_name

    # 缓存最终图
    with _RENDER_LOCK:
        cached = _get_png_cache(weld_id, badge, best_name)
        if cached is None:
            png = _render_final_png(
                mesh_pv, weld_spheres,
                bad_xyz, weld_radius,
                weld_id=weld_id,
                badge_text=badge_text,
                view_vec=best_vec,
                view_up=best_up,
            )
            _set_png_cache(weld_id, badge, best_name, png)
        else:
            png = cached

    # ✅ 仍旧返回 base64（前端直接 img.src=data_url）
    return {
        "status": "success",
        "data": {
            "weld_id": weld_id,
            "value": badge,
            "best_view": best_name,   # 这里会是 front/back/left/right/top/bottom
            "views": {
                "front": {"data_url": _png_to_data_url(png)},  # 前端继续读 front 就行
            },
        },
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


# =============================================================================
# FastAPI
# =============================================================================
app = FastAPI(title="WeldPoint Views Base64 Service", version="7.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/mobo/img/weldpoint_views_base64")
async def weldpoint_views_base64(
    weld_id: str = Query(..., description="异常焊点ID（有问题的焊点）"),
    value: Optional[str] = Query(None, description="封样编号（独立圆形徽标）"),
):
    try:
        return get_views_base64(weld_id=weld_id, value=value)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010, reload=False)
