"""
Fixed CATIA rendering - filters out point clouds from GLB
"""
from pathlib import Path
import trimesh
import pandas as pd
import numpy as np
import pyvista as pv
import re

# --- Configuration ---
step_path = Path(r"C:\Users\abrik\Desktop\Work\Ballsnow\Code\Xiaomi\T16-Welding Spot.stp")
glb_path  = Path("exports_visualization") / "T16-Welding Spot.glb"
out_dir = Path("exports_pv")
out_dir.mkdir(exist_ok=True, parents=True)

# --- 1) Load weld points ---
def extract_rsw_points(step_file: Path, encoding: str = "latin-1") -> pd.DataFrame:
    with open(step_file, "r", encoding=encoding, errors="ignore") as f:
        text = f.read()
    pattern = re.compile(
        r"#\d+\s*=\s*CARTESIAN_POINT\('([^']*RSW[^']*)',\s*\(([^)]*)\)\s*\)\s*;",
        re.DOTALL,
    )
    rows = []
    for name, coords_str in pattern.findall(text):
        parts = [p for p in re.split(r"[,\s]+", coords_str.strip()) if p]
        try:
            coords = [float(p) for p in parts]
        except ValueError:
            continue
        if len(coords) != 3:
            continue
        rows.append({"name": name, "x": coords[0], "y": coords[1], "z": coords[2]})
    return pd.DataFrame(rows)

print("Loading weld points...")
weld_df = extract_rsw_points(step_path)
weld_points = weld_df[['x', 'y', 'z']].values
print(f"✓ Loaded {len(weld_df)} weld points")

# --- 2) Load mesh and FILTER OUT POINT CLOUDS ---
print(f"\nLoading GLB file: {glb_path}")

obj = trimesh.load(glb_path)

if isinstance(obj, trimesh.Scene):
    print(f"  Scene with {len(obj.geometry)} geometries")
    
    # ⭐ CRITICAL FIX: Filter out point clouds, keep only meshes
    mesh_geometries = []
    point_clouds = []
    
    for name, geom in obj.geometry.items():
        if isinstance(geom, trimesh.PointCloud):
            point_clouds.append(name)
        elif isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 0:
            mesh_geometries.append(geom)
            print(f"    ✓ Mesh: {name} - {len(geom.vertices)} verts, {len(geom.faces)} faces")
        else:
            print(f"    ⚠️  Skipping {name}: {type(geom).__name__}")
    
    print(f"\n  Found {len(mesh_geometries)} meshes, {len(point_clouds)} point clouds")
    print(f"  Filtered out {len(point_clouds)} point clouds")
    
    if len(mesh_geometries) == 0:
        raise ValueError("No mesh geometries found! GLB only contains point clouds.")
    
    # Concatenate only the mesh geometries
    mesh_body = trimesh.util.concatenate(mesh_geometries)
    
else:
    mesh_body = obj

print(f"\n✓ Final mesh: {mesh_body.vertices.shape[0]} vertices, {mesh_body.faces.shape[0]} faces")
print(f"  Bounds: {mesh_body.bounds}")

# Validate mesh
if mesh_body.vertices.shape[0] == 0 or mesh_body.faces.shape[0] == 0:
    raise ValueError("Mesh is empty after loading!")

# --- 3) Clean up mesh ---
print("\nCleaning mesh...")
original_faces = len(mesh_body.faces)
mesh_body.remove_degenerate_faces()
mesh_body.merge_vertices()
mesh_body.fix_normals()

if len(mesh_body.faces) < original_faces:
    print(f"  Removed {original_faces - len(mesh_body.faces)} degenerate faces")

# --- 4) Convert to PyVista ---
print("\nConverting to PyVista...")

body_vertices = np.asarray(mesh_body.vertices, dtype=float)
body_faces = np.asarray(mesh_body.faces, dtype=int)

n_faces = body_faces.shape[0]
faces_pv = np.hstack(
    (np.full((n_faces, 1), 3, dtype=np.int64), body_faces.astype(np.int64))
).ravel()

mesh_pv = pv.PolyData(body_vertices, faces_pv)
print(f"✓ PyVista mesh: {mesh_pv.n_points} points, {mesh_pv.n_cells} cells")

# --- 5) Compute normals (CRITICAL!) ---
print("  Computing normals...")
mesh_pv = mesh_pv.compute_normals(
    cell_normals=True,
    point_normals=True,
    split_vertices=False,
    flip_normals=False,
    consistent_normals=True,
    auto_orient_normals=True
)

if hasattr(mesh_pv, 'point_normals') and mesh_pv.point_normals is not None:
    print(f"  ✓ Normals: {mesh_pv.point_normals.shape}")
else:
    print("  ⚠️  Warning: Normals may not have computed correctly")

# --- 6) Smooth mesh ---
print("  Smoothing mesh...")
try:
    mesh_pv_smooth = mesh_pv.smooth(
        n_iter=15,
        relaxation_factor=0.1,
        feature_smoothing=True,
        boundary_smoothing=True,
        edge_angle=20,
        feature_angle=50
    )
    print(f"  ✓ Smoothed: {mesh_pv_smooth.n_points} points, {mesh_pv_smooth.n_cells} cells")
except Exception as e:
    print(f"  ⚠️  Smoothing failed: {e}")
    mesh_pv_smooth = mesh_pv

# --- 7) Rendering function ---
def render_catia_style(mesh, weld_pts, view_fn, filename):
    """Professional CATIA-style rendering"""
    p = pv.Plotter(off_screen=True, window_size=(1920, 1080))
    
    # Background
    p.set_background('white', top=(0.85, 0.90, 0.95))
    
    # Anti-aliasing
    try:
        p.enable_anti_aliasing('msaa')
    except:
        pass
    
    # Add mesh with full opacity
    print(f"  Rendering mesh: {mesh.n_cells} cells")
    p.add_mesh(
        mesh,
        color=(0.7, 0.75, 0.8),
        smooth_shading=True,
        show_edges=True,
        edge_color=(0.2, 0.2, 0.25),
        line_width=0.5,
        lighting=True,
        opacity=1.0,  # Fully opaque
        metallic=0.5,
        roughness=0.3,
        ambient=0.3,
        diffuse=0.7,
        specular=0.8,
        specular_power=30,
        pbr=True,
    )
    
    # Add lighting
    light1 = pv.Light(position=(1, 1, 1), light_type='scene light')
    light1.intensity = 0.8
    p.add_light(light1)
    
    light2 = pv.Light(position=(-1, -1, 1), light_type='scene light')
    light2.intensity = 0.3
    p.add_light(light2)
    
    light3 = pv.Light(position=(0, 0, -1), light_type='scene light')
    light3.intensity = 0.2
    p.add_light(light3)
    
    # Enable shadows
    try:
        p.enable_shadows()
    except:
        pass
    
    # Add weld points
    if weld_pts is not None and len(weld_pts) > 0:
        print(f"  Adding {len(weld_pts)} weld points")
        p.add_points(
            weld_pts,
            color='crimson',
            render_points_as_spheres=True,
            point_size=12,
        )
    
    # Set view
    view_fn(p)
    p.reset_camera()
    p.camera.zoom(1.2)
    
    # Print camera info for debugging
    print(f"  Camera position: {p.camera.position}")
    print(f"  Camera focal point: {p.camera.focal_point}")
    
    # Render
    filepath = out_dir / filename
    p.show(screenshot=str(filepath))
    p.close()
    
    print(f"✓ Saved: {filepath}\n")

# --- 8) Define views ---
def view_front(p):
    p.view_yz()

def view_rear(p):
    p.view_yz()
    p.camera.azimuth = p.camera.azimuth + 180

def view_top(p):
    p.view_xy()

def view_iso(p):
    p.view_isometric()

# --- 9) Generate renders ---
print("\n" + "="*80)
print("GENERATING CATIA-STYLE RENDERS")
print("="*80 + "\n")

views = [
    (view_front, "catia_front_fixed.png"),
    (view_rear, "catia_rear_fixed.png"),
    (view_top, "catia_top_fixed.png"),
    (view_iso, "catia_isometric_fixed.png"),
]

for view_fn, filename in views:
    print(f"Rendering {filename}...")
    try:
        render_catia_style(mesh_pv_smooth, weld_points, view_fn, filename)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

print("="*80)
print("RENDERING COMPLETE")
print("="*80)
print(f"\nGenerated files in: {out_dir.absolute()}")
print(f"Total mesh faces rendered: {mesh_pv_smooth.n_cells}")
