import io
import math
from typing import Tuple, List, Optional

import streamlit as st
import trimesh
import numpy as np
import plotly.graph_objects as go  # 用于基础网格可视化
import os
import tempfile


def convert_step_to_stl(step_bytes: bytes) -> Optional[trimesh.Trimesh]:
    """
    Converts STEP file bytes to a trimesh.Trimesh object using python-occ,
    applying adaptive meshing quality based on the input file size.

    使用 python-occ 将 STEP 文件字节转换为 trimesh.Trimesh 对象，
    并根据输入文件大小应用自适应网格化质量。
    """
    # 导入 python-occ 核心模块
    # 如果 'python-occ-core' 未安装, 此处将引发 ImportError
    try:
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.StlAPI import StlAPI_Writer
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    except ImportError:
        st.error("STEP file processing requires the 'python-occ-core' library. Please install it.")
        return None

    # 1. 定义自适应参数
    FILE_SIZE_LIMIT_BYTES = 20 * 1024 * 1024  # 20 MB 阈值

    # 默认/高精度参数
    HIGH_QUALITY_LINEAR_DEFLECTION = 0.01
    HIGH_QUALITY_ANGULAR_DEFLECTION = 0.5

    # 优化/低精度参数
    LOW_QUALITY_LINEAR_DEFLECTION = 500.0
    LOW_QUALITY_ANGULAR_DEFLECTION = 20.0

    # 2. 根据文件大小选择参数
    if len(step_bytes) > FILE_SIZE_LIMIT_BYTES:
        linear_deflection = LOW_QUALITY_LINEAR_DEFLECTION
        angular_deflection = LOW_QUALITY_ANGULAR_DEFLECTION
    else:
        linear_deflection = HIGH_QUALITY_LINEAR_DEFLECTION
        angular_deflection = HIGH_QUALITY_ANGULAR_DEFLECTION

    step_path = None
    stl_path = None
    mesh = None

    try:
        # 3. 写入临时 STEP 文件
        with tempfile.NamedTemporaryFile(suffix='.step', delete=False) as tmp_step:
            tmp_step.write(step_bytes)
            tmp_step.flush()
            step_path = tmp_step.name

        # 4. 读取 STEP 文件
        reader = STEPControl_Reader()
        status = reader.ReadFile(step_path)

        if status != IFSelect_RetDone:
            raise IOError("Failed to read STEP file with python-occ (ReadFile status not OK).")

        reader.TransferRoots()
        shape = reader.OneShape()

        # 5. 创建网格化工具并设置自适应参数
        mesh_algo = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection)
        mesh_algo.Perform()

        if not mesh_algo.IsDone():
            raise RuntimeError("Meshing failed or was not completed.")

        # 6. 写入 STL 文件
        stl_path = step_path.replace('.step', '.stl')
        stl_writer = StlAPI_Writer()
        wrote = stl_writer.Write(shape, stl_path)

        if not wrote:
            raise IOError("Failed to write STL from STEP shape.")

        # 7. 使用 trimesh 加载 STL
        mesh = trimesh.load_mesh(stl_path, force='mesh')

    finally:
        # 8. 清理临时文件
        if step_path and os.path.exists(step_path):
            os.unlink(step_path)
        if stl_path and os.path.exists(stl_path):
            os.unlink(stl_path)

    return mesh


# --- 2. File loading and mesh processing ---
def load_and_process_mesh(uploaded_file) -> Optional[trimesh.Trimesh]:
    """加载并处理上传的网格文件 (STL/STEP)。"""
    if uploaded_file is None:
        return None

    name = getattr(uploaded_file, "name", "uploaded")
    ext = name.lower().split('.')[-1]

    mesh = None

    if ext in ('stl',):
        try:
            with st.spinner("Loading STL..."):
                file_bytes = uploaded_file.read()
                mesh = trimesh.load_mesh(io.BytesIO(file_bytes), file_type='stl', force='mesh')
        except Exception as e:
            st.error(f"Failed to load STL: {e}")
            return None

    elif ext in ('step', 'stp'):
        try:
            with st.spinner("Converting STEP to mesh..."):
                step_bytes = uploaded_file.read()
                mesh = convert_step_to_stl(step_bytes)
                if mesh is None:
                    st.error("Failed to convert STEP file.")
                    return None
        except Exception as e:
            st.error(f"Error processing STEP file: {e}")
            return None

    else:
        st.error("Unsupported file type. Please upload an STL or STEP (.step/.stp) file.")
        return None

    if not isinstance(mesh, trimesh.Trimesh):
        st.error("Loaded object is not a mesh.")
        return None

    with st.spinner("Processing mesh..."):
        try:
            mesh.remove_duplicate_faces()

            if not mesh.is_watertight:
                mesh.fill_holes()

            try:
                mesh.fix_normals()
            except Exception:
                # 这是一个有意义的回退
                mesh.rezero()

            mesh.vertices = mesh.vertices.astype(np.float64)
        except Exception as e:
            st.warning(f"Mesh cleaning had issues: {e}")

    return mesh


# --- 3. Orientation calculation utility (Kept for potential future use or completeness, though not used in current main logic) ---
def calculate_orientation_zyx(normal: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    """
    Calculate ZYX Euler angles from a given normal vector (Z_tool = -normal)
    and a tangent vector (Y_tool). The resulting coordinate system is right-handed.

    从给定的法向量 (Z_tool = -normal) 和切向量 (Y_tool) 计算 ZYX 欧拉角。
    生成的坐标系是右手坐标系。
    """
    normal = normal / (np.linalg.norm(normal) + 1e-9)
    tangent = tangent / (np.linalg.norm(tangent) + 1e-9)

    Z_tool = -normal

    Y_tool = tangent - np.dot(tangent, Z_tool) * Z_tool
    Y_norm = np.linalg.norm(Y_tool)

    # 如果切向量与法向量平行（万向锁情况）
    if Y_norm < 1e-6:
        # 选择一个与Z_tool垂直的默认Y方向
        if abs(Z_tool[2]) < 0.9:
            Y_tool = np.array([0.0, 0.0, 1.0])
        else:
            Y_tool = np.array([1.0, 0.0, 0.0])
        Y_tool = Y_tool - np.dot(Y_tool, Z_tool) * Z_tool
        Y_tool = Y_tool / (np.linalg.norm(Y_tool) + 1e-9)
    else:
        Y_tool = Y_tool / Y_norm

    # X轴：由Y×Z确定，构成右手坐标系
    X_tool = np.cross(Y_tool, Z_tool)

    # 构建旋转矩阵 R = [X_tool | Y_tool | Z_tool]
    R = np.vstack([X_tool, Y_tool, Z_tool]).T

    # 从旋转矩阵提取ZYX欧拉角
    R11, R12, R13 = R[0, :]
    R21, R22, R23 = R[1, :]
    R31, R32, R33 = R[2, :]

    R_proj = np.sqrt(R11 ** 2 + R21 ** 2)

    if R_proj < 1e-6:
        # 万向锁情况
        ry_radians = np.pi / 2.0 if R31 < 0 else -np.pi / 2.0
        rx_radians = 0.0
        rz_radians = np.arctan2(R22, R12) if R31 < 0 else np.arctan2(-R22, -R12)
    else:
        # 正常情况
        ry_radians = np.arctan2(-R31, R_proj)
        rz_radians = np.arctan2(R21, R11)
        rx_radians = np.arctan2(R32, R33)

    # 转换为角度
    return np.array([np.degrees(rx_radians), np.degrees(ry_radians), np.degrees(rz_radians)])


# --- 4. Mesh visualization utility ---
def visualize_mesh_plotly(mesh: trimesh.Trimesh):
    """使用 Plotly 渲染 trimesh 对象，专注于模型本身的展示。"""
    if mesh is None:
        return

    verts = mesh.vertices
    faces = mesh.faces

    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

    # 使用 Plotly Mesh3d 绘制网格
    mesh_trace = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color='#3182CE',  # 简单的蓝色
        name="Loaded Mesh",
        opacity=0.7,
        flatshading=True,
        showscale=False
    )

    fig = go.Figure(data=[mesh_trace])

    # 优化布局，使其居中并保持数据比例
    fig.update_layout(
        scene=dict(
            aspectmode='data',  # 确保X, Y, Z轴使用相同的比例
            xaxis=dict(showbackground=True, zerolinecolor="gray", title="X"),
            yaxis=dict(showbackground=True, zerolinecolor="gray", title="Y"),
            zaxis=dict(showbackground=True, zerolinecolor="gray", title="Z"),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )

    st.subheader("模型几何可视化 (3D)")
    st.plotly_chart(fig, use_container_width=True)


# --- 5. Streamlit main application flow (Cleaned) ---
def main():
    st.set_page_config(page_title="Vehicle Geometry Processor", layout="wide")
    st.title("Vehicle Geometry Processor (Mesh Utility)")

    st.sidebar.header("输入")

    uploaded_file = st.sidebar.file_uploader("上传网格文件 (.stl, .step, .stp)", type=['stl', 'step', 'stp'])

    mesh = None
    if uploaded_file is not None:
        mesh = load_and_process_mesh(uploaded_file)

        if mesh is not None:
            st.sidebar.markdown("---")
            st.sidebar.subheader("模型信息")

            extents = mesh.extents
            max_extent = float(np.max(extents))

            st.sidebar.write(f"顶点数量: {len(mesh.vertices)}")
            st.sidebar.write(f"面片数量: {len(mesh.faces)}")
            st.sidebar.write(f"模型范围(mm): {np.round(extents, 6)}")
            st.sidebar.write(f"最大范围(mm): {np.round(max_extent, 6)}")

            st.success("网格加载与处理成功。")

            # 仅显示加载成功的网格信息
            st.header("处理后的模型几何信息")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("顶点数", len(mesh.vertices))
            with col2:
                st.metric("面片数", len(mesh.faces))

            st.subheader("模型边界 Extents (X, Y, Z)")
            st.json({
                "Min": mesh.bounds[0].tolist(),
                "Max": mesh.bounds[1].tolist(),
                "Extents": mesh.extents.tolist(),
                "Center": mesh.centroid.tolist()
            })

            # 调用网格可视化函数
            visualize_mesh_plotly(mesh)

            st.info("当前应用专注于模型几何处理和可视化。所有轨迹生成功能已移除。")

        else:
            st.error("上传的文件无法处理成网格。")
    else:
        st.info("请上传 .stl 或 .step 文件。")

    st.markdown("---")


if __name__ == "__main__":
    main()