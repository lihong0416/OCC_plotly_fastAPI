# MOBO 焊点可视化（FastAPI + Base64 前端展示）

## 1. 项目做什么
- 后端：加载整车/零件的 3D 网格 + 焊点坐标，输入一个“异常焊点ID”，自动在 **6 个标准视角（前/后/左/右/上/下）**里选一个更合适的角度渲染，并把异常焊点用 **红色**高亮；其他焊点为 **蓝色**；可选绘制红色引线与标注框，并把“封样编号”以**独立圆形徽标**形式放在标注旁边。
- 前端：输入焊点ID/封样编号，请求后端接口拿到 **base64 data_url**，直接显示到 `<img>`。

> 前端固定读取：`data.data.views.front.data_url`，即使后端选出来的最佳视角不是“正面”，也会把最终渲染图放在 `front` 字段里，前端无需改动。:contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}

---

## 2. 文件说明
### ✅ 需要用来跑的
- `http_code.py`：**最终后端服务**（FastAPI）。提供接口：
  - `GET /mobo/img/weldpoint_views_base64?weld_id=...&value=...`
  - 返回 JSON，里面有 base64 的 `data_url`，以及后端选到的最佳视角 `best_view`（front/back/left/right/top/bottom）。:contentReference[oaicite:2]{index=2}
- `show.html`：**最终前端页面**。通过 `fetch()` 调接口，拿到 `data.data.views.front.data_url` 后塞给 `<img>` 展示。:contentReference[oaicite:3]{index=3}

### 用于调试
- `show.html`：调试页，会把接口返回 JSON 打印出来，也能同时展示两张图（更适合联调排错）。

---

## 3. 环境依赖
后端是 Python（FastAPI + 渲染库）。你可以按你现有环境补齐常用依赖：
- `fastapi`, `uvicorn`
- `numpy`, `pandas`
- `pyvista`, `vtk`, `trimesh`
- （可能还会用到）`Pillow`

> 说明：渲染通常是 **离屏渲染**（服务器没显示器也能跑）。如果你机器上 VTK/渲染报错，最常见的做法是用 `xvfb-run` 包一层启动。

---

## 4. 一键跑起来

### 4.1 启动后端（必须先启动）
进入代码目录后执行：

```bash
pip install requirements.txt
uvicorn http_code:app --host 0.0.0.0 --port 8010
