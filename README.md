# MOBO 焊点可视化（FastAPI + Base64 前端展示）

## 1. 项目做什么
- **后端（FastAPI）**：加载整车/零件的 3D 网格（GLB）与焊点坐标（STEP 中提取 RSW 点），输入一个“异常焊点 ID”，在 **6 个标准视角（前/后/左/右/上/下）**里自动选取一个更适合展示焊点的面进行渲染：  
  - **正常焊点：蓝色**  
  - **异常焊点：红色 + 更大**  
  - 可选绘制：**红色引线 + 焊点ID标注框 + 封样编号（value）独立圆形徽标**
- **前端（HTML）**：输入焊点ID/封样编号，请求后端接口拿到 **base64 data_url**，直接显示在 `<img>` 中。

> 前端固定读取 `data.data.views.front.data_url`。即使后端实际选出的最佳视角不是“正面”，也会把最终渲染图放在 `front` 字段里，前端不需要改动。

---

## 2. 目录与文件说明

### ✅ 运行必须
- `http_code.py`：最终后端服务（FastAPI），提供接口：
  - `GET /mobo/img/weldpoint_views_base64?weld_id=...&value=...`
  - 返回 JSON，包含 base64 的 `data_url` 与后端选到的最佳视角 `best_view`（front/back/left/right/top/bottom）
- `show.html`：最终前端页面，`fetch()` 调接口，取 `data.data.views.front.data_url` 显示图片
- `requirements.txt`：后端依赖
- `Dockerfile` / `.dockerignore`：容器部署
- `config.yml`：配置模板（目前代码默认路径写在 `http_code.py` 顶部；如需读配置，可按此结构扩展）

### 🧪 调试/历史脚本（一般不需要跑）
- `fixed_catia_render.py`：离线渲染/出图脚本（非 HTTP 服务）
- `another_2_visualization_code.py`：离线可视化/导出逻辑（非 HTTP 服务）
- `xiaomi.py`：Streamlit/Plotly 的 mesh viewer（非 FastAPI）

---

## 3. 依赖与环境

### 3.1 Python 依赖安装
```bash
pip install -r requirements.txt
uvicorn http_code:app --host 0.0.0.0 --port 8010
