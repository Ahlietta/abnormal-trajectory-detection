from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from trajectory_detector import TrajectoryAnomalyDetector

# 服务启动时一次性加载检测器，避免重复加载模型
detector = TrajectoryAnomalyDetector(model_dir='../data/processed/')
app = FastAPI(title="ATC商场轨迹异常检测API", version="1.1")

# ========== 请求体定义 ==========
class SingleTrajectoryRequest(BaseModel):
    x_coords: List[float]
    y_coords: List[float]
    facing_angles: Optional[List[float]] = None
    custom_thresholds: Optional[dict] = None

class BatchTrajectoryRequest(BaseModel):
    trajectories: List[dict]  # 每项包含 x_coords, y_coords，可选 facing_angles

# ========== 接口实现 ==========
# 1. 健康检查
@app.get("/health", summary="服务健康检查")
def health_check():
    return {"status": "ok", "service": "ATC-trajectory-anomaly-detector"}

# 2. 单条轨迹检测
@app.post("/detect", summary="单条轨迹异常检测")
def detect_single(req: SingleTrajectoryRequest):
    result = detector.detect(
        req.x_coords,
        req.y_coords,
        req.facing_angles,
        req.custom_thresholds
    )
    
    # 错误处理：参数/检测失败时返回400错误
    if result['code'] != 0:
        raise HTTPException(status_code=400, detail=result['msg'])
    
    return result

# 3. 批量轨迹检测
@app.post("/batch_detect", summary="批量轨迹异常检测")
def detect_batch(req: BatchTrajectoryRequest):
    traj_list = []
    for item in req.trajectories:
        x = item['x_coords']
        y = item['y_coords']
        facing = item.get('facing_angles', None)
        traj_list.append((x, y, facing))
    
    # 直接返回字典列表，省去DataFrame转换
    result = detector.batch_detect(traj_list, return_type='list')
    return result
