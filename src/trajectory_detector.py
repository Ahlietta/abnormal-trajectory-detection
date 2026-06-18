import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

# ===================== 基础工具函数（与训练逻辑完全对齐） =====================
def angle_diff(angles):
    """计算角度差分，处理0-2π周期性"""
    diff = np.diff(angles)
    return np.arctan2(np.sin(diff), np.cos(diff))

def sliding_window_smooth(data, window=5):
    """滑动窗口均值平滑"""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='same')

def angle_smooth(angles, window=5):
    """角度专用平滑：向量平均法，避免周期跳变"""
    if len(angles) < window:
        return angles
    sin_smooth = sliding_window_smooth(np.sin(angles), window)
    cos_smooth = sliding_window_smooth(np.cos(angles), window)
    return np.arctan2(sin_smooth, cos_smooth)


# ===================== 核心检测器类 =====================
class TrajectoryAnomalyDetector:
    """
    ATC商场行人轨迹异常检测器
    评分规则：基于单特征AUC + 相关性去冗余的数据驱动权重，完全贴合ATC场景
    支持功能：单条检测、批量检测、轨迹可视化、实时滑动窗口检测
    适配：桌面GUI、Web接口、脚本调用等多入口
    """
    def __init__(self, model_dir='../data/processed/', dt=0.1):
        """
        初始化检测器
        :param model_dir: 模型与配置文件所在目录
        :param dt: 轨迹采样间隔（秒），默认10Hz=0.1s，与ATC数据集一致
        """
        # 加载核心模型与配置
        self.dbscan_model = joblib.load(os.path.join(model_dir, 'best_dbscan_model.pkl'))
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler_final.pkl'))
        self.normal_stats = joblib.load(os.path.join(model_dir, 'normal_feature_stats.pkl'))
        
        # 禁区配置
        forbidden = np.load(os.path.join(model_dir, 'forbidden_zone.npy'))
        self.forbid_x = forbidden[0]
        self.forbid_y = forbidden[1]
        self.forbid_radius = 5000  # 禁区半径，单位毫米，与训练时一致
        
        # 提取DBSCAN核心样本，用于计算整体异常度
        self.core_samples = self.dbscan_model.components_
        self.dt = dt
        
        # 默认报警阈值（全局默认值，仅作兜底，单请求修改不影响全局）
        self.default_thresholds = {
            'running': 0.6,
            'loitering': 0.6,
            'trespassing': 0.5
        }

        # ========== 数据驱动的三类异常特征权重配置 ==========
        # 格式：(特征索引, 方向(1=越大越异常/-1=越小越异常), 优化后权重)
        # 奔跑异常：6个有效特征，无高冗余降权，速度/加速度维度均衡贡献
        self.running_feat_config = [
            (17, 1, 0.1678),  # std_acceleration 加速度标准差
            (3, 1, 0.1677),   # speed_cv 速度变异系数
            (2, 1, 0.1672),   # std_speed 速度标准差
            (16, 1, 0.1672),  # max_acceleration 最大加速度
            (1, 1, 0.1671),   # max_speed 最大速度
            (0, 1, 0.1630),   # avg_speed 平均速度
        ]

        # 徘徊异常：12个有效特征，含2个冗余降权特征，多维度综合判定
        self.loitering_feat_config = [
            (0, -1, 0.1044),  # avg_speed 平均速度（速度越小越倾向徘徊）
            (12, 1, 0.1040),  # low_speed_ratio 低速占比
            (3, 1, 0.1037),   # speed_cv 速度变异系数
            (5, 1, 0.1035),   # max_dir_change 最大方向变化
            (16, 1, 0.1030),  # max_acceleration 最大加速度
            (6, 1, 0.1002),   # std_dir_change 方向变化标准差
            (11, 1, 0.0906),  # tortuosity 轨迹迂回度
            (4, 1, 0.0254),   # avg_dir_change 平均方向变化（冗余降权）
            (9, 1, 0.0824),   # path_length 轨迹总长度
            (2, 1, 0.0814),   # std_speed 速度标准差
            (13, 1, 0.0238),  # duration 轨迹时长（冗余降权）
            (19, 1, 0.0775),  # angle_entropy 方向熵
        ]

        # 闯入禁区：4个有效特征，核心权重集中在位置与禁区占比
        self.trespassing_feat_config = [
            (14, -1, 0.3857), # center_x 轨迹中心x坐标
            (15, 1, 0.1157),  # center_y 轨迹中心y坐标（冗余降权）
            (21, -1, 0.1157), # min_dist_to_forbidden 到禁区最小距离（冗余降权）
            (22, 1, 0.3830),  # in_forbidden_ratio 禁区内停留占比
        ]

    def get_forbidden_zone(self):
        """统一获取禁区配置，供上层GUI/可视化调用"""
        return {
            'center_x': self.forbid_x,
            'center_y': self.forbid_y,
            'radius': self.forbid_radius
        }

    def _normalize_feature(self, feat_val, feat_idx, direction):
        """
        基于正常样本分位数的鲁棒归一化
        输出0~1分值，统一对齐为「值越大，异常程度越高」
        """
        p1 = self.normal_stats['p1'][feat_idx]
        p99 = self.normal_stats['p99'][feat_idx]
        # 截断到正常样本1%~99%分位数范围，避免极端值放大影响
        norm_val = np.clip((feat_val - p1) / (p99 - p1 + 1e-8), 0, 1)
        # 负向特征取反，统一方向
        if direction == -1:
            norm_val = 1 - norm_val
        return norm_val

    def _preprocess_trajectory(self, x_coords, y_coords, facing_angles=None):
        """轨迹预处理：平滑、计算速度与角度，与训练集逻辑完全一致"""
        x = np.asarray(x_coords, dtype=float)
        y = np.asarray(y_coords, dtype=float)
        n = len(x)

        # 计算速度
        dx = np.diff(x)
        dy = np.diff(y)
        velocity = np.sqrt(dx**2 + dy**2) / self.dt
        velocity = np.insert(velocity, 0, velocity[0])  # 首位补值对齐长度

        # 计算运动方向角
        motion_angle = np.arctan2(dy, dx)
        motion_angle = np.insert(motion_angle, 0, motion_angle[0])

        # 朝向角处理：无输入时默认朝向与运动方向一致
        if facing_angles is None:
            facing_angle = motion_angle.copy()
        else:
            facing_angle = np.asarray(facing_angles, dtype=float)

        # 滑动窗口平滑
        x_smooth = sliding_window_smooth(x, 5)
        y_smooth = sliding_window_smooth(y, 5)
        v_smooth = sliding_window_smooth(velocity, 5)
        motion_smooth = angle_smooth(motion_angle, 5)
        facing_smooth = angle_smooth(facing_angle, 5)

        return pd.DataFrame({
            'pos_x_smooth': x_smooth,
            'pos_y_smooth': y_smooth,
            'velocity_smooth': v_smooth,
            'motion_angle_smooth': motion_smooth,
            'facing_angle_smooth': facing_smooth
        })

    def _extract_features(self, traj_df):
        """提取完整23维特征，与训练时特征提取逻辑100%对齐"""
        v = traj_df['velocity_smooth'].values
        motion_ang = traj_df['motion_angle_smooth'].values
        facing_ang = traj_df['facing_angle_smooth'].values
        x = traj_df['pos_x_smooth'].values
        y = traj_df['pos_y_smooth'].values
        n = len(traj_df)

        # 速度类特征
        avg_speed = np.mean(v)
        max_speed = np.max(v)
        std_speed = np.std(v)
        speed_cv = std_speed / (avg_speed + 1e-6)

        # 方向变化类特征
        motion_diffs = angle_diff(motion_ang)
        avg_dir_change = np.mean(np.abs(motion_diffs))
        max_dir_change = np.max(np.abs(motion_diffs))
        std_dir_change = np.std(motion_diffs)

        # 朝向偏差类特征
        facing_motion_diff = np.arctan2(
            np.sin(facing_ang - motion_ang), 
            np.cos(facing_ang - motion_ang)
        )
        avg_facing_deviation = np.mean(np.abs(facing_motion_diff))
        max_facing_deviation = np.max(np.abs(facing_motion_diff))

        # 轨迹形态类特征
        path_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
        start_end_dist = np.sqrt((x[-1]-x[0])**2 + (y[-1]-y[0])**2)
        tortuosity = path_length / (start_end_dist + 1e-6)

        # 停留与时长特征
        low_speed_ratio = np.sum(v < 300) / n
        duration = n

        # 空间位置特征
        center_x = np.mean(x)
        center_y = np.mean(y)

        # 加速度类特征
        acceleration = np.diff(v)
        max_acceleration = np.max(np.abs(acceleration))
        std_acceleration = np.std(acceleration)

        # 转向模式特征
        turn_threshold = 0.1
        valid_turns = np.sum(np.abs(motion_diffs) > turn_threshold)
        turn_frequency = valid_turns / n

        n_bins = 8
        bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        angle_counts, _ = np.histogram(motion_ang, bins=bins)
        angle_probs = angle_counts / np.sum(angle_counts)
        angle_probs = angle_probs[angle_probs > 0]
        angle_entropy = -np.sum(angle_probs * np.log2(angle_probs))

        turn_signs = np.sign(motion_diffs)
        turn_signs = turn_signs[turn_signs != 0]
        if len(turn_signs) < 2:
            direction_reversal_rate = 0.0
        else:
            reversal_count = np.sum(np.abs(np.diff(turn_signs)) > 1)
            direction_reversal_rate = reversal_count / (len(turn_signs) - 1)

        # 21维基础特征
        base_features = np.array([
            avg_speed, max_speed, std_speed, speed_cv,
            avg_dir_change, max_dir_change, std_dir_change,
            avg_facing_deviation, max_facing_deviation,
            path_length, start_end_dist, tortuosity,
            low_speed_ratio, duration,
            center_x, center_y,
            max_acceleration, std_acceleration,
            turn_frequency, angle_entropy, direction_reversal_rate
        ])

        # 2维禁区特征
        dists = np.sqrt((x - self.forbid_x)**2 + (y - self.forbid_y)**2)
        min_dist_to_forbidden = np.min(dists)
        in_forbidden_ratio = np.sum(dists < self.forbid_radius) / n
        forbid_features = np.array([min_dist_to_forbidden, in_forbidden_ratio])

        # 拼接23维完整特征
        return np.concatenate([base_features, forbid_features]).reshape(1, -1)

    def compute_category_scores(self, x_coords, y_coords, facing_angles=None):
        """
        快捷接口：仅计算三类专项异常分数（不含DBSCAN整体异常度）
        适合实时检测、高频刷新等轻量场景
        """
        if len(x_coords) < 5:
            return {'running': 0.0, 'loitering': 0.0, 'trespassing': 0.0}
        
        traj_df = self._preprocess_trajectory(x_coords, y_coords, facing_angles)
        raw_features = self._extract_features(traj_df)
        return self._compute_category_scores(raw_features)

    def _compute_category_scores(self, feature_raw):
        """内部方法：数据驱动的三类异常分数计算"""
        feat = feature_raw.flatten()
        running_score = 0.0
        loitering_score = 0.0
        trespassing_score = 0.0

        # 1. 奔跑异常评分
        for feat_idx, direction, weight in self.running_feat_config:
            norm_val = self._normalize_feature(feat[feat_idx], feat_idx, direction)
            running_score += weight * norm_val

        # 2. 徘徊异常评分
        for feat_idx, direction, weight in self.loitering_feat_config:
            norm_val = self._normalize_feature(feat[feat_idx], feat_idx, direction)
            loitering_score += weight * norm_val

        # 3. 闯入禁区评分
        for feat_idx, direction, weight in self.trespassing_feat_config:
            norm_val = self._normalize_feature(feat[feat_idx], feat_idx, direction)
            trespassing_score += weight * norm_val

        return {
            'running': round(float(running_score), 4),
            'loitering': round(float(loitering_score), 4),
            'trespassing': round(float(trespassing_score), 4)
        }

    def detect(self, x_coords, y_coords, facing_angles=None, custom_thresholds=None):
        """
        主检测接口：输入坐标序列，输出完整检测结果
        【并发安全】自定义阈值仅对当前请求生效，不修改全局默认值
        :param x_coords: x坐标列表/数组（单位：毫米，与ATC数据集一致）
        :param y_coords: y坐标列表/数组
        :param facing_angles: 可选，朝向角列表
        :param custom_thresholds: 可选，自定义报警阈值字典
        :return: 标准化检测结果字典
        """
        # 入参校验
        if len(x_coords) != len(y_coords):
            return {
                'code': -1,
                'msg': '坐标长度不一致',
                'category_scores': None,
                'total_anomaly_score': None,
                'alarms': [],
                'is_abnormal': False
            }
        if len(x_coords) < 5:
            return {
                'code': -2,
                'msg': '轨迹点过少，至少需要5个坐标点',
                'category_scores': {'running': 0.0, 'loitering': 0.0, 'trespassing': 0.0},
                'total_anomaly_score': 0.0,
                'alarms': [],
                'is_abnormal': False
            }

        # 阈值：使用局部副本，不修改全局默认值，保证并发安全
        thresholds = self.default_thresholds.copy()
        if custom_thresholds:
            thresholds.update(custom_thresholds)

        try:
            # 1. 轨迹预处理
            traj_df = self._preprocess_trajectory(x_coords, y_coords, facing_angles)
            # 2. 特征提取
            raw_features = self._extract_features(traj_df)
            # 3. 标准化 + 计算DBSCAN整体异常度
            std_features = self.scaler.transform(raw_features)
            dist_to_core = euclidean_distances(std_features, self.core_samples)
            total_anomaly_score = float(np.min(dist_to_core))
            # 4. 计算三类专项异常分数
            category_scores = self._compute_category_scores(raw_features)

            # 5. 报警判定
            alarms = []
            if category_scores['running'] > thresholds['running']:
                alarms.append('奔跑异常：检测到行人移动速度远超正常步行范围')
            if category_scores['loitering'] > thresholds['loitering']:
                alarms.append('徘徊异常：检测到行人在区域内长时间往复逗留')
            if category_scores['trespassing'] > thresholds['trespassing']:
                alarms.append('闯入禁区异常：检测到行人进入禁入区域')

            return {
                'code': 0,
                'msg': 'success',
                'category_scores': category_scores,
                'total_anomaly_score': round(total_anomaly_score, 4),
                'alarms': alarms,
                'is_abnormal': len(alarms) > 0
            }

        except Exception as e:
            return {
                'code': -99,
                'msg': f'检测失败：{str(e)}',
                'category_scores': None,
                'total_anomaly_score': None,
                'alarms': [],
                'is_abnormal': False
            }

    # ===================== 功能1：批量轨迹检测 =====================
    def batch_detect(self, trajectory_list, custom_thresholds=None, return_type='dataframe'):
        """
        批量检测多条轨迹
        :param trajectory_list: 轨迹列表，每个元素为 (x_list, y_list) 或 (x_list, y_list, facing_list)
        :param custom_thresholds: 自定义报警阈值
        :param return_type: 返回格式，'dataframe' 或 'list'
        :return: DataFrame 或 字典列表
        """
        results = []
        for idx, traj in enumerate(trajectory_list):
            if len(traj) == 2:
                x, y = traj
                facing = None
            else:
                x, y, facing = traj

            res = self.detect(x, y, facing, custom_thresholds)
            results.append({
                'trajectory_id': idx,
                'running_score': res['category_scores']['running'] if res['code'] == 0 else None,
                'loitering_score': res['category_scores']['loitering'] if res['code'] == 0 else None,
                'trespassing_score': res['category_scores']['trespassing'] if res['code'] == 0 else None,
                'total_anomaly_score': res['total_anomaly_score'],
                'is_abnormal': res['is_abnormal'],
                'alarm_types': '; '.join(res['alarms']) if res['alarms'] else '无',
                'status': res['msg']
            })
        
        if return_type == 'list':
            return results
        else:
            return pd.DataFrame(results)

    # ===================== 功能2：轨迹可视化 =====================
    def plot_trajectory(self, x_coords, y_coords, result=None, save_path=None):
        """
        绘制轨迹图 + 禁区范围 + 检测结果标注
        :param x_coords: x坐标序列
        :param y_coords: y坐标序列
        :param result: detect()返回的检测结果，可选
        :param save_path: 图片保存路径，可选
        """
        x = np.array(x_coords)
        y = np.array(y_coords)

        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制禁区范围
        circle = plt.Circle(
            (self.forbid_x, self.forbid_y), self.forbid_radius,
            color='red', alpha=0.15, label='禁区范围'
        )
        ax.add_patch(circle)
        ax.scatter(self.forbid_x, self.forbid_y, c='red', marker='x', s=100, label='禁区中心')

        # 绘制轨迹，异常标红、正常标绿
        line_color = 'green'
        if result and result['is_abnormal']:
            line_color = 'red'
        ax.plot(x, y, color=line_color, linewidth=2, label='行人轨迹')
        ax.scatter(x[0], y[0], c='blue', s=80, zorder=5, label='起点')
        ax.scatter(x[-1], y[-1], c='orange', s=80, zorder=5, label='终点')

        # 叠加检测结果文本
        if result:
            text = (
                f"奔跑异常分: {result['category_scores']['running']:.2f}\n"
                f"徘徊异常分: {result['category_scores']['loitering']:.2f}\n"
                f"闯入异常分: {result['category_scores']['trespassing']:.2f}\n"
                f"整体异常度: {result['total_anomaly_score']:.2f}"
            )
            ax.text(0.02, 0.98, text, transform=ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.9),
                    verticalalignment='top', fontsize=10)

        ax.set_xlabel('X 坐标 (mm)', fontsize=12)
        ax.set_ylabel('Y 坐标 (mm)', fontsize=12)
        ax.set_title('行人轨迹与异常检测结果', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
        ax.axis('equal')  # 保证坐标比例一致，轨迹不畸变

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


# ===================== 功能3：实时滑动窗口检测器 =====================
class RealtimeTrajectoryDetector:
    """
    实时增量轨迹检测器：滑动窗口维护轨迹，每新增一个坐标点实时输出检测结果
    适用于视频流、实时跟踪系统、GUI实时绘制等场景
    """
    def __init__(self, window_size=100, min_detect_points=10, model_dir='../data/processed/'):
        """
        :param window_size: 滑动窗口最大长度（帧数），超出后丢弃最早的点
        :param min_detect_points: 最少积累多少点才开始输出检测结果
        """
        self.detector = TrajectoryAnomalyDetector(model_dir)
        self.window_size = window_size
        self.min_points = min_detect_points
        self.x_buffer = []
        self.y_buffer = []

    def add_point(self, x, y):
        """新增一个轨迹点，返回当前窗口的检测结果"""
        self.x_buffer.append(float(x))
        self.y_buffer.append(float(y))

        # 维护滑动窗口，丢弃过期点
        if len(self.x_buffer) > self.window_size:
            self.x_buffer.pop(0)
            self.y_buffer.pop(0)

        # 点数不足时返回缓冲状态
        if len(self.x_buffer) < self.min_points:
            return {
                'status': 'buffering',
                'current_points': len(self.x_buffer),
                'min_points': self.min_points,
                'result': None
            }

        # 执行实时检测
        result = self.detector.detect(self.x_buffer, self.y_buffer)
        return {
            'status': 'detecting',
            'current_points': len(self.x_buffer),
            'result': result
        }

    def reset(self):
        """重置缓冲区，切换新行人时调用"""
        self.x_buffer.clear()
        self.y_buffer.clear()


# ===================== 本地运行示例 =====================
if __name__ == '__main__':
    # 初始化检测器
    detector = TrajectoryAnomalyDetector(model_dir='../data/processed/')

    # 示例：生成一段正常直线行走轨迹
    t = np.arange(0, 10, 0.1)
    x_normal = 5000 + 1000 * t  # 1000mm/s 正常步行速度
    y_normal = 3000 + 0 * t

    # 执行单条检测
    result = detector.detect(x_normal, y_normal)

    # 打印结果
    print("="*50)
    print("正常轨迹检测结果")
    print("="*50)
    print(f"奔跑异常分数: {result['category_scores']['running']:.2%}")
    print(f"徘徊异常分数: {result['category_scores']['loitering']:.2%}")
    print(f"闯入禁区分数: {result['category_scores']['trespassing']:.2%}")
    print(f"DBSCAN整体异常度: {result['total_anomaly_score']:.4f}")
    print("-"*30)
    if result['is_abnormal']:
        print("触发警报：")
        for alarm in result['alarms']:
            print(f"  ⚠️  {alarm}")
    else:
        print("✅ 轨迹正常，未检测到异常行为")

    # 可视化轨迹
    detector.plot_trajectory(x_normal, y_normal, result)
