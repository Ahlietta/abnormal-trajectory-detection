import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

# ===================== 新增：Matplotlib 中文全局配置 =====================
# 按优先级匹配系统自带中文字体，Windows优先微软雅黑，mac优先黑体，Linux优先文泉驿
matplotlib.rcParams['font.sans-serif'] = [
    'Microsoft YaHei',  # Windows 微软雅黑
    'SimHei',           # Windows 黑体
    'PingFang SC',      # macOS 苹方
    'Arial Unicode MS', # macOS 系统黑体
    'WenQuanYi Micro Hei' # Linux 文泉驿微米黑
]
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
# ========================================================================

from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from trajectory_detector import TrajectoryAnomalyDetector


class TrajectoryDetectorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ATC商场行人轨迹异常检测系统")
        self.geometry("1280x850")
        self.resizable(True, True)

        # ===================== 新增：Tkinter 控件全局中文字体 =====================
        style = ttk.Style(self)
        # 统一所有ttk控件的字体为微软雅黑，确保按钮、标签、单选框中文正常
        style.configure('.', font=('微软雅黑', 10))
        style.configure('TLabel', font=('微软雅黑', 10))
        style.configure('TButton', font=('微软雅黑', 10))
        style.configure('TRadiobutton', font=('微软雅黑', 10))
        style.configure('Treeview', font=('微软雅黑', 9))
        style.configure('Treeview.Heading', font=('微软雅黑', 10, 'bold'))
        # ========================================================================

        # ========== 1. 初始化检测器与配置 ==========
        self.detector = TrajectoryAnomalyDetector(model_dir='../data/processed/')
        # 规范调用接口获取禁区配置
        forbid_zone = self.detector.get_forbidden_zone()
        self.forbid_x = forbid_zone['center_x']
        self.forbid_y = forbid_zone['center_y']
        self.forbid_radius = forbid_zone['radius']

        # ========== 2. 状态变量 ==========
        self.mode = tk.StringVar(value="batch")  # batch=批量模式, realtime=实时模式
        self.is_drawing = False       # 是否正在绘制
        self.draw_locked = False      # 绘制结束锁定，必须清除才能重画
        self.current_x = []           # 当前轨迹x坐标
        self.current_y = []           # 当前轨迹y坐标
        self.batch_samples = []       # 批量模式：已保存的样本列表

        # ========== 3. 构建界面 ==========
        self._build_control_panel()   # 顶部控制面板
        self._build_main_area()       # 主内容区（画布+结果面板）

        # ========== 4. 初始化画布与事件 ==========
        self._init_draw_canvas()
        self._bind_mouse_events()

        # 初始刷新界面状态
        self._switch_mode()

    # ===================== 界面构建 =====================
    def _build_control_panel(self):
        """顶部控制面板：模式选择、操作按钮、样本计数"""
        control_frame = ttk.Frame(self, padding=12)
        control_frame.pack(fill=tk.X)

        # 模式选择组
        ttk.Label(control_frame, text="检测模式：", font=("微软雅黑", 10, "bold")).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(
            control_frame, text="批量检测模式", variable=self.mode,
            value="batch", command=self._switch_mode
        ).pack(side=tk.LEFT, padx=8)
        ttk.Radiobutton(
            control_frame, text="实时检测模式", variable=self.mode,
            value="realtime", command=self._switch_mode
        ).pack(side=tk.LEFT, padx=8)

        # 样本计数
        self.sample_count_label = ttk.Label(
            control_frame, text="已保存样本：0 条",
            font=("微软雅黑", 10), foreground="#1f77b4"
        )
        self.sample_count_label.pack(side=tk.LEFT, padx=30)

        # 操作按钮组（从右到左排列）
        self.btn_reset_all = ttk.Button(control_frame, text="重置全部", command=self._reset_all)
        self.btn_reset_all.pack(side=tk.RIGHT, padx=6)
        self.btn_clear_current = ttk.Button(control_frame, text="清除当前轨迹", command=self._clear_current)
        self.btn_clear_current.pack(side=tk.RIGHT, padx=6)
        self.btn_batch_detect = ttk.Button(control_frame, text="批量检测全部样本", command=self._run_batch_detect)
        self.btn_batch_detect.pack(side=tk.RIGHT, padx=6)
        self.btn_save_sample = ttk.Button(control_frame, text="保存当前样本", command=self._save_current_sample)
        self.btn_save_sample.pack(side=tk.RIGHT, padx=6)

    def _build_main_area(self):
        """主内容区：左侧绘图画布 + 右侧结果面板"""
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=5)

        # 画布容器
        self.canvas_frame = ttk.LabelFrame(main_frame, text="轨迹绘制区", padding=8)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 右侧结果面板
        result_frame = ttk.LabelFrame(main_frame, text="检测结果", padding=15, width=280)
        result_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=12)
        result_frame.pack_propagate(False)

        # 分数显示项
        self.result_labels = {}
        score_items = [
            ("running", "奔跑异常分："),
            ("loitering", "徘徊异常分："),
            ("trespassing", "闯入禁区分：")
        ]

        for key, label_text in score_items:
            ttk.Label(result_frame, text=label_text, font=("微软雅黑", 10)).pack(anchor=tk.W, pady=3)
            lbl = ttk.Label(result_frame, text="0.0000", font=("微软雅黑", 13, "bold"), foreground="green")
            lbl.pack(anchor=tk.W, pady=2)
            self.result_labels[key] = lbl

        ttk.Separator(result_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)

        ttk.Label(result_frame, text="整体异常度：", font=("微软雅黑", 10)).pack(anchor=tk.W, pady=3)
        self.lbl_total_score = ttk.Label(result_frame, text="0.0000", font=("微软雅黑", 13, "bold"))
        self.lbl_total_score.pack(anchor=tk.W, pady=2)

        ttk.Separator(result_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)

        ttk.Label(result_frame, text="状态：", font=("微软雅黑", 10)).pack(anchor=tk.W, pady=3)
        self.lbl_alarm_status = ttk.Label(
            result_frame, text="✅ 轨迹正常", font=("微软雅黑", 11, "bold"),
            foreground="green", wraplength=240, justify=tk.LEFT
        )
        self.lbl_alarm_status.pack(anchor=tk.W, pady=3)

        # 操作提示
        tip_text = (
            "\n--- 操作说明 ---\n"
            "• 按住鼠标左键绘制轨迹\n"
            "• 松开左键/移出画布结束\n"
            "• 结束后需清除才能画下一条"
        )
        ttk.Label(
            result_frame, text=tip_text, font=("微软雅黑", 9),
            foreground="gray", justify=tk.LEFT
        ).pack(anchor=tk.W, pady=20, side=tk.BOTTOM)

    # ===================== 画布初始化 =====================
    def _init_draw_canvas(self):
        """初始化绘图画布，设置坐标范围与禁区标注"""
        self.fig = Figure(figsize=(9, 7.5), dpi=100)
        self.ax = self.fig.add_subplot(111)

        # 坐标范围严格符合要求：中心为原点，x±50000，y±40000
        self.ax.set_xlim(-50000, 50000)
        self.ax.set_ylim(-40000, 40000)
        self.ax.set_xlabel("X 坐标 (mm)", fontsize=11)
        self.ax.set_ylabel("Y 坐标 (mm)", fontsize=11)
        self.ax.set_title("ATC 场景轨迹绘制画布（中心为坐标原点）", fontsize=12, pad=10)
        self.ax.grid(alpha=0.3, linestyle='--')
        self.ax.set_aspect('equal', adjustable='box')  # 等比例，轨迹不失真

        # 绘制禁区范围
        forbid_area = Circle(
            (self.forbid_x, self.forbid_y), self.forbid_radius,
            color='red', alpha=0.18, label='禁区范围'
        )
        self.ax.add_patch(forbid_area)
        self.ax.scatter(self.forbid_x, self.forbid_y, c='red', marker='x', s=120, linewidths=2, label='禁区中心')
        self.ax.legend(loc='upper right', fontsize=10)

        # 当前轨迹线（初始为空）
        self.traj_line, = self.ax.plot([], [], color='#1f77b4', linewidth=2.2, label='当前轨迹')

        # 嵌入Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _bind_mouse_events(self):
        """绑定鼠标交互事件"""
        self.fig.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.fig.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.fig.canvas.mpl_connect('figure_leave_event', self._on_mouse_leave)

    # ===================== 模式切换 =====================
    def _switch_mode(self):
        """切换批量/实时模式，同步更新按钮状态"""
        self._reset_all()
        if self.mode.get() == "batch":
            self.btn_save_sample.config(state=tk.NORMAL)
            self.btn_batch_detect.config(state=tk.NORMAL)
            self.sample_count_label.config(text="已保存样本：0 条")
        else:
            self.btn_save_sample.config(state=tk.DISABLED)
            self.btn_batch_detect.config(state=tk.DISABLED)
            self.sample_count_label.config(text="实时模式：绘制过程同步检测")

    # ===================== 鼠标事件处理 =====================
    def _on_mouse_press(self, event):
        """鼠标左键按下：开始绘制"""
        # 仅左键、在坐标轴内、未锁定时生效
        if event.button != 1 or event.inaxes != self.ax or self.draw_locked:
            return
        self.is_drawing = True
        self.current_x = [event.xdata]
        self.current_y = [event.ydata]

    def _on_mouse_move(self, event):
        """鼠标移动：记录轨迹点，实时更新画布"""
        if not self.is_drawing or event.inaxes != self.ax:
            return

        # 记录坐标点
        self.current_x.append(event.xdata)
        self.current_y.append(event.ydata)

        # 更新轨迹显示
        self.traj_line.set_data(self.current_x, self.current_y)
        self.canvas.draw_idle()

        # 实时模式：点数足够时同步计算结果
        if self.mode.get() == "realtime" and len(self.current_x) >= 5:
            self._update_realtime_result()

    def _on_mouse_release(self, event):
        """鼠标左键松开：结束绘制，锁定画布"""
        if event.button != 1 or not self.is_drawing:
            return
        self.is_drawing = False
        self.draw_locked = True  # 锁定，必须清除才能重画

        # 实时模式最终刷新一次结果
        if self.mode.get() == "realtime" and len(self.current_x) >= 5:
            self._update_realtime_result()

    def _on_mouse_leave(self, event):
        """鼠标移出画布：强制结束绘制"""
        if self.is_drawing:
            self.is_drawing = False
            self.draw_locked = True
            if self.mode.get() == "realtime" and len(self.current_x) >= 5:
                self._update_realtime_result()

    # ===================== 检测逻辑 =====================
    def _update_realtime_result(self):
        """实时模式：更新检测结果到界面，带错误保护"""
        try:
            result = self.detector.detect(self.current_x, self.current_y)
            if result['code'] == 0:  # 仅检测成功时刷新界面
                self._refresh_result_display(result)
        except Exception:
            pass

    def _refresh_result_display(self, result):
        """刷新右侧结果面板的分数与状态"""
        scores = result['category_scores']

        # 更新分数与颜色
        for key, label in self.result_labels.items():
            score = scores[key]
            label.config(text=f"{score:.4f}")
            # 超过阈值标红，正常标绿
            threshold = self.detector.default_thresholds[key]
            label.config(foreground="red" if score > threshold else "green")

        # 整体异常度
        self.lbl_total_score.config(text=f"{result['total_anomaly_score']:.4f}")

        # 报警状态
        if result['is_abnormal']:
            alarm_text = "⚠️ 触发警报：\n" + "\n".join(result['alarms'])
            self.lbl_alarm_status.config(text=alarm_text, foreground="red")
        else:
            self.lbl_alarm_status.config(text="✅ 轨迹正常", foreground="green")

    # ===================== 批量模式功能 =====================
    def _save_current_sample(self):
        """保存当前轨迹为批量样本"""
        if self.mode.get() != "batch":
            messagebox.showinfo("提示", "仅批量检测模式可保存样本")
            return
        if len(self.current_x) < 5:
            messagebox.showwarning("警告", "轨迹点过少（至少需要5个点），无法保存")
            return

        # 加入样本库
        self.batch_samples.append((np.array(self.current_x), np.array(self.current_y)))
        self.sample_count_label.config(text=f"已保存样本：{len(self.batch_samples)} 条")

        # 自动清空当前轨迹，准备绘制下一条
        self._clear_current()
        messagebox.showinfo("保存成功", f"已保存第 {len(self.batch_samples)} 条轨迹样本")

    def _run_batch_detect(self):
        """执行批量检测，弹出结果窗口"""
        if len(self.batch_samples) == 0:
            messagebox.showwarning("警告", "请先保存至少一条轨迹样本")
            return

        # 调用批量检测接口（默认返回DataFrame，兼容原有展示逻辑）
        result_df = self.detector.batch_detect(self.batch_samples)

        # 弹出结果窗口
        result_win = tk.Toplevel(self)
        result_win.title("批量检测结果汇总")
        result_win.geometry("900x550")

        # 表格展示
        columns = list(result_df.columns)
        tree = ttk.Treeview(result_win, columns=columns, show='headings')
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=110, anchor=tk.CENTER)

        for _, row in result_df.iterrows():
            tree.insert('', tk.END, values=list(row))

        tree.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # 导出按钮
        def export_to_csv():
            save_path = "../results/gui_batch_result.csv"
            result_df.to_csv(save_path, index=False, encoding='utf-8-sig')
            messagebox.showinfo("导出成功", f"结果已保存至：\n{save_path}")

        btn_frame = ttk.Frame(result_win)
        btn_frame.pack(fill=tk.X, padx=12, pady=5)
        ttk.Button(btn_frame, text="导出为CSV文件", command=export_to_csv).pack(side=tk.RIGHT)

    # ===================== 清除与重置 =====================
    def _clear_current(self):
        """清除当前轨迹，解锁画布"""
        self.current_x = []
        self.current_y = []
        self.traj_line.set_data([], [])
        self.canvas.draw_idle()
        self.draw_locked = False

        # 重置结果显示
        for label in self.result_labels.values():
            label.config(text="0.0000", foreground="green")
        self.lbl_total_score.config(text="0.0000")
        self.lbl_alarm_status.config(text="✅ 轨迹正常", foreground="green")

    def _reset_all(self):
        """重置所有状态：清空样本、清空轨迹、解锁"""
        self._clear_current()
        self.batch_samples.clear()
        self.sample_count_label.config(text="已保存样本：0 条")
        self.is_drawing = False
        self.draw_locked = False


if __name__ == '__main__':
    app = TrajectoryDetectorApp()
    app.mainloop()
