import serial
import threading
import time
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import classification
import utils

# 全局变量
acc_data = {
    'time': [],
    'ax': [], 'ay': [], 'az': [],
    'gx': [], 'gy': [], 'gz': [],
    'roll': [], 'pitch': [], 'yaw': []
}
running = True
position = None


def parse_imu_line(line):
    # 假设格式：ax,ay,az,gx,gy,gz,roll,pitch,yaw
    try:
        parts = list(map(float, line.strip().split(',')))
        if len(parts) == 9:
            return parts
    except Exception as e:
        print("Parse error:", e)
    return None


def read_serial_data(serial_port, baud_rate, save_path):
    global acc_data
    ser = serial.Serial(serial_port, baud_rate)
    with open(save_path, 'w') as f:
        f.write("gx,gy,gz,ax,ay,az,roll,pitch,yaw\n")
        start_time = time.time()
        while running:
            if ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='ignore')
                data = parse_imu_line(line)
                if data:
                    f.write(line.strip("\n"))
                    f.flush()
                    t = time.time() - start_time
                    acc_data['time'].append(t)
                    acc_data['gx'].append(data[0])
                    acc_data['gy'].append(data[1])
                    acc_data['gz'].append(data[2])
                    acc_data['ax'].append(data[3])
                    acc_data['ay'].append(data[4])
                    acc_data['az'].append(data[5])
                    acc_data['roll'].append(data[6])
                    acc_data['pitch'].append(data[7])
                    acc_data['yaw'].append(data[8])
    ser.close()


def start_serial_thread(port, baud, path):
    thread = threading.Thread(target=read_serial_data, args=(port, baud, path))
    thread.start()


def update_plot(canvas, lines, axes):
    if acc_data['time']:
        t = acc_data['time']
        for key, line in lines.items():
            line.set_data(t, acc_data[key])
            axes[key].relim()
            axes[key].autoscale_view()
        canvas.draw()
    if running:
        canvas.get_tk_widget().after(100, update_plot, canvas, lines, axes)


def calculate_displacement():
    """根据加速度数据计算位移"""
    global acc_data, position
    # sample_rate = 100  # 假设采样率为 100 Hz
    position = utils.parse_position(acc_data)

    return position[:, 0], position[:, 1], position[:, 2]


def distinguish_trajectory():
    """轨迹识别"""
    global position
    model = classification.get_model('xgboost')
    position_parsed = classification.parse_position_sequence('xgboost', position) 
    pred = model.predict(position_parsed)
    print(pred)

    return pred


def create_ui():
    root = tk.Tk()
    root.title("Magic Hand Viewer")
    root.geometry("800x600")

    # 控件区域
    control_frame = tk.Frame(root)
    control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

    tk.Label(control_frame, text="Serial Port:").grid(row=0, column=0, sticky="e")
    port_entry = ttk.Entry(control_frame, width=10)
    port_entry.insert(0, "COM18")
    port_entry.grid(row=0, column=1)

    tk.Label(control_frame, text="Baud Rate:").grid(row=0, column=2, sticky="e")
    baud_entry = ttk.Entry(control_frame, width=10)
    baud_entry.insert(0, "9600")
    baud_entry.grid(row=0, column=3)

    def choose_path():
        path = filedialog.asksaveasfilename(defaultextension=".csv")
        path_entry.delete(0, tk.END)
        path_entry.insert(0, path)

    tk.Button(control_frame, text="Save File As...", command=choose_path).grid(row=0, column=4, padx=5)
    path_entry = ttk.Entry(control_frame, width=30)
    path_entry.grid(row=0, column=5, padx=5)

    tk.Button(control_frame, text="Start", command=lambda: start()).grid(row=0, column=6, padx=5)
    tk.Button(control_frame, text="Stop", command=lambda: stop()).grid(row=0, column=7, padx=5)

    # 图像区域
    fig, axs = plt.subplots(3, 2, figsize=(8, 6))
    fig.tight_layout(pad=3.0)
    axes = {}
    lines = {}

    keys = ['ax', 'ay', 'az', 'roll', 'pitch', 'yaw']
    titles = ['Ax (m/s²)', 'Ay (m/s²)', 'Az (m/s²)', 'Roll (°)', 'Pitch (°)', 'Yaw (°)']

    for ax, key, title in zip(axs.flatten(), keys, titles):
        line, = ax.plot([], [], label=key)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(key)
        ax.grid(True)
        lines[key] = line
        axes[key] = ax

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)

    # 弹出位移路径面板的按钮
    def show_displacement_window():
        displacement_window = tk.Toplevel(root)
        displacement_window.title("Displacement Path")
        displacement_window.geometry("1000x400")

        displacement_fig, displacement_axs = plt.subplots(1, 2, figsize=(5, 4))
        displacement_canvas = FigureCanvasTkAgg(displacement_fig, master=displacement_window)
        displacement_canvas_widget = displacement_canvas.get_tk_widget()
        displacement_canvas_widget.pack(fill=tk.BOTH, expand=True)

        # 绘制轨迹
        x, y, z = calculate_displacement()
        displacement_axs[0].plot(x, y, label="Displacement Path", color="blue")
        displacement_axs[0].scatter(x[-1], y[-1], color="red", label="Current Position")
        displacement_axs[0].set_title("Displacement Path")
        displacement_axs[0].set_xlabel("X Position (m)")
        displacement_axs[0].set_ylabel("Y Position (m)")
        displacement_axs[0].legend()
        displacement_axs[0].grid(True)

        # 绘制分类概率
        methods = ['method'+str(i) for i in range(107)]

        pred = distinguish_trajectory().flatten()
        top_indices = utils.topN_indices(pred, N=10)
        legend = [methods[i] for i in top_indices]
        values = [pred[i] for i in top_indices]
        bars = displacement_axs[1].barh(legend, values, edgecolor='black', alpha=0.8)

        # 添加准确率数值标签
        for bar, acc in zip(bars, pred[top_indices]):
            displacement_axs[1].text(acc+0.01, bar.get_y() + bar.get_height()/2, f'{acc:.5f}', va='center', fontsize=9)

        # 图表设置
        displacement_axs[1].set_xlabel("Accuracy")
        displacement_axs[1].set_ylabel("Methods")
        displacement_axs[1].set_title("Comparison of Accuracy for Different Methods")

        # displacement_axs[1].xlim(0, 1.05)  # 设置X轴范围
        displacement_axs[1].grid(True)  # 添加网格线
        # displacement_axs[1].tight_layout()  # 自动调整布局

        displacement_canvas.draw()

    tk.Button(control_frame, text="Generate Path", command=show_displacement_window).grid(row=0, column=8, padx=5)

    def start():
        global running
        running = True
        port = port_entry.get()
        baud = int(baud_entry.get())
        path = path_entry.get()
        if port and baud and path:
            start_serial_thread(port, baud, path)
            update_plot(canvas, lines, axes)


    def stop():
        global running
        running = False

    root.mainloop()


if __name__ == "__main__":
    create_ui()
    running = False
    # distinguish_trajectory(position)
