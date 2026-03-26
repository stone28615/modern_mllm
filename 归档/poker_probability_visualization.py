import matplotlib.pyplot as plt
import numpy as np

# ---------------------- 1. 数据准备（来源于文档中n人局最高牌型概率对照表）----------------------
# 玩家数（x轴数据）
player_num = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])

# 各牌型最高概率（y轴数据，单位：%），顺序对应：高牌、一对、两对、三条、顺子、同花、葫芦
prob_data = {
    "高牌": np.array([49.9000, 44.9000, 38.6000, 34.9000, 31.6000, 28.7000, 26.1000, 23.7000, 17.4000]),
    "一对": np.array([39.5000, 35.8000, 31.9000, 29.1000, 26.5000, 24.1000, 21.9000, 19.9000, 18.6000]),
    "两对": np.array([2.3000, 2.2000, 2.1200, 2.0300, 1.9500, 1.8700, 1.7900, 1.7200, 1.6400]),
    "三条": np.array([0.4640, 0.4460, 0.4290, 0.4120, 0.3960, 0.3800, 0.3650, 0.3510, 0.3320]),
    "顺子": np.array([0.0445, 0.0430, 0.0415, 0.0400, 0.0386, 0.0372, 0.0358, 0.0345, 0.0339]),
    "同花": np.array([0.0297, 0.0292, 0.0287, 0.0282, 0.0277, 0.0272, 0.0267, 0.0263, 0.0260]),
    "葫芦": np.array([0.0259, 0.0258, 0.0257, 0.0256, 0.0255, 0.0254, 0.0253, 0.0252, 0.0252])
}

# 定义颜色和线型（区分不同牌型）
colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FECA57", "#FF9FF3", "#54A0FF"]
linestyles = ["-", "--", "-.", ":", "-", "--", "-."]

# ---------------------- 2. 创建可视化图表（正常坐标 + 对数坐标）----------------------
# 设置画布大小和字体（适配中文显示）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))  # 1行2列布局，总宽度16，高度6

# ---------------------- 2.1 正常坐标图（ax1）----------------------
ax1.set_title("德州扑克n人局最高牌型概率变化（正常坐标）", fontsize=14, fontweight='bold')
ax1.set_xlabel("玩家数 n", fontsize=12)
ax1.set_ylabel("最高概率（%）", fontsize=12)
ax1.set_xlim(1.5, 10.5)  # x轴范围微调，避免数据贴边
ax1.grid(True, alpha=0.3)  # 显示网格，透明度0.3

# 绘制各牌型概率曲线
for (card_type, prob), color, linestyle in zip(prob_data.items(), colors, linestyles):
    ax1.plot(player_num, prob, label=card_type, color=color, linestyle=linestyle, linewidth=2, marker='o', markersize=4)

# 添加图例（自动识别label）
ax1.legend(loc='upper right', fontsize=10)

# ---------------------- 2.2 对数坐标图（ax2，y轴对数）----------------------
ax2.set_title("德州扑克n人局最高牌型概率变化（对数坐标）", fontsize=14, fontweight='bold')
ax2.set_xlabel("玩家数 n", fontsize=12)
ax2.set_ylabel("最高概率（%，对数尺度）", fontsize=12)
ax2.set_xlim(1.5, 10.5)
ax2.set_yscale('log')  # 设置y轴为对数坐标
ax2.grid(True, alpha=0.3)

# 绘制各牌型概率曲线（与正常坐标图样式一致，便于对比）
for (card_type, prob), color, linestyle in zip(prob_data.items(), colors, linestyles):
    ax2.plot(player_num, prob, label=card_type, color=color, linestyle=linestyle, linewidth=2, marker='o', markersize=4)

# 添加图例
ax2.legend(loc='upper right', fontsize=10)

# ---------------------- 3. 图表优化与保存 ----------------------
plt.tight_layout()  # 自动调整子图间距，避免标签重叠
# 保存图片（支持多种格式，png清晰度高，可调整dpi）
plt.savefig("德州扑克n人局概率变化图.png", dpi=300, bbox_inches='tight')
# 显示图表
plt.show()

# ---------------------- 4. 可选：输出数据统计信息 ----------------------
print("各牌型概率统计（单位：%）：")
print("-" * 60)
for card_type, prob in prob_data.items():
    print(f"{card_type:6s} | 最大值：{np.max(prob):6.4f} | 最小值：{np.min(prob):6.4f} | 变化幅度：{np.max(prob)-np.min(prob):6.4f}")
print("-" * 60)
