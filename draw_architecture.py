"""
YOLOv12s 网络架构图（可发表级别）
输出：SVG + PNG 格式，可直接用于论文
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Arial']
plt.rcParams['font.size'] = 9

def draw_conv(ax, x, y, w, h, label, sublabel=""):
    """绘制 Conv 模块"""
    color = '#E8D5B7'
    edge = '#8B7355'
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                         facecolor=color, edgecolor=edge, linewidth=1.2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2 + 0.06, label, ha='center', va='center',
            fontsize=7.5, fontweight='bold', color='#4A3728')
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.1, sublabel, ha='center', va='center',
                fontsize=6.5, color='#6B5344')

def draw_c3k2(ax, x, y, w, h, label="C3k2", sublabel=""):
    """绘制 C3k2 模块"""
    color = '#B8D4E8'
    edge = '#4A7A9B'
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                         facecolor=color, edgecolor=edge, linewidth=1.2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2 + 0.06, label, ha='center', va='center',
            fontsize=7.5, fontweight='bold', color='#2C4A5E')
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.1, sublabel, ha='center', va='center',
                fontsize=6.5, color='#3A607A')

def draw_attention(ax, x, y, w, h, label="R-Attention", sublabel="Area Attn"):
    """绘制 R-Attention 模块"""
    color = '#D4E8C4'
    edge = '#4A8B5B'
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                         facecolor=color, edgecolor=edge, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2 + 0.06, label, ha='center', va='center',
            fontsize=7.5, fontweight='bold', color='#2C5E3A')
    ax.text(x + w/2, y + h/2 - 0.1, sublabel, ha='center', va='center',
            fontsize=6.5, color='#3A7A4A')

def draw_sppf(ax, x, y, w, h):
    """绘制 SPPF 模块"""
    color = '#E8D4C4'
    edge = '#9B6A4A'
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                         facecolor=color, edgecolor=edge, linewidth=1.2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2 + 0.06, "SPPF", ha='center', va='center',
            fontsize=7.5, fontweight='bold', color='#5A3A28')
    ax.text(x + w/2, y + h/2 - 0.1, "5/9/13", ha='center', va='center',
            fontsize=6.5, color='#7A5A4A')

def draw_head(ax, x, y, w, h, label="Detect"):
    """绘制检测头"""
    color = '#E8C4D4'
    edge = '#8B4A7A'
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                         facecolor=color, edgecolor=edge, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2 + 0.06, label, ha='center', va='center',
            fontsize=7.5, fontweight='bold', color='#5A2A4A')
    ax.text(x + w/2, y + h/2 - 0.1, "3-scale", ha='center', va='center',
            fontsize=6.5, color='#8B4A7A')

def draw_concat(ax, x, y, w, h, label="Concat"):
    """绘制 Concat 模块"""
    color = '#D4D4D4'
    edge = '#6A6A6A'
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                         facecolor=color, edgecolor=edge, linewidth=1.0, linestyle='--')
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=6.5, color='#3A3A3A')

def draw_upsample(ax, x, y, w, h):
    color = '#E0E0E0'
    edge = '#909090'
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                         facecolor=color, edgecolor=edge, linewidth=0.8)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, "Upsample", ha='center', va='center',
            fontsize=6.5, color='#3A3A3A')

def arrow(ax, x1, y1, x2, y2, color='#555555'):
    """绘制箭头"""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

def create_figure():
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('#FAFAFA')

    # ===== BACKBONE =====
    bw, bh = 0.9, 0.55
    y_row1 = 7.2   # 浅层
    y_row2 = 6.0   # 中层
    y_row3 = 4.8   # 深层

    # Stem
    sx, sy = 0.3, 8.2
    draw_conv(ax, sx, sy, bw, bh, "Conv", "3×3 s2")
    arrow(ax, sx + bw, sy + bh/2, sx + 1.2, sy + bh/2)

    # Layer 1: Conv 3x3
    draw_conv(ax, sx + 1.3, sy, bw, bh, "Conv", "3×3 s2")
    arrow(ax, sx + 1.3 + bw, sy + bh/2, sx + 2.5, sy + bh/2)

    # Layer 2: C3k2 (浅层第一个)
    draw_c3k2(ax, sx + 2.6, sy, bw, bh, "C3k2", "k=3")
    arrow(ax, sx + 2.6 + bw, sy + bh/2, sx + 3.8, sy + bh/2)

    # Layer 3: Conv 下采样
    draw_conv(ax, sx + 3.9, sy - 0.5, bw, bh, "Conv", "3×3 s2")
    arrow(ax, sx + 3.9 + bw, sy - 0.5 + bh/2, sx + 5.1, sy - 0.5 + bh/2)

    # Layer 4: C3k2 (中层)
    draw_c3k2(ax, sx + 5.2, sy - 0.5, bw, bh, "C3k2", "k=3")
    # Arrow down to layer 5
    arrow(ax, sx + 5.2 + bw/2, sy - 0.5, sx + 5.2 + bw/2, sy - 1.3)
    arrow(ax, sx + 5.2 + bw, sy - 0.5 + bh/2, sx + 6.4, sy - 0.5 + bh/2)

    # Layer 5: Conv 下采样 (Row 2)
    draw_conv(ax, sx + 3.9, y_row1, bw, bh, "Conv", "3×3 s2")
    arrow(ax, sx + 5.2 + bw/2, sy - 0.5, sx + 3.9 + bw/2, y_row1 + bh)
    arrow(ax, sx + 3.9 + bw, y_row1 + bh/2, sx + 4.8, y_row1 + bh/2)

    # Layer 5: Conv 下采样
    draw_conv(ax, sx + 3.9, y_row2, bw, bh, "Conv", "3×3 s2")
    arrow(ax, sx + 5.2 + bw/2, sy - 0.5, sx + 3.9 + bw/2, y_row2 + bh)

    # Layer 5: Conv (row3)
    draw_conv(ax, sx + 3.9, y_row3, bw, bh, "Conv", "3×3 s2")
    arrow(ax, sx + 5.2 + bw/2, sy - 0.5, sx + 3.9 + bw/2, y_row3 + bh)

    # ===== R-ATTENTION BLOCK (YOLOv12 核心创新) =====
    # 在 layer 6 位置
    rx, ry = sx + 6.5, sy - 0.5
    draw_attention(ax, rx, ry, bw + 0.1, bh, "R-Attention", "Area+Residual")
    arrow(ax, sx + 5.2 + bw, y_row2 + bh/2, rx, ry + bh/2)

    # ===== 右侧深层主干 =====
    # Layer 7: Conv 下采样
    draw_conv(ax, rx + bw + 0.1 + 0.3, y_row3 + 0.3, bw, bh, "Conv", "3×3 s2")
    arrow(ax, rx + bw + 0.1, ry + bh/2, rx + bw + 0.1 + 0.3, y_row3 + 0.3 + bh/2)

    # Layer 8: R-Attention (深层)
    draw_attention(ax, rx + bw + 0.1 + 0.3 + 1.2, y_row3 + 0.3, bw + 0.1, bh, "R-Attention", "Area+Residual")
    arrow(ax, rx + bw + 0.1 + 0.3 + bw, y_row3 + 0.3 + bh/2, rx + bw + 0.1 + 0.3 + 1.2, y_row3 + 0.3 + bh/2)

    # ===== SPPF =====
    spx, spy = rx + bw + 0.1 + 0.3 + 1.2 + bw + 0.1 + 0.3, y_row3 + 0.3
    draw_sppf(ax, spx, spy, bw, bh)
    arrow(ax, rx + bw + 0.1 + 0.3 + 1.2 + bw + 0.1, y_row3 + 0.3 + bh/2, spx, spy + bh/2)

    # ===== P5 输出标记 =====
    ax.annotate("P5\n20×20", xy=(spx + bw/2, spy - 0.1), xytext=(spx + bw/2, spy - 0.6),
                ha='center', fontsize=7, color='#4A7A9B', fontweight='bold')

    # ===== NECK: PAN-FPN =====
    neck_x = spx + 2.0

    # Upsample + Concat (Top-down FPN)
    draw_upsample(ax, neck_x, spy + 0.5, bw, bh)
    draw_concat(ax, neck_x + bw + 0.1, spy + 0.5, bw, bh, "Concat")
    arrow(ax, spx + bw, spy + bh/2, neck_x, spy + 0.5 + bh/2)
    arrow(ax, neck_x + bw, spy + 0.5 + bh/2, neck_x + bw + 0.1, spy + 0.5 + bh/2)

    # C3k2 in neck
    draw_c3k2(ax, neck_x + 2*bw + 0.2, spy + 0.5, bw, bh, "C3k2", "FPN-P4")
    arrow(ax, neck_x + bw + 0.1 + bw, spy + 0.5 + bh/2, neck_x + 2*bw + 0.2, spy + 0.5 + bh/2)

    # Second upsample + concat
    draw_upsample(ax, neck_x, spy + 0.5 + bh + 0.4, bw, bh)
    draw_concat(ax, neck_x + bw + 0.1, spy + 0.5 + bh + 0.4, bw, bh, "Concat")
    # 连接来自 layer 4 的跳跃连接
    arrow(ax, sx + 5.2 + bw, sy - 0.5 + bh/2, neck_x + bw/2, spy + 0.5 + bh + 0.4 + bh, color='#999999')
    arrow(ax, neck_x + bw, spy + 0.5 + bh + 0.4 + bh/2, neck_x + bw + 0.1, spy + 0.5 + bh + 0.4 + bh/2)

    # C3k2 in neck (P3)
    draw_c3k2(ax, neck_x + 2*bw + 0.2, spy + 0.5 + bh + 0.4, bw, bh, "C3k2", "FPN-P3")

    # PAN: Downsample paths
    # P4 path
    draw_conv(ax, neck_x + 2*bw + 0.2 + bw + 0.4, spy + 0.5 + bh + 0.4 + bh/2 - bh/2, bw, bh, "Conv", "3×3 s2")
    arrow(ax, neck_x + 2*bw + 0.2 + bw, spy + 0.5 + bh + 0.4, neck_x + 2*bw + 0.2 + bw + 0.4, spy + 0.5 + bh + 0.4 + bh/2 - bh/2 + bh/2)

    # ===== Detect Head =====
    head_x = neck_x + 2*bw + 0.2 + bw + 0.4 + bw + 0.4
    draw_head(ax, head_x, spy + 0.5, bw * 2, bh * 3, "Detect")
    arrow(ax, neck_x + 2*bw + 0.2 + bw, spy + 0.5 + bh/2, head_x, spy + 0.5 + bh * 1.5)
    arrow(ax, neck_x + 2*bw + 0.2 + bw/2, spy + 0.5 + bh + 0.4 + bh/2, head_x, spy + 0.5 + bh * 1.5 - 0.2)
    arrow(ax, neck_x + 2*bw + 0.2 + bw/2, spy + 0.5 + bh + 0.4 + bh/2, head_x, spy + 0.5 + bh * 1.5 + 0.2)

    # ===== Labels =====
    # Backbone label
    ax.text(2.5, 9.3, "Backbone", ha='center', va='center',
            fontsize=11, fontweight='bold', color='#2C4A5E',
            bbox=dict(boxstyle='round', facecolor='#B8D4E8', edgecolor='#4A7A9B', pad=0.3))
    # R-Attention 标注
    ax.annotate("", xy=(rx + bw/2 + 0.05, ry - 0.05), xytext=(rx + bw/2 + 0.05, ry + bh + 0.15),
                arrowprops=dict(arrowstyle='<->', color='#4A8B5B', lw=1.0))
    ax.text(rx + bw/2 + 0.2, ry + bh/2, "★ YOLOv12\n  Innovation", ha='left', va='center',
            fontsize=7, color='#4A8B5B', style='italic')

    # Neck label
    ax.text(12.5, 9.3, "Neck (PAN-FPN)", ha='center', va='center',
            fontsize=11, fontweight='bold', color='#5A3A28',
            bbox=dict(boxstyle='round', facecolor='#E8D5B7', edgecolor='#8B7355', pad=0.3))

    # Head label
    ax.text(head_x + bw, 9.3, "Head", ha='center', va='center',
            fontsize=11, fontweight='bold', color='#5A2A4A',
            bbox=dict(boxstyle='round', facecolor='#E8C4D4', edgecolor='#8B4A7A', pad=0.3))

    # ===== Legend =====
    legend_x = 0.3
    legend_y = 2.0
    ax.text(legend_x, legend_y + 0.6, "Legend:", fontsize=8, fontweight='bold', color='#333333')

    items = [
        ('#E8D5B7', '#8B7355', "Conv (卷积层)"),
        ('#B8D4E8', '#4A7A9B', "C3k2 (特征块)"),
        ('#D4E8C4', '#4A8B5B', "R-Attention (注意力)"),
        ('#E8D4C4', '#9B6A4A', "SPPF (空间金字塔)"),
        ('#D4D4D4', '#6A6A6A', "Concat (特征拼接)"),
        ('#E8C4D4', '#8B4A7A', "Detect (检测头)"),
    ]
    for i, (fc, ec, name) in enumerate(items):
        bx = legend_x + i * 2.5
        rect = FancyBboxPatch((bx, legend_y), 0.4, 0.25, boxstyle="round,pad=0.02",
                              facecolor=fc, edgecolor=ec, linewidth=1.0)
        ax.add_patch(rect)
        ax.text(bx + 0.5, legend_y + 0.12, name, va='center', fontsize=7, color='#333333')

    # ===== Title =====
    ax.text(8, 9.7, "YOLOv12s Network Architecture", ha='center', va='center',
            fontsize=14, fontweight='bold', color='#1A1A2E',
            path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    ax.text(8, 9.25, "Backbone + Neck (PAN-FPN) + Decoupled Head", ha='center', va='center',
            fontsize=9, color='#555555', style='italic')

    # ===== Branch labels =====
    ax.text(sx + 4.5, 8.0, "P4", ha='center', fontsize=7, color='#4A7A9B', fontweight='bold')
    ax.text(sx + 4.5, 6.8, "P3", ha='center', fontsize=7, color='#4A7A9B', fontweight='bold')
    ax.text(sx + 4.5, 5.6, "P5", ha='center', fontsize=7, color='#4A7A9B', fontweight='bold')

    plt.tight_layout(pad=0.5)

    # 保存
    fig.savefig("yolov12_architecture.png", dpi=300, bbox_inches='tight',
                facecolor='#FAFAFA', edgecolor='none')
    fig.savefig("yolov12_architecture.svg", bbox_inches='tight',
                facecolor='#FAFAFA', edgecolor='none')
    print("已保存: yolov12_architecture.png 和 yolov12_architecture.svg")
    plt.close(fig)

if __name__ == "__main__":
    create_figure()
