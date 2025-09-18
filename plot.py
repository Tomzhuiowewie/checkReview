# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager as fm
import numpy as np
import pandas as pd
font_path = "./input/YaHei.ttf"  # 或指定你系统中存在的中文字体
my_font = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = my_font.get_name()

def plot_pie(df, save_path=None, paras=None):
    values,labels = df['values'],df['name']
    font = FontProperties(fname=font_path, size=12, weight='bold')
    gray_color = '#444444'

    # 饼图半径设定
    radius = 1.0
    explode = [0.01] * len(values)

    # 饼图设置
    fig, ax = plt.subplots(figsize=(6, 3), dpi=300)

    # 使用自定义颜色
    custom_colors = [
        '#0E7367', '#19B29F', '#4A4A4A', '#DADADA',
        '#9AE1E7', '#E0FAFD', '#3D6660', '#227064',
        '#666666', '#808080', '#A6A6A6', '#CCCCCC'
    ]
    colors = custom_colors[:len(values)]

    wedges, _ = ax.pie(
        values,
        startangle=90,
        radius=radius,
        explode=explode,
        colors=colors
    )

    # 添加引导线和外部标签
    total = np.sum(values)

    for i, (wedge, label, value) in enumerate(zip(wedges, labels, values)):
        theta = 0.5 * (wedge.theta1 + wedge.theta2)
        rad = np.deg2rad(theta)
        
        # wedge.center 是圆心，wedge.r 是半径
        x0 = wedge.r * np.cos(rad)
        y0 = wedge.r * np.sin(rad)

        # 引导线：从弧边中心，先斜线再水平线
        offset = 0.15
        x1 = (radius + offset) * np.cos(rad)   
        y1 = (radius + offset) * np.sin(rad)

        horiz_offset = 0.3 if x1 >= 0 else -0.3
        x2 = x1 + horiz_offset
        y2 = y1

        # 绘制引导线
        ax.plot([x0, x1], [y0, y1], color=gray_color, linewidth=1)
        ax.plot([x1, x2], [y1, y2], color=gray_color, linewidth=1)

        # 标签内容：无小数
        if not paras:
            label_text = f"{label}\n {value} {int(round(value / total * 100))}%"
        else:
            rate = paras.get(label, 0)
            label_text = f"{label}, {value}项\n入围率 {rate}%"
        ha = 'left' if x2 >= 0 else 'right'
        ax.text(x2, y2, label_text, ha=ha, va='center', fontproperties=font, color=gray_color)

    plt.axis('equal')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight')
    plt.close()

def plot_pic7(df, save_path=None):
    # 1. 数据
    values, labels = df['values'], df['name']
    colors = ["#2668ae", '#f28e2b', '#edc948']  # 蓝 / 橙 / 黄

    # 2. 字体（确保系统安装了微软雅黑；否则改成 SimHei 或其他中文字体）
    title_font = FontProperties(fname=font_path, size=10)
    label_font = FontProperties(fname=font_path, size=8)
    
    # 3. 画布
    fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
    # 4. 绘制饼图
    def show_absolute_value(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return f'{val}项'
    wedges, texts, autotexts = ax.pie(
        x=values,
        labels=None,                    # 不在扇区上显示中文标签，只在图例里显示
        autopct=show_absolute_value,    # 直接显示整数
        colors=colors,
        startangle=90,
        wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'},
        textprops={'fontsize': 8, 'color': 'black'}
    )

    # 设置中文字体（关键！）
    for autotext in autotexts:
        autotext.set_fontproperties(label_font)

    # 5. 统一比例、标题
    ax.set_aspect('equal')         # 保证是圆形
    ax.set_title('AI评审误差分布', fontproperties=title_font)
    # 6. 图例放在上方居中
    ax.legend(
        wedges, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.05),
        ncol=3,
        frameon=False,
        prop=label_font,
        handlelength=1,  # 调整图例标记的长度
        columnspacing=1.0  # 调整列之间的间距
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight',transparent=True)
    plt.close()

def plot_bar(df, save_path=None):
    # 排序顺利：关键词、数值、名称
    priority_keywords = ["部","办","工会","调控中心"]

    def is_priority(name):
        return any(keyword in name for keyword in priority_keywords)
    df["_is_priority"] = df['name'].apply(lambda name:0 if is_priority(name) else 1)

    df = df.sort_values(by=["values","_is_priority","name"],ascending=[False,True,True]).drop(columns=["_is_priority"]).reset_index(drop=True)

    bar_count = len(df)
    # 动态设置图宽和柱宽
    
    fig_width = min(max(3, 0.3 * bar_count), 5.5) # 最小3，最大5.5
    fig_height = 3

    font_size = min(10, 20 - bar_count // 10)

    font = FontProperties(fname=font_path, size=font_size, weight='bold')
    gray_color = '#444444'
    bar_color = (24/255., 96/255., 90/255.)

    x = np.linspace(0, 1, bar_count)
    bar_spacing = x[1] - x[0] if bar_count > 1 else 1
    bar_width = 0.6 * bar_spacing

    # 创建图形
    plt.figure(figsize=(fig_width, fig_height), dpi=500)
    bars = plt.bar(x, df["values"], color=bar_color, width=bar_width)

    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        offset = max(0.01 * height, 0.2)
        plt.text(bar.get_x() + bar.get_width() / 2, height + offset,
                 f"{int(height)}", ha='center', va='bottom',
                 fontproperties=font, color=gray_color)

    # 设置 x 轴刻度及标签位置与柱子居中对齐
    if bar_count < 5:
        plt.xticks(ticks=x, labels=df["name"], rotation=0, ha='center',
                   fontproperties=font, color=gray_color)
    else:
        vectical_labels = ['\n'.join(str(label)) for label in df["name"]]
        plt.xticks(ticks=x, labels=vectical_labels, rotation=0, ha='center',
                fontproperties=font, color=gray_color)

    # 去边框美化
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('#666666')
    ax.spines['bottom'].set_linewidth(2.5)

    ax.yaxis.set_ticklabels([])
    ax.yaxis.set_ticks_position('none')

    # 柱子左右留白
    plt.margins(x=0.01)

    # 调整布局
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', transparent=True)
    plt.close()

if __name__ == '__main__':
    data = {
    "铜川公司":17,
    "西安公司":15,
    "宝鸡公司":11,
    "渭南公司":5,
    "汉中公司":10,
    "西咸公司":9,
    "延安公司":8,
    "思极公司":1,
    # "榆林公司":17,
    # "营服公司":15,
    # "咸阳公司":11,
    # "物资公司":5,
    # "信通公司":10,
    # "安康公司":9,
    # "安康水电厂":8,
    # "商洛公司":1
}

    df = pd.DataFrame(data.items(), columns=['name', 'values'])
#     df['values'] = df['values'].astype(int)  # 转换为整数
#     df = df.sort_values(by='values', ascending=False)
#     plot_bar(df,save_path='./input/ReportTemplate/pic/test.png')

    # 数据准备
    plot_pie(df,save_path='./input/ReportTemplate/pic/test2.png')
