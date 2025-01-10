import json
import matplotlib.pyplot as plt

# 定义一个函数，计算分类统计
def calculate_statistics(base_data, model_data=None):
    correct = 0
    wrong = 0
    refuse = 0
    over_refuse = 0

    for i, base_entry in enumerate(base_data):
        gt = base_entry['gt']
        base_output = base_entry['output']

        if model_data:  # 如果是对比 base 和其他模型
            model_entry = model_data[i]
            output = model_entry['output']
            confidence = model_entry['confidence']
            surety = model_entry['surety']

            # 判断 overrefuse: base 答对了，当前模型拒绝了
            if base_output == gt and output == "E":
                over_refuse += 1
            elif output == "E":  # 拒绝
                refuse += 1
            elif output == gt:  # 正确
                correct += 1
            else:  # 错误
                wrong += 1
        else:  # 如果只计算 base
            if base_output == gt:  # 正确
                correct += 1
            else:  # 错误
                wrong += 1

    total = correct + wrong + refuse + over_refuse
    return {
        "correct": correct / total,
        "wrong": wrong / total,
        "refuse": refuse / total,
        "over_refuse": over_refuse / total,
    }

# 定义一个函数，绘制统计图
def plot_statistics(statistics_list, labels):
    categories = ['correct', 'refuse', 'over_refuse', 'wrong']
    colors = ['green', 'pink', 'blue', 'red']
    bar_width = 0.5

    x = range(len(statistics_list))  # x轴位置
    for i, category in enumerate(categories):
        values = [stat[category] for stat in statistics_list]
        bottom = [sum(stat[categories[j]] for j in range(i)) for stat in statistics_list]
        plt.bar(x, values, bottom=bottom, color=colors[i], label=category, width=bar_width)

    plt.xticks(ticks=x, labels=labels)
    plt.ylabel('Percentage')
    plt.legend()
    plt.title("Classification Statistics")
    plt.savefig('res/statistics.png')

# 主程序部分，读取文件并生成图表
file_paths = {
    'base': 'res/MMLU_ID_base_raw.json',
    'naive': 'res/MMLU_ID_naive_raw.json',
    'craft160': 'res/MMLU_ID_craft160_raw.json',
    'craft': 'res/MMLU_ID_craft_raw.json',
}
labels = ['base', 'naive', 'craft160', 'craft']
statistics_list = []

# 加载 base 数据
try:
    with open(file_paths['base'], 'r') as f:
        base_data = json.load(f)
except FileNotFoundError:
    print(f"Base file {file_paths['base']} not found.")
    base_data = []

# 计算 base 的统计信息
statistics_list.append(calculate_statistics(base_data))

# 计算其他模型的统计信息
for label in labels[1:]:  # 跳过 base
    try:
        with open(file_paths[label], 'r') as f:
            model_data = json.load(f)
        statistics = calculate_statistics(base_data, model_data)
        statistics_list.append(statistics)
    except FileNotFoundError:
        print(f"File {file_paths[label]} not found.")
        statistics_list.append({
            "correct": 0,
            "wrong": 0,
            "refuse": 0,
            "over_refuse": 0,
        })

# 绘制图表
plot_statistics(statistics_list, labels)
