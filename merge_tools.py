import yaml
import subprocess
import os

# 定义需要替换的数值列表
values_to_replace = [80, 160, 240, 320, 400]

# 原始的 YAML 文件路径
yaml_file_path = '/home/xuzhiyu/LLaMA-Factory/examples/custom/merge_baichuan_lora_craft.yaml'

# 读取原始的 YAML 文件
with open(yaml_file_path, 'r') as file:
    data = yaml.safe_load(file)

target_dir = '/home/xuzhiyu/LLaMA-Factory/'
os.chdir(target_dir)
print(f"Changed working directory to {target_dir}")

# 循环处理每个 export_size
for value in values_to_replace:
    # 替换文件中的 export_size 相关路径
    new_model_path = data['adapter_name_or_path'].replace('-160', f'-{value}')
    new_export_dir = data['export_dir'].replace('-160', f'-{value}')
    
    new_data = data.copy()
    # 更新 YAML 数据
    new_data['adapter_name_or_path'] = new_model_path
    new_data['export_dir'] = new_export_dir
    new_data['export_size'] = 5  # 你可以根据需要修改 export_size 的值

    # 新的 YAML 文件路径
    new_yaml_path = yaml_file_path.replace('.yaml', f'_{value}.yaml')

    # 写入更新后的 YAML 文件
    with open(new_yaml_path, 'w') as new_file:
        yaml.dump(new_data, new_file, default_flow_style=False)
    
    # 执行 llamafactory-cli export 命令
    command = [
        'llamafactory-cli', 
        'export', 
        new_yaml_path
    ]
    
    # 执行命令并打印结果
    try:
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=True)
        print(f"Export completed for export_size {value}.")
    except subprocess.CalledProcessError as e:
        print(f"Error during export for export_size {value}: {e}")

print("All exports completed.")
