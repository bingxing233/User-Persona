import subprocess
import sys
import os

def run_step(script_name, description):
    """
    运行指定的Python脚本
    """
    print(f"\n{'='*50}")
    print(f"正在运行: {description}")
    print(f"脚本: {script_name}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✓ {description} 完成")
            if result.stdout:
                print("输出:")
                print(result.stdout)
        else:
            print(f"✗ {description} 失败")
            if result.stderr:
                print("错误信息:")
                print(result.stderr)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"✗ {description} 超时")
        return False
    except Exception as e:
        print(f"✗ {description} 出现异常: {str(e)}")
        return False

def main():
    """
    主函数，按顺序执行所有步骤
    """
    print("用户画像分析系统 - 全流程执行")
    print("注意: 首次运行可能需要较长时间，特别是机器学习和可视化部分")
    
    # 检查数据文件是否存在
    if not os.path.exists('shopping_trends.csv'):
        print("错误: 未找到 shopping_trends.csv 数据文件")
        return
    
    # 步骤1: 数据预处理和特征工程
    if not run_step('data_preprocessing.py', '数据预处理与特征工程'):
        print("数据预处理失败，停止执行")
        return
    
    # 步骤2: 机器学习建模
    if not run_step('ml_modeling.py', '机器学习建模'):
        print("机器学习建模失败，停止执行")
        return
    
    # 步骤3: 可视化分析
    if not run_step('visualization.py', '可视化分析'):
        print("可视化分析失败，停止执行")
        return
    
    # 步骤4: 启动Streamlit应用
    print(f"\n{'='*50}")
    print("所有分析已完成!")
    print("启动Streamlit应用: streamlit run app.py")
    print(f"{'='*50}")
    
    try:
        subprocess.run(['streamlit', 'run', 'app.py'], check=True)
    except subprocess.CalledProcessError:
        print("启动Streamlit应用失败")
    except FileNotFoundError:
        print("未找到streamlit命令，请确保已安装streamlit")
        print("可以使用以下命令安装: pip install streamlit")

if __name__ == "__main__":
    main()