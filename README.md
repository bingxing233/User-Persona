# 用户画像分析系统

这是一个完整的用户画像分析系统，包含数据预处理、特征工程、机器学习建模、可视化分析和Web界面展示等功能。

## 功能模块

### 1. 数据预处理与特征工程 (data_preprocessing.py)
- 数据清洗与异常值检测
- 缺失值处理
- 特征编码（独热编码、标签编码）
- 特征构建（年龄分组、平均购买金额等）

### 2. 机器学习建模 (ml_modeling.py)
- K-Means聚类分析
- 分类模型（逻辑回归、决策树）
- 模型评估与优化
- SHAP模型解释

### 3. 可视化分析 (visualization.py)
- 用户基本特征分布可视化
- 购买行为特征分析
- 聚类结果可视化
- 相关性分析
- t-SNE用户画像可视化

### 4. Web界面展示 (app.py)
- 基于Streamlit的交互式Web应用
- 多维度数据分析展示
- 实时预测功能

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行方式

### 方法一：一键运行所有步骤
```bash
python main.py
```

### 方法二：分步运行
1. 数据预处理：
   ```bash
   python data_preprocessing.py
   ```

2. 机器学习建模：
   ```bash
   python ml_modeling.py
   ```

3. 可视化分析：
   ```bash
   python visualization.py
   ```

4. 启动Web应用：
   ```bash
   streamlit run app.py
   ```

## 文件说明

- `shopping_trends.csv`: 原始数据文件
- `processed_data.csv`: 预处理后的数据
- `encoded_data.csv`: 编码后的数据
- `clustered_data.csv`: 聚类分析后的数据
- `*.pkl`: 保存的机器学习模型文件
- `requirements.txt`: 项目依赖包列表

## 系统要求

- Python 3.7+
- 建议使用虚拟环境运行

## 注意事项

1. 首次运行可能需要较长时间，特别是在机器学习建模和可视化部分
2. SHAP解释部分可能需要较多计算资源
3. 如需重新训练模型，请删除旧的.pkl文件