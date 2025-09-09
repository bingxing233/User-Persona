import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# 设置页面配置
st.set_page_config(
    page_title="用户画像分析系统",
    page_icon="👥",
    layout="wide"
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

@st.cache_data
def load_data():
    """加载数据"""
    df = pd.read_csv('processed_data.csv')
    return df

@st.cache_data
def load_encoded_data():
    """加载编码后的数据"""
    df = pd.read_csv('encoded_data.csv')
    return df

@st.cache_data
def load_clustered_data():
    """加载聚类后的数据"""
    df = pd.read_csv('clustered_data.csv')
    return df

def main():
    st.title("👥 用户画像分析系统")
    st.markdown("---")
    
    # 侧边栏
    st.sidebar.title("导航")
    page = st.sidebar.selectbox(
        "选择分析模块",
        ["数据概览", "用户基本特征", "购买行为分析", "聚类分析", "分类预测", "相关性分析", "用户画像(t-SNE)"]
    )
    
    # 加载数据
    try:
        df = load_data()
        df_encoded = load_encoded_data()
        df_clustered = load_clustered_data()
    except FileNotFoundError:
        st.error("数据文件未找到，请先运行数据预处理脚本")
        return
    
    if page == "数据概览":
        show_data_overview(df)
    elif page == "用户基本特征":
        show_user_demographics(df)
    elif page == "购买行为分析":
        show_purchase_behavior(df)
    elif page == "聚类分析":
        show_clustering_analysis(df_clustered)
    elif page == "分类预测":
        show_classification_prediction(df)
    elif page == "相关性分析":
        show_correlation_analysis(df)
    elif page == "用户画像(t-SNE)":
        st.header("用户画像 (t-SNE)")
        visualize_user_portrait_tsne(df_clustered)

def show_data_overview(df):
    """显示数据概览"""
    st.header("数据概览")
    
    # 数据基本信息
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总记录数", len(df))
    with col2:
        st.metric("特征数量", len(df.columns))
    with col3:
        st.metric("缺失值", df.isnull().sum().sum())
    
    # 描述统计表格
    st.subheader("描述统计")
    st.dataframe(df.describe())
    
    # 特征工程展示
    st.subheader("特征工程")
    st.markdown("""
    在数据预处理阶段，我们进行了以下特征工程：
    
    1. **年龄组划分**: 根据年龄范围将用户划分为6个年龄段组
    2. **平均购买金额计算**: 通过购买金额与以往购买次数计算得出
    3. **独热编码**: 对性别、类别、地区等分类变量进行编码
    4. **标签编码**: 对有序分类变量进行编码
    """)
    
    # 显示前几行数据
    st.subheader("数据样本")
    st.dataframe(df.head())
    
    # 数据字段说明
    st.subheader("字段说明")
    field_descriptions = {
        "Customer ID": "客户ID",
        "Age": "年龄",
        "Gender": "性别",
        "Item Purchased": "购买商品",
        "Category": "商品类别",
        "Purchase Amount (USD)": "购买金额(美元)",
        "Location": "位置",
        "Size": "尺寸",
        "Color": "颜色",
        "Season": "季节",
        "Review Rating": "评价评分",
        "Subscription Status": "订阅状态",
        "Payment Method": "支付方式",
        "Shipping Type": "配送类型",
        "Discount Applied": "是否应用折扣",
        "Promo Code Used": "是否使用促销码",
        "Previous Purchases": "以往购买次数",
        "Preferred Payment Method": "首选支付方式",
        "Frequency of Purchases": "购买频率",
        "Age Group": "年龄组",
        "Average Purchase Amount": "平均购买金额"
    }
    
    for field, description in field_descriptions.items():
        if field in df.columns:
            st.write(f"**{field}**: {description}")

def show_user_demographics(df):
    """显示用户基本特征"""
    st.header("用户基本特征分析")
    
    # 创建多列布局展示关键指标
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("平均年龄", f"{df['Age'].mean():.1f}岁")
    with col2:
        st.metric("年龄中位数", f"{df['Age'].median():.0f}岁")
    with col3:
        st.metric("年龄标准差", f"{df['Age'].std():.1f}岁")
    
    # 年龄分布
    st.subheader("年龄分布")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 直方图
    axes[0].hist(df['Age'], bins=20, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('年龄')
    axes[0].set_ylabel('频次')
    axes[0].set_title('年龄分布直方图')
    
    # 箱线图
    axes[1].boxplot(df['Age'])
    axes[1].set_ylabel('年龄')
    axes[1].set_title('年龄分布箱线图')
    
    st.pyplot(fig)
    
    # 性别分布
    st.subheader("性别分布")
    col1, col2 = st.columns(2)
    
    with col1:
        gender_counts = df['Gender'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
        ax.set_title('性别分布')
        st.pyplot(fig)
    
    with col2:
        st.dataframe(gender_counts)
        st.write(f"男性占比: {gender_counts['Male'] / len(df) * 100:.1f}%")
        st.write(f"女性占比: {gender_counts['Female'] / len(df) * 100:.1f}%")
    
    # 地区分布
    st.subheader("地区分布")
    
    # Top 10 地区分布
    st.markdown("#### 前10个地区分布")
    location_counts = df['Location'].value_counts()
    top10_locations = location_counts.head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(top10_locations)), top10_locations.values, color='lightcoral')
    ax.set_yticks(range(len(top10_locations)))
    ax.set_yticklabels(top10_locations.index)
    ax.set_xlabel('客户数量')
    ax.set_title('前10个地区分布')
    ax.invert_yaxis()
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, f'{width}', 
                ha='left', va='center', fontweight='bold')
    
    st.pyplot(fig)
    
    # 所有地区分布统计
    st.markdown("#### 地区分布统计")
    st.dataframe(location_counts)
    
    # 年龄组分布
    st.subheader("年龄组分布")
    col1, col2 = st.columns(2)
    
    with col1:
        age_group_counts = df['Age Group'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(age_group_counts.index, age_group_counts.values, color='gold', edgecolor='black')
        ax.set_xlabel('年龄组')
        ax.set_ylabel('客户数量')
        ax.set_title('年龄组分布')
        plt.setp(ax.get_xticklabels(), rotation=45)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height, f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig)
    
    with col2:
        st.dataframe(age_group_counts)
        st.write(f"主要年龄组: {age_group_counts.idxmax()}")
        st.write(f"该年龄组占比: {age_group_counts.max() / len(df) * 100:.1f}%")
    
    # 商品类别分布
    st.subheader("商品类别分布")
    category_counts = df['Category'].value_counts()
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', startangle=90)
        ax.set_title('商品类别分布')
        st.pyplot(fig)
        
    with col2:
        st.dataframe(category_counts)
        st.write(f"最受欢迎的类别: {category_counts.idxmax()}")
        st.write(f"该类别占比: {category_counts.max() / len(df) * 100:.1f}%")
    
    # 颜色偏好分布
    st.subheader("颜色偏好分布")
    color_counts = df['Color'].value_counts().head(10)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(color_counts.index, color_counts.values, color='lightgreen')
    ax.set_xlabel('颜色')
    ax.set_ylabel('客户数量')
    ax.set_title('前10种颜色偏好分布')
    plt.setp(ax.get_xticklabels(), rotation=45)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    st.pyplot(fig)
    
    # 尺寸偏好分布
    st.subheader("尺寸偏好分布")
    size_counts = df['Size'].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(size_counts.index, size_counts.values, color='lightblue')
    ax.set_xlabel('尺寸')
    ax.set_ylabel('客户数量')
    ax.set_title('尺寸偏好分布')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    st.pyplot(fig)
    
    # 其他特征分布
    st.subheader("其他特征分布")
    
    # 支付方式分布
    st.markdown("#### 支付方式分布")
    payment_counts = df['Payment Method'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(payment_counts.index, payment_counts.values, color='lightgreen')
    ax.set_xlabel('支付方式')
    ax.set_ylabel('客户数量')
    ax.set_title('支付方式分布')
    plt.setp(ax.get_xticklabels(), rotation=45)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    st.pyplot(fig)
    
    # 配送类型分布
    st.markdown("#### 配送类型分布")
    shipping_counts = df['Shipping Type'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(shipping_counts.index, shipping_counts.values, color='lightblue')
    ax.set_xlabel('配送类型')
    ax.set_ylabel('客户数量')
    ax.set_title('配送类型分布')
    plt.setp(ax.get_xticklabels(), rotation=45)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    st.pyplot(fig)

def show_purchase_behavior(df):
    """显示购买行为分析"""
    st.header("购买行为分析")
    
    # 创建关键指标展示
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("平均购买金额", f"${df['Purchase Amount (USD)'].mean():.2f}")
    with col2:
        st.metric("购买金额中位数", f"${df['Purchase Amount (USD)'].median():.2f}")
    with col3:
        st.metric("平均评价评分", f"{df['Review Rating'].mean():.2f}")
    with col4:
        st.metric("平均以往购买次数", f"{df['Previous Purchases'].mean():.1f}")
    
    # 购买金额分布
    st.subheader("购买金额分布")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df['Purchase Amount (USD)'], bins=30, color='lightcoral', edgecolor='black')
        ax.set_xlabel('购买金额 (USD)')
        ax.set_ylabel('频次')
        ax.set_title('购买金额分布直方图')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot(df['Purchase Amount (USD)'])
        ax.set_ylabel('购买金额 (USD)')
        ax.set_title('购买金额分布箱线图')
        st.pyplot(fig)
    
    # 不同季节的购买行为分析
    st.subheader("不同季节的购买行为分析")
    
    # 季节购买金额分析
    season_purchase = df.groupby('Season')['Purchase Amount (USD)'].agg(['mean', 'count']).reset_index()
    season_purchase.columns = ['Season', '平均购买金额', '购买次数']
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(season_purchase['Season'], season_purchase['平均购买金额'], marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('季节')
        ax.set_ylabel('平均购买金额 (USD)')
        ax.set_title('不同季节的平均购买金额')
        ax.grid(True)
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(season_purchase['Season'], season_purchase['购买次数'], color='lightcoral')
        ax.set_xlabel('季节')
        ax.set_ylabel('购买次数')
        ax.set_title('不同季节的购买次数')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height, f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig)
    
    st.dataframe(season_purchase.style.format({"平均购买金额": "${:.2f}", "购买次数": "{:.0f}"}))
    
    # 不同类别的购买行为分析
    st.subheader("不同类别的购买行为分析")
    
    # 类别购买金额分布
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        df.boxplot(column='Purchase Amount (USD)', by='Category', ax=ax)
        ax.set_xlabel('商品类别')
        ax.set_ylabel('购买金额 (USD)')
        ax.set_title('不同类别的购买金额分布箱线图')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        category_stats = df.groupby('Category')['Purchase Amount (USD)'].agg(['mean', 'count']).reset_index()
        category_stats.columns = ['Category', '平均购买金额', '购买次数']
        category_stats = category_stats.sort_values('平均购买金额', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(category_stats['Category'], category_stats['平均购买金额'], color='lightblue')
        ax.set_xlabel('商品类别')
        ax.set_ylabel('平均购买金额 (USD)')
        ax.set_title('不同类别的平均购买金额')
        plt.setp(ax.get_xticklabels(), rotation=45)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height, f'${height:.0f}',
                    ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig)
    
    st.dataframe(category_stats.style.format({"平均购买金额": "${:.2f}", "购买次数": "{:.0f}"}))
    
    # 购买频率分布
    st.subheader("购买频率分析")
    
    freq_counts = df['Frequency of Purchases'].value_counts()
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(freq_counts.index, freq_counts.values, color='mediumseagreen', edgecolor='black')
        ax.set_xlabel('购买频率')
        ax.set_ylabel('客户数量')
        ax.set_title('购买频率分布')
        plt.setp(ax.get_xticklabels(), rotation=45)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height, f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig)
    
    with col2:
        st.dataframe(freq_counts)
        st.write(f"最常见的购买频率: {freq_counts.idxmax()}")
        st.write(f"该频率占比: {freq_counts.max() / len(df) * 100:.1f}%")
    
    # 评价评分分布
    st.subheader("评价评分分析")
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df['Review Rating'], bins=20, color='orange', edgecolor='black')
        ax.set_xlabel('评价评分')
        ax.set_ylabel('频次')
        ax.set_title('评价评分分布')
        st.pyplot(fig)
    
    with col2:
        rating_stats = df['Review Rating'].describe()
        st.dataframe(rating_stats)
        st.write(f"平均评分: {df['Review Rating'].mean():.2f}")
        st.write(f"评分标准差: {df['Review Rating'].std():.2f}")
    
    # 折扣和促销分析
    st.subheader("折扣和促销分析")
    
    col1, col2 = st.columns(2)
    with col1:
        discount_counts = df['Discount Applied'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(discount_counts.values, labels=discount_counts.index, autopct='%1.1f%%', startangle=90)
        ax.set_title('折扣应用情况')
        st.pyplot(fig)
        
        st.dataframe(discount_counts)
    
    with col2:
        promo_counts = df['Promo Code Used'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(promo_counts.values, labels=promo_counts.index, autopct='%1.1f%%', startangle=90)
        ax.set_title('促销码使用情况')
        st.pyplot(fig)
        
        st.dataframe(promo_counts)
    
    # 购买金额与评价评分的关系
    st.subheader("购买金额与评价评分关系")
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df['Purchase Amount (USD)'], df['Review Rating'], alpha=0.6)
    ax.set_xlabel('购买金额 (USD)')
    ax.set_ylabel('评价评分')
    ax.set_title('购买金额与评价评分关系散点图')
    plt.grid(True)
    st.pyplot(fig)
    
    # 相关性分析
    correlation = df[['Purchase Amount (USD)', 'Review Rating', 'Previous Purchases', 'Age']].corr()
    
    # 订阅状态分析
    st.subheader("订阅状态分析")
    subscription_counts = df['Subscription Status'].value_counts()
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(subscription_counts.values, labels=subscription_counts.index, autopct='%1.1f%%', startangle=90)
        ax.set_title('订阅状态分布')
        st.pyplot(fig)
    
    with col2:
        st.dataframe(subscription_counts)
        st.write(f"订阅用户占比: {subscription_counts['Yes'] / len(df) * 100:.1f}%")
        st.write(f"非订阅用户占比: {subscription_counts['No'] / len(df) * 100:.1f}%")

def show_clustering_analysis(df_clustered):
    """显示聚类分析结果"""
    st.header("聚类分析")
    
    # 加载聚类模型和标准化器
    try:
        kmeans_model = joblib.load('kmeans_model.pkl')
        scaler = joblib.load('scaler.pkl')
        
        # 从模型中获取最优聚类数
        optimal_k = kmeans_model.n_clusters
    except FileNotFoundError:
        st.warning("聚类模型文件未找到，请先运行 ml_modeling.py 脚本")
        return
    except Exception as e:
        st.error(f"加载聚类模型时出错: {str(e)}")
        return
    
    # 显示最优聚类数
    st.subheader("最优聚类数")
    st.success(f"系统自动选择的最优聚类数: {optimal_k}")
    
    # 显示手肘法和评估指标图
    st.subheader("聚类数选择 - 手肘法和评估指标")
    
    # 加载聚类评估图表
    try:
        # 显示聚类评估图表
        st.image('clustering_evaluation.png', caption='聚类评估指标', width='stretch')
    except FileNotFoundError:
        st.info("聚类评估图表未找到，请运行 ml_modeling.py 脚本生成图表")
    
    # 聚类数量选择
    n_clusters = st.slider("选择聚类数量", 2, 10, optimal_k)
    
    # 显示聚类结果
    st.subheader(f"聚类结果 (k={n_clusters})")
    
    # 年龄 vs 购买金额
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df_clustered['Age'], df_clustered['Purchase Amount (USD)'], 
                        c=df_clustered['Cluster'], cmap='viridis', alpha=0.6)
    ax.set_xlabel('年龄')
    ax.set_ylabel('购买金额 (USD)')
    ax.set_title('客户聚类: 年龄 vs 购买金额')
    plt.colorbar(scatter)
    st.pyplot(fig)
    
    # 评价评分 vs 以往购买次数
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df_clustered['Review Rating'], df_clustered['Previous Purchases'], 
                        c=df_clustered['Cluster'], cmap='viridis', alpha=0.6)
    ax.set_xlabel('评价评分')
    ax.set_ylabel('以往购买次数')
    ax.set_title('客户聚类: 评价评分 vs 以往购买次数')
    plt.colorbar(scatter)
    st.pyplot(fig)
    
    # 聚类分布
    st.subheader("聚类分布")
    cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(cluster_counts.index, cluster_counts.values, color='purple', edgecolor='black')
    ax.set_xlabel('聚类')
    ax.set_ylabel('客户数量')
    ax.set_title('各聚类的客户分布')
    ax.set_xticks(cluster_counts.index)
    st.pyplot(fig)
    
    # 各聚类的特征统计
    st.subheader("各聚类特征统计")
    cluster_stats = df_clustered.groupby('Cluster').agg({
        'Age': 'mean',
        'Purchase Amount (USD)': 'mean',
        'Review Rating': 'mean',
        'Previous Purchases': 'mean'
    }).round(2)
    st.dataframe(cluster_stats)
    
    # 添加聚类分析说明
    st.markdown("""
    **聚类分析说明**:
    - 不同颜色代表不同的客户群体
    - 聚类结果可以帮助识别具有相似特征的客户群
    - 可以根据不同客户群体制定个性化的营销策略
    """)

def show_classification_prediction(df):
    """显示分类预测"""
    st.header("分类预测")
    
    st.subheader("模型说明")
    st.markdown("""
    本系统采用多种分类算法进行预测：
    
    1. **逻辑回归**: 线性分类器，适用于特征间线性关系明显的情况
    2. **决策树**: 非线性分类器，能处理复杂的特征关系，具有良好的可解释性
    3. **随机森林**: 集成学习算法，通过组合多个决策树提高准确性和稳定性
    4. **支持向量机**: 能处理复杂的非线性关系，适用于中等规模数据集
    """)
    
    try:
        # 加载模型、编码器和特征信息
        lr_model = joblib.load('logistic_regression_model.pkl')
        dt_model = joblib.load('decision_tree_model.pkl')
        rf_model = joblib.load('random_forest_model.pkl')
        svm_model = joblib.load('svm_model.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        feature_scaler = joblib.load('feature_scaler.pkl')
        feature_info = joblib.load('feature_info.pkl')
        
        feature_columns = feature_info['feature_columns']
        numeric_features = feature_info['numeric_features']
        categorical_features = feature_info['categorical_features']
        
        st.success("分类模型已加载")
        
        # 显示模型评估结果
        st.subheader("模型评估结果")
        try:
            # 尝试加载模型评估结果
            model_evaluation = joblib.load('model_evaluation.pkl')
            metrics_df = pd.DataFrame(model_evaluation).T
            
            st.dataframe(metrics_df.style.format("{:.4f}"))
            
            # 可视化模型评估结果
            fig, ax = plt.subplots(figsize=(12, 6))
            metrics_df[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']].plot(kind='bar', ax=ax)
            plt.title('模型评估指标比较')
            plt.ylabel('Score')
            plt.xticks(rotation=45)
            plt.legend(loc='lower right')
            plt.tight_layout()
            st.pyplot(fig)
            
        except FileNotFoundError:
            st.info("模型评估结果文件未找到，请运行 ml_modeling.py 脚本生成完整评估报告")
        
        # 显示特征重要性
        st.subheader("特征重要性分析")
        try:
            # 尝试加载特征重要性
            feature_importance = joblib.load('feature_importance.pkl')
            
            # 可视化特征重要性
            fig, ax = plt.subplots(figsize=(10, 6))
            top_features = feature_importance.head(10)
            ax.barh(range(len(top_features)), top_features['importance'])
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'])
            ax.set_xlabel('Importance')
            ax.set_title('Top 10 Feature Importance')
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
            
            st.dataframe(feature_importance.head(10).style.format({"importance": "{:.4f}"}))
            
        except FileNotFoundError:
            st.info("特征重要性文件未找到，请运行 ml_modeling.py 脚本生成分析报告")
        
        # 显示SHAP解释
        st.subheader("SHAP模型解释")
        st.markdown("""
        SHAP (SHapley Additive exPlanations) 是一种用于解释机器学习模型预测结果的方法。
        它可以帮助我们理解每个特征对预测结果的贡献程度。
        """)
        
        try:
            # 尝试加载SHAP解释结果
            shap_explanation = joblib.load('shap_explanation.pkl')
            st.info("SHAP解释结果已生成，请运行 ml_modeling.py 查看详细可视化图表")
            
            # 显示SHAP特征重要性图
            try:
                st.image('shap_summary.png', caption='SHAP特征重要性', width='stretch')
                st.image('shap_beeswarm.png', caption='SHAP蜜蜂图', width='stretch')
            except FileNotFoundError:
                st.info("SHAP可视化图表未找到，请运行 ml_modeling.py 脚本生成图表")
                
        except FileNotFoundError:
            st.info("SHAP解释结果文件未找到，请运行 ml_modeling.py 脚本生成解释报告")
        
        st.write("模型特征列:", feature_columns)
        
        st.subheader("输入特征进行预测")
        
        # 创建输入控件，按照特征类型分组
        st.markdown("#### 数值型特征")
        numeric_inputs = {}
        cols = st.columns(min(len(numeric_features), 5))
        for i, feature in enumerate(numeric_features):
            with cols[i % 5]:
                if feature == 'Age':
                    numeric_inputs[feature] = st.number_input(feature, min_value=18, max_value=100, value=30)
                elif feature in ['Purchase Amount (USD)', 'Average Purchase Amount']:
                    numeric_inputs[feature] = st.number_input(feature, min_value=0.0, value=50.0, step=1.0)
                elif feature == 'Review Rating':
                    numeric_inputs[feature] = st.slider(feature, 0.0, 5.0, 3.0, 0.1)
                elif feature == 'Previous Purchases':
                    numeric_inputs[feature] = st.number_input(feature, min_value=0, value=5)
                else:
                    numeric_inputs[feature] = st.number_input(feature, value=0)
        
        st.markdown("#### 分类型特征")
        categorical_inputs = {}
        cols = st.columns(min(len(categorical_features), 5))
        for i, feature in enumerate(categorical_features):
            with cols[i % 5]:
                # 获取该特征的所有唯一值
                if feature in df.columns:
                    unique_values = df[feature].unique().tolist()
                    categorical_inputs[feature] = st.selectbox(feature, unique_values)
                else:
                    categorical_inputs[feature] = st.text_input(feature, "Unknown")
        
        # 选择模型
        model_options = {
            "逻辑回归": lr_model,
            "决策树": dt_model,
            "随机森林": rf_model,
            "支持向量机": svm_model
        }
        
        selected_model_name = st.selectbox("选择预测模型", list(model_options.keys()))
        selected_model = model_options[selected_model_name]
        
        # 进行预测
        if st.button("预测订阅状态"):
            # 构造输入特征
            input_data = pd.DataFrame([{**numeric_inputs, **categorical_inputs}])
            
            # 确保列的顺序与训练时一致
            input_data = input_data[feature_columns]
            
            # 对文本特征进行编码
            input_encoded = input_data.copy()
            for col, encoder in label_encoders.items():
                if col in input_encoded.columns:
                    try:
                        input_encoded[col] = encoder.transform(input_encoded[col].astype(str))
                    except ValueError:
                        # 如果遇到未见过的标签，使用默认值
                        input_encoded[col] = 0
            
            # 确保列的顺序正确
            input_encoded = input_encoded[feature_columns]
            
            # 特征缩放
            input_scaled = feature_scaler.transform(input_encoded)
            
            try:
                # 预测
                prediction = selected_model.predict(input_scaled)
                probability = selected_model.predict_proba(input_scaled)
                
                st.subheader("预测结果")
                st.write(f"预测订阅状态: {prediction[0]}")
                st.write(f"预测概率:")
                st.write(f"- No: {probability[0][0]:.2%}")
                st.write(f"- Yes: {probability[0][1]:.2%}")
                
            except Exception as e:
                st.error(f"预测过程中出现错误: {str(e)}")
                st.write("请确保所有特征都已正确输入")
            
    except FileNotFoundError as e:
        st.warning(f"模型文件未找到，请先运行 ml_modeling.py 脚本训练模型: {str(e)}")
    except Exception as e:
        st.error(f"加载模型过程中出现错误: {str(e)}")

def show_correlation_analysis(df):
    """显示相关性分析"""
    st.header("相关性分析")
    
    # 选择数值型特征
    
    # 选择数值型特征
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 移除不需要的列
    exclude_columns = ['Customer ID']
    if 'Cluster' in numeric_features:
        exclude_columns.append('Cluster')
    numeric_features = [col for col in numeric_features if col not in exclude_columns]
    
    # 计算相关性矩阵
    corr_matrix = df[numeric_features].corr()
    
    # 绘制热力图
    st.subheader("特征相关性热力图")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax)
    ax.set_title('数值特征相关性矩阵')
    st.pyplot(fig)
    
    # 显示高相关性特征对
    st.subheader("高相关性特征对")
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > 0.5:  # 相关性绝对值大于0.5认为是高相关
                high_corr_pairs.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_value
                })
    
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', key=abs, ascending=False)
        st.dataframe(high_corr_df)
    else:
        st.info("没有发现相关性绝对值大于0.5的特征对")

def visualize_user_portrait_tsne(df):
    """
    使用t-SNE可视化用户画像
    """
    try:
        from sklearn.manifold import TSNE
        
        # 选择用于t-SNE的特征
        features_for_tsne = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']
        if 'Cluster' in df.columns:
            features_for_tsne.append('Cluster')
        
        X_tsne = df[features_for_tsne]
        
        # 处理缺失值
        X_tsne = X_tsne.fillna(X_tsne.mean())
        
        # 应用t-SNE降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_tsne)//3))
        X_tsne_reduced = tsne.fit_transform(X_tsne)
        
        # 可视化
        fig, ax = plt.subplots(figsize=(12, 8))
        if 'Cluster' in df.columns:
            scatter = ax.scatter(X_tsne_reduced[:, 0], X_tsne_reduced[:, 1], 
                               c=df['Cluster'], cmap='tab10', alpha=0.7)
            plt.colorbar(scatter)
        else:
            ax.scatter(X_tsne_reduced[:, 0], X_tsne_reduced[:, 1], alpha=0.7)
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"t-SNE可视化过程中出现错误: {str(e)}")

if __name__ == "__main__":
    main()