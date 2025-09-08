import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def visualize_user_demographics(df):
    """
    可视化用户基本特征分布
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Age分布柱状图
    axes[0, 0].hist(df['Age'], bins=20, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Age Distribution')
    
    
    # Gender比例饼图
    gender_counts = df['Gender'].value_counts()
    axes[0, 1].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title('Gender Distribution')
    
    
    # Location分布 (Top 10) - 按从大到小排序
    location_counts = df['Location'].value_counts().head(10)  # 显示前10个地区，已默认按数量排序
    axes[1, 0].barh(range(len(location_counts)), location_counts.values, color='lightcoral')
    axes[1, 0].set_yticks(range(len(location_counts)))
    axes[1, 0].set_yticklabels(location_counts.index)
    axes[1, 0].set_xlabel('Number of Customers')
    axes[1, 0].set_title('Top 10 Locations')
    
    
    # Age Group分布 - 按从小到大排列
    age_group_counts = df['Age Group'].value_counts().sort_index()
    axes[1, 1].bar(age_group_counts.index, age_group_counts.values, color='gold', edgecolor='black')
    axes[1, 1].set_xlabel('Age Group')
    axes[1, 1].set_ylabel('Number of Customers')
    axes[1, 1].set_title('Age Group Distribution')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    
    plt.tight_layout()
    plt.show()

def visualize_purchase_behavior(df):
    """
    可视化购买行为特征
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 不同Season的Purchase Amount变化趋势
    season_purchase = df.groupby('Season')['Purchase Amount (USD)'].mean()
    axes[0, 0].plot(season_purchase.index, season_purchase.values, marker='o', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Season')
    axes[0, 0].set_ylabel('Average Purchase Amount (USD)')
    axes[0, 0].set_title('Average Purchase Amount by Season')
    axes[0, 0].grid(True)
    
    
    # 不同Category购买金额差异箱线图
    category_purchase = [df[df['Category'] == cat]['Purchase Amount (USD)'] for cat in df['Category'].unique()]
    axes[0, 1].boxplot(category_purchase, labels=df['Category'].unique())
    axes[0, 1].set_xlabel('Category')
    axes[0, 1].set_ylabel('Purchase Amount (USD)')
    axes[0, 1].set_title('Purchase Amount Distribution by Category')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    
    # 不同Frequency of Purchases人数分布
    freq_counts = df['Frequency of Purchases'].value_counts()
    axes[1, 0].bar(freq_counts.index, freq_counts.values, color='mediumseagreen', edgecolor='black')
    axes[1, 0].set_xlabel('Frequency of Purchases')
    axes[1, 0].set_ylabel('Number of Customers')
    axes[1, 0].set_title('Customer Distribution by Purchase Frequency')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    
    # Review Rating分布
    axes[1, 1].hist(df['Review Rating'], bins=20, color='orange', edgecolor='black')
    axes[1, 1].set_xlabel('Review Rating')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Review Rating Distribution')
    
    
    plt.tight_layout()
    plt.show()

def visualize_model_results(df):
    """
    可视化模型结果
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 聚类结果散点图
    scatter = axes[0].scatter(df['Age'], df['Purchase Amount (USD)'], c=df['Cluster'], cmap='viridis', alpha=0.6)
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Purchase Amount (USD)')
    axes[0].set_title('Customer Clusters: Age vs Purchase Amount')
    plt.colorbar(scatter, ax=axes[0])
    
    # 聚类大小分布
    cluster_counts = df['Cluster'].value_counts().sort_index()
    axes[1].bar(cluster_counts.index, cluster_counts.values, color='purple', edgecolor='black')
    axes[1].set_xlabel('Cluster')
    axes[1].set_ylabel('Number of Customers')
    axes[1].set_title('Customer Distribution Across Clusters')
    axes[1].set_xticks(cluster_counts.index)
    
    plt.tight_layout()
    plt.show()

def visualize_silhouette_analysis(df):
    """
    可视化轮廓系数分析
    """
    # 选择用于聚类的特征
    cluster_features = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']
    X_cluster = df[cluster_features]
    
    # 标准化特征
    scaler = StandardScaler()
    X_cluster_scaled = scaler.fit_transform(X_cluster)
    
    # 计算不同k值的轮廓系数
    silhouette_scores = []
    K_range = range(2, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(X_cluster_scaled)
        silhouette_avg = silhouette_score(X_cluster_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    # 绘制轮廓系数图
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, silhouette_scores, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Different k Values')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def visualize_feature_importance():
    """
    可视化特征重要性
    """
    # 这个函数将在主函数中根据实际模型结果调用
    pass

def visualize_correlation_matrix(df):
    """
    相关性分析可视化
    """
    # 选择数值型特征
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 移除不需要的列
    exclude_columns = ['Customer ID', 'Cluster']
    numeric_features = [col for col in numeric_features if col not in exclude_columns]
    
    # 计算相关性矩阵
    corr_matrix = df[numeric_features].corr()
    
    # 绘制热力图
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Matrix of Numeric Features')
    plt.tight_layout()
    plt.show()

def visualize_user_portrait(df):
    """
    可视化用户画像 (t-SNE)
    """
    # 选择用于t-SNE的特征
    features_for_tsne = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']
    X_tsne = df[features_for_tsne]
    
    # 应用t-SNE降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne_reduced = tsne.fit_transform(X_tsne)
    
    # 可视化
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_tsne_reduced[:, 0], X_tsne_reduced[:, 1], c=df['Cluster'], cmap='tab10', alpha=0.7)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Visualization of Customer Profiles')
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.show()

def main():
    # 加载数据
    df = pd.read_csv('clustered_data.csv')
    
    # 可视化用户基本特征分布
    print("可视化用户基本特征分布...")
    visualize_user_demographics(df)
    
    # 可视化购买行为特征
    print("可视化购买行为特征...")
    visualize_purchase_behavior(df)
    
    # 可视化模型结果
    print("可视化模型结果...")
    visualize_model_results(df)
    
    # 相关性分析可视化
    print("可视化相关性分析...")
    visualize_correlation_matrix(df)
    
    # 轮廓系数分析可视化
    print("可视化轮廓系数分析...")
    visualize_silhouette_analysis(df)
    
    # 可视化用户画像 (t-SNE)
    print("可视化用户画像 (t-SNE)...")
    visualize_user_portrait(df)
    
    print("所有可视化完成!")

if __name__ == "__main__":
    main()