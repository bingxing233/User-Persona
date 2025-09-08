import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

def perform_clustering(df):
    """
    执行K-Means聚类分析
    """
    # 选择用于聚类的特征
    cluster_features = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']
    X_cluster = df[cluster_features]
    
    # 标准化特征
    scaler = StandardScaler()
    X_cluster_scaled = scaler.fit_transform(X_cluster)
    
    # 使用手肘法确定最佳聚类数
    inertias = []
    silhouette_scores = []
    ch_scores = []
    K_range = range(1, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_cluster_scaled)
        inertias.append(kmeans.inertia_)
        
        # 对于k>1的情况，计算轮廓系数和CH指数
        if k > 1:
            cluster_labels = kmeans.labels_
            silhouette_avg = silhouette_score(X_cluster_scaled, cluster_labels)
            ch_score = calinski_harabasz_score(X_cluster_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            ch_scores.append(ch_score)
    
    # 绘制手肘法折线图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 手肘法图
    axes[0].plot(range(1, 11), inertias, 'bo-')
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Inertia')
    axes[0].set_title('Elbow Method for Optimal k')
    axes[0].grid(True)
    
    # 轮廓系数图
    axes[1].plot(range(2, 11), silhouette_scores, 'ro-')
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Score for Different k Values')
    axes[1].grid(True)
    
    # CH指数图
    axes[2].plot(range(2, 11), ch_scores, 'go-')
    axes[2].set_xlabel('Number of Clusters (k)')
    axes[2].set_ylabel('Calinski-Harabasz Score')
    axes[2].set_title('Calinski-Harabasz Score for Different k Values')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('clustering_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 添加评估说明
    print("聚类评估说明:")
    print("1. 手肘法(Elbow Method): 通过观察Inertia随k值变化的'肘部'来选择最佳k值")
    print("2. 轮廓系数(Silhouette Score): 衡量聚类的紧密度和分离度，值越接近1越好")
    print("3. Calinski-Harabasz指数: 衡量聚类间分离度与聚类内紧密度的比率，值越大越好")
    
    # 选择最佳聚类数（这里选择轮廓系数最高的k值）
    best_k_index = np.argmax(silhouette_scores)
    best_k = best_k_index + 2  # 因为silhouette_scores从k=2开始计算
    print(f"\n根据轮廓系数选择最佳聚类数: {best_k}")
    
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_cluster_scaled)
    
    # 将聚类标签添加到数据框
    df['Cluster'] = cluster_labels
    
    # 可视化聚类结果
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(df['Age'], df['Purchase Amount (USD)'], c=df['Cluster'], cmap='viridis')
    plt.xlabel('Age')
    plt.ylabel('Purchase Amount (USD)')
    plt.title('Clustering Results: Age vs Purchase Amount')
    plt.colorbar(scatter)
    
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(df['Review Rating'], df['Previous Purchases'], c=df['Cluster'], cmap='viridis')
    plt.xlabel('Review Rating')
    plt.ylabel('Previous Purchases')
    plt.title('Clustering Results: Review Rating vs Previous Purchases')
    plt.colorbar(scatter)
    
    plt.tight_layout()
    plt.show()
    
    # 保存聚类模型
    joblib.dump(kmeans, 'kmeans_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    print(f"K-Means聚类完成，最佳聚类数: {best_k}")
    print("聚类分布:")
    print(df['Cluster'].value_counts())
    
    # 聚类评估结果
    final_silhouette = silhouette_score(X_cluster_scaled, cluster_labels)
    final_ch = calinski_harabasz_score(X_cluster_scaled, cluster_labels)
    print(f"\n最终聚类评估结果:")
    print(f"轮廓系数: {final_silhouette:.4f}")
    print(f"Calinski-Harabasz指数: {final_ch:.4f}")
    
    return df, kmeans, scaler

def perform_classification(df):
    """
    执行分类任务 - 预测订阅状态
    """
    # 准备数据
    # 选择特征和目标变量
    # 首先排除明显不需要的列
    exclude_columns = [
        'Customer ID', 'Subscription Status', 'Item Purchased', 'Frequency of Purchases',
        'Preferred Payment Method', 'Age Group'
    ]
    
    # 如果存在聚类列也排除
    if 'Cluster' in df.columns:
        exclude_columns.append('Cluster')
    
    # 获取所有数值型列
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 同时添加一些已知的文本特征列（如果存在）
    additional_columns = ['Gender', 'Category', 'Location', 'Season', 'Payment Method', 
                         'Shipping Type', 'Discount Applied', 'Promo Code Used']
    
    # 合并所有可能的特征列
    potential_features = numeric_columns.copy()
    for col in additional_columns:
        if col in df.columns and col not in exclude_columns:
            potential_features.append(col)
    
    # 最终特征列（排除不需要的）
    feature_columns = [col for col in potential_features if col not in exclude_columns]
    
    print("用于分类的特征列:")
    print(feature_columns)
    
    X = df[feature_columns]
    y = df['Subscription Status']
    
    # 对文本特征进行标签编码
    label_encoders = {}
    X_encoded = X.copy()
    
    for col in X_encoded.columns:
        if X_encoded[col].dtype == 'object':
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            label_encoders[col] = le
    
    print("编码后的特征数据类型:")
    print(X_encoded.dtypes)
    
    # 保存特征列信息
    feature_info = {
        'feature_columns': feature_columns,
        'numeric_features': [col for col in feature_columns if col in numeric_columns],
        'categorical_features': [col for col in feature_columns if col not in numeric_columns]
    }
    joblib.dump(feature_info, 'feature_info.pkl')
    
    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    # 特征缩放
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 使用默认参数或手动设置参数（替代网格搜索）
    # 逻辑回归
    lr_model = LogisticRegression(random_state=42, max_iter=1000, C=1.0, penalty='l2', solver='liblinear')
    lr_model.fit(X_train_scaled, y_train)
    print("逻辑回归模型已训练（使用默认参数）")
    
    # 决策树
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=20, min_samples_leaf=10)
    dt_model.fit(X_train_scaled, y_train)
    print("决策树模型已训练（使用预设参数）")
    
    # 随机森林
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10, min_samples_split=20)
    rf_model.fit(X_train_scaled, y_train)
    print("随机森林模型已训练（使用预设参数）")
    
    # 支持向量机
    from sklearn.svm import SVC
    svm_model = SVC(random_state=42, probability=True, C=1.0, kernel='rbf', gamma='scale')
    svm_model.fit(X_train_scaled, y_train)
    print("支持向量机模型已训练（使用预设参数）")
    
    # 预测
    lr_pred = lr_model.predict(X_test_scaled)
    dt_pred = dt_model.predict(X_test_scaled)
    rf_pred = rf_model.predict(X_test_scaled)
    svm_pred = svm_model.predict(X_test_scaled)
    
    # 评估模型
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    
    def evaluate_model(y_true, y_pred, y_prob, model_name):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except:
            roc_auc = 0.0
        
        print(f"\n{model_name} 评估报告:")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
    
    # 获取预测概率
    try:
        lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]  # 取正类概率
    except:
        lr_prob = np.zeros(len(X_test_scaled))
        
    try:
        dt_prob = dt_model.predict_proba(X_test_scaled)[:, 1]
    except:
        dt_prob = np.zeros(len(X_test_scaled))
        
    try:
        rf_prob = rf_model.predict_proba(X_test_scaled)[:, 1]
    except:
        rf_prob = np.zeros(len(X_test_scaled))
        
    try:
        svm_prob = svm_model.predict_proba(X_test_scaled)[:, 1]
    except:
        svm_prob = np.zeros(len(X_test_scaled))
    
    # 转换y_test为数值型标签
    y_test_numeric = (y_test == 'Yes').astype(int)
    
    # 评估所有模型
    lr_metrics = evaluate_model(y_test_numeric, (lr_pred == 'Yes').astype(int), lr_prob, "逻辑回归")
    dt_metrics = evaluate_model(y_test_numeric, (dt_pred == 'Yes').astype(int), dt_prob, "决策树")
    rf_metrics = evaluate_model(y_test_numeric, (rf_pred == 'Yes').astype(int), rf_prob, "随机森林")
    svm_metrics = evaluate_model(y_test_numeric, (svm_pred == 'Yes').astype(int), svm_prob, "支持向量机")
    
    # 保存模型评估结果
    model_evaluation = {
        '逻辑回归': lr_metrics,
        '决策树': dt_metrics,
        '随机森林': rf_metrics,
        '支持向量机': svm_metrics
    }
    joblib.dump(model_evaluation, 'model_evaluation.pkl')
    
    # 保存模型和编码器
    joblib.dump(lr_model, 'logistic_regression_model.pkl')
    joblib.dump(dt_model, 'decision_tree_model.pkl')
    joblib.dump(rf_model, 'random_forest_model.pkl')
    joblib.dump(svm_model, 'svm_model.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(scaler, 'feature_scaler.pkl')
    
    # 特征重要性分析 (比较多种方法)
    # 随机森林特征重要性
    rf_feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # 决策树特征重要性
    dt_feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': dt_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n随机森林特征重要性:")
    print(rf_feature_importance.head(10))
    
    print("\n决策树特征重要性:")
    print(dt_feature_importance.head(10))
    
    # 保存特征重要性
    joblib.dump(rf_feature_importance, 'feature_importance.pkl')
    
    # 可视化特征重要性
    plt.figure(figsize=(10, 6))
    top_features = rf_feature_importance.head(10)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importance (Random Forest)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    # 模型比较可视化
    metrics_df = pd.DataFrame({
        '逻辑回归': pd.Series(lr_metrics),
        '决策树': pd.Series(dt_metrics),
        '随机森林': pd.Series(rf_metrics),
        '支持向量机': pd.Series(svm_metrics)
    }).T
    
    plt.figure(figsize=(12, 6))
    metrics_df[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']].plot(kind='bar')
    plt.title('模型评估指标比较')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
    
    return lr_model, dt_model, rf_model, svm_model, X_train_scaled, X_test_scaled, y_train, y_test, label_encoders, feature_columns

def explain_model_with_shap(model, X_train, X_test, model_name):
    """
    使用SHAP解释模型
    """
    try:
        # 创建SHAP解释器
        explainer = shap.Explainer(model, X_train)
        # 只选择前50个样本以减少计算复杂度
        shap_values = explainer(X_test[:50])
        
        # 保存SHAP解释结果
        shap_explanation = {
            'shap_values': shap_values,
            'X_test_sample': X_test[:50],
            'model_name': model_name
        }
        joblib.dump(shap_explanation, 'shap_explanation.pkl')
        
        # 绘制SHAP图
        plt.figure(figsize=(10, 6))
        # 对于多类分类问题，需要指定输出列
        if hasattr(shap_values, 'values') and len(shap_values.values.shape) > 2:
            shap.plots.beeswarm(shap_values[:, :, 1], show=False)  # 选择第二类(Yes类)
        else:
            shap.plots.beeswarm(shap_values, show=False)
        plt.title(f'SHAP Beeswarm Plot - {model_name}')
        plt.tight_layout()
        plt.savefig('shap_beeswarm.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 添加SHAP特征重要性排序
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test[:50], plot_type="bar", show=False)
        plt.title(f'SHAP Feature Importance - {model_name}')
        plt.tight_layout()
        plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"SHAP解释完成，结果已保存到 shap_explanation.pkl")
        
    except Exception as e:
        print(f"SHAP解释过程中出现错误: {str(e)}")
        print("跳过SHAP可视化")

def main():
    # 加载处理后的数据
    df = pd.read_csv('processed_data.csv')
    
    print("执行聚类分析...")
    df, kmeans_model, scaler = perform_clustering(df)
    
    print("\n执行分类分析...")
    lr_model, dt_model, rf_model, svm_model, X_train, X_test, y_train, y_test, label_encoders, feature_columns = perform_classification(df)
    
    # 使用SHAP解释最佳模型 (随机森林通常表现最好)
    print("\n使用SHAP解释随机森林模型...")
    explain_model_with_shap(rf_model, X_train, X_test, "Random Forest")
    
    # 保存带有聚类标签的数据
    df.to_csv('clustered_data.csv', index=False)
    
    print("\n机器学习建模完成!")

if __name__ == "__main__":
    main()