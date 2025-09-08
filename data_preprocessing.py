import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data(file_path):
    """
    加载并清洗数据
    """
    # 读取数据
    df = pd.read_csv(file_path)
    
    print("原始数据形状:", df.shape)
    print("数据基本信息:")
    print(df.info())
    
    # 检查缺失值
    print("\n缺失值统计:")
    print(df.isnull().sum())
    
    # 异常值检查 - Age字段
    print("\nAge字段统计:")
    print(df['Age'].describe())
    
    # 绘制Age箱线图
    plt.figure(figsize=(10, 6))
    plt.boxplot(df['Age'])
    plt.title('Age Distribution Boxplot (Before Processing)')
    plt.ylabel('Age')
    plt.savefig('age_boxplot_before.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 异常值检查 - Purchase Amount字段
    print("\nPurchase Amount (USD)字段统计:")
    print(df['Purchase Amount (USD)'].describe())
    
    # 绘制Purchase Amount箱线图
    plt.figure(figsize=(10, 6))
    plt.boxplot(df['Purchase Amount (USD)'])
    plt.title('Purchase Amount Distribution Boxplot (Before Processing)')
    plt.ylabel('Purchase Amount (USD)')
    plt.savefig('purchase_amount_boxplot_before.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 检查Gender字段的唯一值
    print("\nGender字段唯一值:")
    print(df['Gender'].unique())
    
    # 检查Category字段的唯一值
    print("\nCategory字段唯一值:")
    print(df['Category'].unique())
    
    # 检查Item Purchased字段是否有明显错误
    print("\nItem Purchased字段样本:")
    print(df['Item Purchased'].head(10))
    
    return df

def handle_missing_values(df):
    """
    处理缺失值
    """
    # 检查各字段缺失值
    missing_data = df.isnull().sum()
    print("缺失值统计:")
    print(missing_data[missing_data > 0])
    
    # 数值型字段使用中位数填充
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"使用中位数 {median_val} 填充 {col} 的缺失值")
    
    # 文本型字段使用众数填充
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col].fillna(mode_val, inplace=True)
            print(f"使用众数 {mode_val} 填充 {col} 的缺失值")
    
    return df

def feature_encoding(df):
    """
    特征编码
    """
    # 独热编码 - Gender, Category, Location等
    onehot_columns = ['Gender', 'Category', 'Location', 'Season', 'Payment Method', 'Shipping Type']
    df_encoded = pd.get_dummies(df, columns=onehot_columns, prefix=onehot_columns)
    
    # 标签编码 - Frequency of Purchases
    le = LabelEncoder()
    if 'Frequency of Purchases' in df.columns:
        df_encoded['Frequency of Purchases_Encoded'] = le.fit_transform(df['Frequency of Purchases'])
        print("Frequency of Purchases 编码映射:")
        for i, class_name in enumerate(le.classes_):
            print(f"  {class_name} -> {i}")
    
    return df_encoded

def feature_engineering(df):
    """
    特征构建
    """
    # 根据Age划分年龄组 - 根据最大最小年龄分成6个组
    min_age = df['Age'].min()
    max_age = df['Age'].max()
    # 计算6个年龄组的边界
    bins = np.linspace(min_age, max_age, 7)  # 6个组需要7个边界点
    labels = [f'{int(bins[i])}-{int(bins[i+1])}' for i in range(6)]
    df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, include_lowest=True)
    print("Age Group 分布:")
    print(df['Age Group'].value_counts())
    
    # 计算平均购买金额
    df['Average Purchase Amount'] = df['Purchase Amount (USD)'] / (df['Previous Purchases'] + 1)
    print("\nAverage Purchase Amount 统计:")
    print(df['Average Purchase Amount'].describe())
    
    # 添加RFM特征
    # Recency - 最近购买时间(这里用Previous Purchases作为近似)
    df['Recency'] = df['Previous Purchases']
    
    # Frequency - 购买频率(已存在)
    # Monetary - 购买金额(已存在)
    
    # 计算总消费金额
    df['Total Spending'] = df['Purchase Amount (USD)'] * (df['Previous Purchases'] + 1)
    
    # 计算用户价值评分
    df['Value Score'] = df['Average Purchase Amount'] * df['Review Rating']
    
    return df

def main():
    # 加载并清洗数据
    df = load_and_clean_data('shopping_trends.csv')
    
    # 处理缺失值
    df = handle_missing_values(df)
    
    # 特征工程
    df = feature_engineering(df)
    
    # 特征编码
    df_encoded = feature_encoding(df)
    
    # 保存处理后的数据
    df.to_csv('processed_data.csv', index=False)
    df_encoded.to_csv('encoded_data.csv', index=False)
    
    # 显示处理后的数据分布
    plt.figure(figsize=(10, 6))
    plt.boxplot(df['Age'])
    plt.title('Age Distribution Boxplot (After Processing)')
    plt.ylabel('Age')
    plt.savefig('age_boxplot_after.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(df['Purchase Amount (USD)'])
    plt.title('Purchase Amount Distribution Boxplot (After Processing)')
    plt.ylabel('Purchase Amount (USD)')
    plt.savefig('purchase_amount_boxplot_after.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n处理完成!")
    print(f"原始数据形状: {df.shape}")
    print(f"编码后数据形状: {df_encoded.shape}")
    print("数据已保存为 processed_data.csv 和 encoded_data.csv")
    print("处理前后的对比图表已保存为:")
    print("  - age_boxplot_before.png")
    print("  - age_boxplot_after.png")
    print("  - purchase_amount_boxplot_before.png")
    print("  - purchase_amount_boxplot_after.png")

if __name__ == "__main__":
    main()