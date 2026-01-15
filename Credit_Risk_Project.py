import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import os
import warnings

warnings.filterwarnings('ignore')

# 1. 读取数据
print("正在读取数据...")
if not os.path.exists('german_credit_data.csv'):
    print("错误：找不到 credit_data.csv！")
    exit()

df = pd.read_csv('german_credit_data.csv', index_col=0)

# 2. 构建模拟 Target (Risk)

print("检测到数据缺失标签，正在基于业务逻辑生成 'Risk' 列...")

# 逻辑：如果没有存款(NaN)或者存款很少(little)，并且借钱比较多或时间长，算作高风险
def simulate_risk(row):
    risk_score = 0

    if pd.isna(row['Saving accounts']) or row['Saving accounts'] == 'little':
        risk_score += 1

    if pd.isna(row['Checking account']) or row['Checking account'] == 'little':
        risk_score += 1

    if row['Credit amount'] > 5000:
        risk_score += 1

    if row['Duration'] > 24:
        risk_score += 1

    return 1 if risk_score >= 3 else 0

df['Target'] = df.apply(simulate_risk, axis=1)

print(f"标签生成完毕！")
print(f"坏人(1)数量: {sum(df['Target']==1)}")
print(f"好人(0)数量: {sum(df['Target']==0)}")
print("-" * 30)

# 3. 自动化预处理
print("正在清洗数据...")
for col in df.columns:
    if col != 'Target':
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('Unknown')
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        else:
            df[col] = df[col].fillna(df[col].median())

# 4. 准备训练
X = df.drop('Target', axis=1)
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 训练模型 (加入权重处理不平衡)
scale_pos_weight = sum(y==0) / sum(y==1)
print(f"应用类别权重 (scale_pos_weight): {scale_pos_weight:.2f}")

print("正在训练 XGBoost...")
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    use_label_encoder=False
)

model.fit(X_train, y_train)

# 6. 评估结果
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)

print("\n" + "="*50)
print(f"准确率 (Accuracy): {acc*100:.2f}%")
print("="*50)

print("\n详细分类报告:")
print(classification_report(y_test, predictions))

print("\n决定违约的关键因素 (Top 5):")
importances = pd.Series(model.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False).head(5))