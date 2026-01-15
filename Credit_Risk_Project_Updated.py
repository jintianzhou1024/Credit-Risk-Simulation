import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import os
import random

# 1. 读取数据
if not os.path.exists('german_credit_data.csv'):
    print("找不到 credit_data.csv")
    exit()
df = pd.read_csv('german_credit_data.csv', index_col=0)

# 2. 构建带噪音的目标变量 (Risk)
print("正在生成模拟数据 (加入 15% 随机噪音)...")

def simulate_real_world_risk(row):
    # 基础分
    risk_score = 0
    if pd.isna(row['Saving accounts']) or row['Saving accounts'] == 'little':
        risk_score += 1
    if pd.isna(row['Checking account']) or row['Checking account'] == 'little':
        risk_score += 1
    if row['Credit amount'] > 5000:
        risk_score += 1
    if row['Duration'] > 24:
        risk_score += 1

    # 原始判决：>=3 分算坏人
    is_bad = 1 if risk_score >= 3 else 0

    # 注入噪音 (Noise Injection)
    if random.random() < 0.10:
        is_bad = 1 - is_bad # 0变1，1变0

    return is_bad

df['Target'] = df.apply(simulate_real_world_risk, axis=1)

# 3. 预处理
for col in df.columns:
    if col != 'Target':
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('Unknown')
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        else:
            df[col] = df[col].fillna(df[col].median())

# 4. 训练
X = df.drop('Target', axis=1)
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scale_pos_weight = sum(y==0) / sum(y==1)

model = XGBClassifier(
    n_estimators=100,      # 树少一点，防止过拟合
    max_depth=3,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# 5. 评估
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)

print("\n" + "="*50)
print(f"真实模拟完成！准确率: {acc*100:.2f}%")
print("="*50)

print("\n分类报告:")
print(classification_report(y_test, predictions))

print("\n关键风险特征 (Top 5):")
importances = pd.Series(model.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False).head(5))