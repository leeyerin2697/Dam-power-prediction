import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ===== 1. 데이터 불러오기 =====
df = pd.read_csv("한국수자원공사_수문현황정보_일별.csv", encoding='utf-8')

# ===== 2. 사용할 컬럼 =====
features = [
    '저수위', 
    '저수량', 
    '유입량', 
    '총방류량', 
    '강수량', 
    '금년누가강우량', 
    '저수율'
]
target = '발전방류량'

# ===== 3. 컬럼 추출 + 결측치 제거 =====
data = df[features + [target, '댐명']].dropna()
data = data[
    (data['저수위'] >= 0) &
    (data['저수량'] >= 0) &
    (data['유입량'] >= 0) &
    (data['총방류량'] >= 0) &
    (data['강수량'] >= 0) &
    (data['금년누가강우량'] >= 0) &
    (data['저수율'] >= 0)
]


print("원본 데이터 크기:", len(data))

# ===== 4. 대용량 데이터 샘플링 (속도 개선 핵심) =====
if len(data) > 50000:
    data = data.sample(n=50000, random_state=42)
    print("샘플링 후 데이터 크기:", len(data))

# ===== 5. 타입 최적화 =====
X = data[features].astype('float32')
y = data[target].astype('float32')

# ===== 6. 데이터 분할 =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ===== 7. 모델 1: 선형 회귀 =====
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

# ===== 8. 모델 2: Random Forest (경량화 버전) =====
rf = RandomForestRegressor(
    n_estimators=30,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

# ===== 9. 성능 비교 =====
print("\n====== 모델 성능 비교 ======")
print("Linear Regression MSE:", mean_squared_error(y_test, pred_lr))
print("Random Forest MSE:", mean_squared_error(y_test, pred_rf))

print("Linear Regression R2:", r2_score(y_test, pred_lr))
print("Random Forest R2:", r2_score(y_test, pred_rf))

import matplotlib.pyplot as plt

models = ['Linear Regression', 'Random Forest']
mse_values = [
    mean_squared_error(y_test, pred_lr),
    mean_squared_error(y_test, pred_rf)
]

plt.figure()
plt.bar(models, mse_values)
plt.title("Model Comparison (MSE)")
plt.xlabel("Model")
plt.ylabel("MSE")
plt.show()


r2_values = [
    r2_score(y_test, pred_lr),
    r2_score(y_test, pred_rf)
]

plt.figure()
plt.bar(models, r2_values)
plt.title("Model Comparison (R² Score)")
plt.xlabel("Model")
plt.ylabel("R²")
plt.show()


plt.figure()
plt.scatter(y_test, pred_lr)
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()])
plt.title("Linear Regression: Actual vs Predicted")
plt.xlabel("Actual Value")
plt.ylabel("Predicted Value")
plt.show()

plt.figure()
plt.scatter(y_test, pred_rf)
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()])
plt.title("Random Forest: Actual vs Predicted")
plt.xlabel("Actual Value")
plt.ylabel("Predicted Value")
plt.show()

import matplotlib.pyplot as plt
import pandas as pd

# 변수 중요도 추출
importances = rf.feature_importances_

# 데이터프레임으로 정리
feat_importance = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(feat_importance)


plt.figure()
plt.barh(feat_importance['Feature'], feat_importance['Importance'])
plt.xlabel("Feature Importance")
plt.ylabel("Variables")
plt.title("Feature Importance - Random Forest")
plt.gca().invert_yaxis()
plt.show()