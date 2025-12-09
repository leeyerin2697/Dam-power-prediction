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

