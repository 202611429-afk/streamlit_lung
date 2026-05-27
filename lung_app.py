import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# --- 🔍 [코드로만 한글 깨짐 방지 처리] ---
@st.cache_resource
def init_korean_font():
    # 리눅스 배포 서버 환경에서 한글 깨짐을 방지하기 위해 기본 고딕 서체를 강제 매핑합니다.
    plt.rcParams['font.family'] = 'sans-serif'
    # sans-serif 폰트 그룹의 최우선 순위로 한글 지원 폰트들을 선언합니다.
    plt.rcParams['font.sans-serif'] = ['Malgun Gothic', 'NanumGothic', 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

init_korean_font()


# --- 1. 모델, 스케일러 및 데이터셋 불러오기 ---
@st.cache_resource
def load_models_and_data():
    try:
        scaler = joblib.load('lung_scaler.pkl')
        model = joblib.load('lung_model.pkl')
        df = pd.read_csv('lung.csv') 
        return scaler, model, df
    except Exception as e:
        st.error(f"파일을 로드하는 중 오류가 발생했습니다: {e}")
        return None, None, None

scaler, model, df = load_models_and_data()


# --- 2. 웹 UI 구성 ---
st.set_page_config(page_title="환자 군집 예측 시스템", page_icon="🏥", layout="centered")
st.title("🏥 폐암환자 데이터 군집 예측")
st.write("나이, 흡연량, 음주량을 입력하여 환자의 군집(Cluster)을 확인하고 위치를 시각화합니다.")

st.divider()


# --- 3. 사용자 데이터 입력 (입력 위젯) ---
col1, col2, col3 = st.columns(3)

with col1:
    age_val = st.number_input("나이 입력", min_value=0.0, max_value=120.0, value=30.0, step=1.0)
with col2:
    smoking_val = st.number_input("흡연량 입력", min_value=0.0, value=0.0, step=0.1)
with col3:
    drinking_val = st.number_input("음주량 입력", min_value=0.0, value=0.0, step=0.1)


# --- 4. 예측 및 시각화 실행 ---
if st.button("예측하기", type="primary"):
    if scaler is not None and model is not None and df is not None:
        # 데이터프레임 생성
        new_patient = pd.DataFrame(
            [[age_val, smoking_val, drinking_val]], 
            columns=['나이', '흡연량', '음주량']
        )
        
        # 스케일링 및 모델 예측
        new_patient_scaled = scaler.transform(new_patient)
        pred_cluster = model.predict(new_patient_scaled)
        
        # 🎯 결과 텍스트 출력
        st.subheader("🎯 예측 결과")
        st.success(f"이 환자는 **{pred_cluster[0]}번 군집**에 속합니다.")
        
        st.divider()
        
        # 📊 원래 그래프 출력 영역
        st.subheader("📈 환자 군집 내 신규 환자 위치")
        
        fig, ax = plt.subplots(figsize=(8, 6))
