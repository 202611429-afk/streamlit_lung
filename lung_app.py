import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm  # 폰트 관리를 위해 추가
import seaborn as sns
import os

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

# 🔥 [한글 깨짐 방지 핵심 코드 추가] ---
# 작업 폴더 내에 저장된 NanumGothic.ttf 경로를 잡습니다.
font_path = os.path.join(os.path.dirname(__file__), "NanumGothic.ttf")

if os.path.exists(font_path):
    # 서버 환경(배포용): 파일이 있으면 해당 나눔폰트 적용
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
else:
    # 로컬 환경(컴퓨터 테스트용): 파일이 없으면 PC에 기본 내장된 맑은 고딕 적용
    plt.rcParams['font.family'] = 'Malgun Gothic'

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호가 깨지는 현상 방지


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
        
        # 📊 원래 그래프 출력
        st.subheader("📈 환자 군집 내 신규 환자 위치")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        try:
            # 원본 데이터에 실시간으로 군집 정보를 계산해 채워 넣음
            df_features = df[['나이', '흡연량', '음주량']]
            df_scaled = scaler.transform(df_features)
            df['cluster'] = model.predict(df_scaled)
            
            # 원래 주피터의 선명한 색상 구성(viridis), 투명도(alpha=0.5) 반영
            sns.scatterplot(
                data=df, x='나이', y='흡연량', 
                hue='cluster', palette='viridis', 
                alpha=0.5, s=80, ax=ax, edgecolor='none'
            )
        except Exception as e:
            # 예외 발생 시 안전 장치
            sns.scatterplot(data=df, x='나이', y='흡연량', color='skyblue', alpha=0.5, ax=ax)
            
        # 2. 새 환자 표시 강조 (zorder=5로 설정하여 최상단 배치)
        ax.scatter(age_val, smoking_val, color='black', marker='X', s=350, linewidths=4, label='신규 환자 위치', zorder=5)
        
        # 원래 축 이름 지정 반영
        ax.set_title("나이 및 흡연량에 따른 환자 분포", fontsize=14, pad=15)
        ax.set_xlabel("나이", fontsize=12)
        ax.set_ylabel("흡연량", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # 범례 갱신
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, title="환자 분류/위치")
        
        # Streamlit 화면에 띄우기
        st.pyplot(fig)
        
    else:
        st.error("필수 파일(모델, 스케일러, 혹은 원본 데이터 CSV)이 누락되었습니다. 경로와 파일명을 다시 확인해주세요.")

# --- 하단 안내 ---
st.caption("제작: 환자 데이터 분석 시스템")
