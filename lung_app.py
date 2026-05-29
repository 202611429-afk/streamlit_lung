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
        
        try:
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
            sns.scatterplot(data=df, x='나이', y='흡연량', color='skyblue', alpha=0.5, ax=ax)
            
        # 2. 새 환자 표시 강조 (zorder=5로 설정하여 최상단 배치)
        ax.scatter(age_val, smoking_val, color='black', marker='X', s=350, linewidths=4, label='신규 환자 위치', zorder=5)
        
        # 🇰🇷 그래프 내부 라벨 및 타이틀 한글화 설정
        ax.set_title("나이 및 흡연량에 따른 환자 분포 그래프", fontsize=14, pad=15)
        ax.set_xlabel("나이", fontsize=12)
        ax.set_ylabel("흡연량", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # 범례 추출 후 매칭하여 한글로 재정의
        handles, labels = ax.get_legend_handles_labels()
        clean_labels = [f"군집 {l}" if l.isdigit() else "신규 환자 위치" for l in labels]
        ax.legend(handles=handles, labels=clean_labels, title="환자 분류 / 위치", fontsize=10, title_fontsize=11)
        
        # Streamlit 화면에 그래프 반영
        st.pyplot(fig)
        
    else:
        st.error("필수 파일(모델, 스케일러, 혹은 원본 데이터 CSV)이 누락되었습니다. 경로와 파일명을 다시 확인해주세요.")


# --- 하단 안내 ---
st.caption("제작: 환자 데이터 분석 시스템")
