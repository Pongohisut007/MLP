# app.py - AI Menu Predictor with Recommendations
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# ---- โหลดโมเดลและ encoder/scaler ----
model = load_model('menu_model.h5')
le_music = joblib.load('encoder_music.pkl')
le_menu = joblib.load('encoder_menu.pkl')
scaler = joblib.load('scaler_features.pkl')

# ---- Streamlit UI ----
st.set_page_config(page_title="Menu Predictor", layout="centered")
st.title("🍽️ AI Menu Predictor")
st.markdown("กรอกข้อมูลเพื่อให้ AI ทำนายเมนูอาหารที่คุณน่าจะเลือก")

# ---- Inputs ----
with st.form("predict_form"):
    st.subheader("🕒 Time & Day")
    col1, col2 = st.columns(2)
    
    # เวลาแบบ AM/PM
    with col1:
        hours = []
        for h in range(7, 21):
            suffix = "AM" if h < 12 else "PM"
            display_hour = h if h <= 12 else h - 12
            hours.append(f"{display_hour} {suffix}")
        hour_str = st.selectbox("Hour", options=hours, index=5)  # default 12 PM
        
        # แปลงกลับเป็นชั่วโมง 24 ชม.
        if "AM" in hour_str:
            hour = int(hour_str.split()[0])
        else:
            hour = int(hour_str.split()[0])
            if hour != 12:
                hour += 12

        weekday = st.selectbox("Weekday", 
                               options=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])

    with col2:
        last_time_eaten = st.slider("Days since last eaten this menu", 0, 10, 1)

    st.subheader("💰 Budget & Group")
    col3, col4 = st.columns(2)
    with col3:
        budget = st.number_input("Budget (THB)", min_value=10, max_value=500, value=50, step=10)
        group_size = st.slider("Group Size", 1, 10, 1)
    with col4:
        temperature = st.slider("Temperature (°C)", 20, 40, 30)
        rain = st.radio("Raining?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")

    st.subheader("🚶‍♂️ Travel Option")
    travel = st.radio("ต้องเดินทางหรือไม่?", [0, 1], format_func=lambda x: "ไม่ต้องเดินทาง" if x==0 else "ต้องเดินทาง")

    st.subheader("🎵 Music Mood")
    music_mood = st.selectbox("Music Mood", ['Happy', 'Chill', 'Energetic', 'Sad'])

    submitted = st.form_submit_button("Predict Menu")

# ---- Predict ----
if submitted:
    # เตรียมข้อมูลเข้าโมเดล
    input_data = np.array([[hour, 
                            list(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]).index(weekday),
                            budget,
                            last_time_eaten,
                            group_size,
                            temperature,
                            rain,
                            travel,
                            le_music.transform([music_mood])[0]]])
    input_scaled = scaler.transform(input_data)

    # ทำนาย
    pred_prob = model.predict(input_scaled)[0]
    
    # จัด Top 3 เมนู
    top3_idx = np.argsort(pred_prob)[-3:][::-1]
    top3_menus = le_menu.inverse_transform(top3_idx)
    top3_probs = pred_prob[top3_idx]
    
    st.subheader("🍴 Top 3 Predicted Menus")
    for menu, prob in zip(top3_menus, top3_probs):
        st.write(f"**{menu}** — {prob*100:.1f}%")
        if prob > 0.5:
            st.success("✅ เมนูนี้น่าจะเหมาะกับคุณ")
        elif prob > 0.3:
            st.info("⚖️ เมนูนี้โอเค แต่ยังมีตัวเลือกอื่นที่น่าสนใจ")
        else:
            st.warning("❌ เมนูนี้อาจไม่ตรงความชอบเท่าไหร่")

    # กราฟความน่าจะเป็นทั้งหมด
    all_menus = le_menu.inverse_transform(np.arange(len(pred_prob)))
    prob_df = pd.DataFrame({'Menu': all_menus, 'Probability': pred_prob})
    st.subheader("📊 Probability of All Menus")
    st.bar_chart(prob_df.set_index('Menu'))
