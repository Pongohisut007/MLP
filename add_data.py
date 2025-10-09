# add_data.py - Streamlit Web to Add Menu Data
import streamlit as st
import pandas as pd
from datetime import datetime
import os

DATA_FILE = 'my_menu_dataset.csv'

st.set_page_config(page_title="Add Menu Data", layout="centered")
st.title("➕ Add New Menu Data")
st.markdown("กรอกข้อมูลเพื่อลง dataset ของเมนูอาหาร")

col1, col2 = st.columns(2)
with col1:
    hour_options = [f"{h} {'AM' if h < 12 else 'PM'}" for h in range(7, 21)]
    hour_display = st.selectbox("Select Time", hour_options, index=5)
    hour = int(hour_display.split()[0]) + (12 if 'PM' in hour_display and int(hour_display.split()[0]) != 12 else 0)
    weekday = st.selectbox("Weekday (0=Mon)", list(range(7)))
    budget = st.number_input("Budget (THB)", min_value=10, max_value=500, value=50)
    last_time_eaten = st.date_input("Last Time Eaten", datetime.today())

with col2:
    group_size = st.slider("Group Size", 1, 10, 1)
    temperature = st.slider("Temperature (°C)", 20, 40, 30)
    rain = st.selectbox("Raining?", [0, 1])
    music_mood = st.selectbox("Music Mood", ['Happy', 'Chill', 'Energetic', 'Sad'])
    menu = st.text_input("Menu Name", "SomTam")

if st.button("Add Data"):
    last_time_str = last_time_eaten.strftime('%Y-%m-%d')
    new_data = {
        'hour': hour,
        'weekday': weekday,
        'budget': budget,
        'last_time_eaten': last_time_str,
        'group_size': group_size,
        'temperature': temperature,
        'rain': rain,
        'music_mood': music_mood,
        'menu': menu
    }

    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    else:
        df = pd.DataFrame([new_data])

    df.to_csv(DATA_FILE, index=False)
    st.success(f"✅ Added new data for menu '{menu}'!")

# ---- Show only menu list ----
st.divider()
st.subheader("📜 เมนูที่เคยเพิ่มไว้")
if os.path.exists(DATA_FILE):
    df_show = pd.read_csv(DATA_FILE)
    st.write(df_show['menu'].dropna().unique().tolist())
else:
    st.info("ยังไม่มีข้อมูลเมนูที่เคยเพิ่ม 😋")
