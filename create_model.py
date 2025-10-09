import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# ---------------- โหลด dataset ----------------
df = pd.read_csv('my_menu_dataset.csv')

# ---------------- แปลง last_time_eaten เป็นจำนวนวัน ----------------
# แก้ไขเพื่อรองรับวันที่ dd/mm/yyyy
df['last_time_eaten'] = pd.to_datetime(df['last_time_eaten'], dayfirst=True)
df['last_time_eaten'] = (pd.Timestamp.today() - df['last_time_eaten']).dt.days

# ---------------- Encode categorical ----------------
le_music = LabelEncoder()
df['music_mood'] = le_music.fit_transform(df['music_mood'])

le_menu = LabelEncoder()
df['menu'] = le_menu.fit_transform(df['menu'])

le_weekday = LabelEncoder()
df['weekday'] = le_weekday.fit_transform(df['weekday'])

# ---------------- Scale features ----------------
X = df.drop('menu', axis=1)
y = df['menu']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- แบ่ง Train/Test ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---------------- สร้างโมเดล Deep Learning ----------------
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(len(df['menu'].unique()), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# ---------------- Train โมเดล ----------------
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# ---------------- บันทึกโมเดล + encoder + scaler ----------------
model.save('menu_model.h5')
joblib.dump(le_music, 'encoder_music.pkl')
joblib.dump(le_menu, 'encoder_menu.pkl')
joblib.dump(le_weekday, 'encoder_weekday.pkl')
joblib.dump(scaler, 'scaler_features.pkl')

print("Model training complete and all files saved successfully.")
