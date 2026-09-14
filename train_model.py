import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# -------------------------------
# 1. Load and Preprocess Dataset
# -------------------------------
data_path = "F:/msc/Diabetes/dataset/augment.csv"
df = pd.read_csv(data_path)

# Fill missing numeric columns with mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Define features and target
features = [
    'Treatment_Type', 'Enrollment_Timing',
    'Edu_Category', 'Marital_Status', 'Loc_Type',
    'Age_Group', 'Sex', 'Blood_Sugar', 'Diabetes_Type'
]
target = 'Outcome'

# Extract features (X) and target (y)
X = df[features].values
y_raw = df[target].values

# Binary vs Multi-class classification handling
if len(np.unique(y_raw)) > 2:
    y = to_categorical(y_raw)
    output_units = y.shape[1]
    loss_fn = 'categorical_crossentropy'
    activation_fn = 'softmax'
else:
    y = y_raw.reshape(-1, 1)
    output_units = 1
    loss_fn = 'binary_crossentropy'
    activation_fn = 'sigmoid'

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# -------------------------------
# 2. Build & Compile Model
# -------------------------------
input_layer = Input(shape=(X_train.shape[1],))
x = Dense(64, activation='relu')(input_layer)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(output_units, activation=activation_fn)(x)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer=Adam(learning_rate=0.01), loss=loss_fn, metrics=['accuracy'])

# -------------------------------
# 3. Train the Model
# -------------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    verbose=1
)

# -------------------------------
# 4. Save Model and Scaler
# -------------------------------
model.save("model.h5", include_optimizer=False)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Training complete. Model and scaler saved successfully.")
