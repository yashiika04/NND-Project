import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

#Step-1 Create synthetic data
np.random.seed(42)
latency = np.random.uniform(1,100,1000)
throughput = np.random.uniform(10,1000,1000)
signal_strength = np.random.uniform(-120,-40,1000)
user_density = np.random.uniform(10,1000,1000)
bandwidth = np.random.uniform(5,100,1000)
packet_loss_rate = np.random.uniform(0,5,1000)
device_type = np.random.choice(['Smartphone','IoT','AR/VR'],size=1000)

network_slice = []
for i in range(1000):
  if latency[i]<10 and throughput[i]>500:
    network_slice.append("eMBB")
  elif latency[i]>50 and user_density[i]>500:
    network_slice.append("URLLC")
  else:
    network_slice.append("mMTC")

df = pd.DataFrame({
    'Latency (ms)' : latency,
    'Throughput (Mbps)': throughput,
    'Signal Strength (dBm)': signal_strength,
    'User Density (user/km²)': user_density,
    'Available Bandwidth (MHz)': bandwidth,
    'Packet Loss Rate (%)': packet_loss_rate,
    'Device Type': device_type,
    'Network Slice': network_slice
})
df['Device Type'] = LabelEncoder().fit_transform(df['Device Type'])

# Step 2: Preprocess data
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values
scaler = StandardScaler()
X[:,:-1]= scaler.fit_transform(X[:,:-1])
encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(Y)
dummy_y = to_categorical(encoded_Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.3, random_state=42)

#Step 3: CNN-based model
model = Sequential([
    Conv1D(32, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=1),
    Dropout(0.3),
    Conv1D(64, kernel_size=2, activation='relu'),
    MaxPooling1D(pool_size=1),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

history = model.fit(X_train_cnn, Y_train, validation_split=0.2, epochs=20, batch_size=128)

#Step 4: Evaluate the model
scores = model.evaluate(X_test_cnn, Y_test)
print("Accuracy: %.2f%%" % (scores[1]*100))

# Step 5: Save the model
model.save('network_slice_model.keras') 

print("Model saved to 'network_slice_model.keras'")
