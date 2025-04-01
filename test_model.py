import pyshark
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

#load the trained model
cnn_model = load_model(r"D:\NND Project\CNN model\network_slice_model.keras")

scaler = StandardScaler()

#function to extract feature based on protocol
def extract_packet_features(packet):
    features = {}

    features['device_type'] = np.random.choice(['Smartphone', 'IoT', 'AR/VR'])

    if 'IP' in packet:
        features['ip_src'] = packet.ip.src 
        features['ip_dst'] = packet.ip.dst 
        features['ip_len'] = packet.ip.len  
        features['ip_ttl'] = packet.ip.ttl 
    
    if 'TCP' in packet:
        features['tcp_src_port'] = packet.tcp.srcport
        features['tcp_dst_port'] = packet.tcp.dstport 
        features['tcp_seq'] = packet.tcp.seq 
        features['tcp_ack'] = packet.tcp.ack  
        features['tcp_len'] = packet.tcp.len 
    
    if 'UDP' in packet:
        features['udp_src_port'] = packet.udp.srcport  
        features['udp_dst_port'] = packet.udp.dstport  
        features['udp_len'] = packet.udp.length 
    
    features['latency'] = np.random.uniform(0, 100) 
    features['throughput'] = np.random.uniform(1, 1000)  
    features['signal_strength'] = np.random.uniform(-100, 0)
    features['user_density'] = np.random.uniform(0, 100) 
    features['bandwidth'] = np.random.uniform(10, 1000) 
    features['packet_loss_rate'] = np.random.uniform(0, 5) 
    features['network_slice'] = np.random.choice(['eMBB', 'URLLC', 'mMTC'])

    return features

def process_pcap(file_path):
    capture = pyshark.FileCapture(file_path, keep_packets=False)
    
    # Lists to store feature values for each packet
    latency = []
    throughput = []
    signal_strength = []
    user_density = []
    bandwidth = []
    packet_loss_rate = []
    device_type = []
    network_slice = []

    # For packet loss rate, track TCP or UDP sequence numbers
    last_tcp_seq = {}
    last_udp_seq = {}

    for packet in capture:
        packet_features = extract_packet_features(packet)

        if 'TCP' in packet:
            tcp_src = packet.tcp.srcport
            tcp_seq = int(packet.tcp.seq)

            if tcp_src in last_tcp_seq:
                loss = tcp_seq - last_tcp_seq[tcp_src] - 1
                packet_features['packet_loss_rate'] = loss if loss >= 0 else 0

            last_tcp_seq[tcp_src] = tcp_seq

        elif 'UDP' in packet:
            udp_src = packet.udp.srcport
            udp_seq = int(packet.udp.length)

            if udp_src in last_udp_seq:
                # Calculate packet loss rate based on packet length (if possible)
                loss = udp_seq - last_udp_seq[udp_src] - 1
                packet_features['packet_loss_rate'] = loss if loss >= 0 else 0

            last_udp_seq[udp_src] = udp_seq
        
        latency.append(packet_features['latency'])
        throughput.append(packet_features['throughput'])
        signal_strength.append(packet_features['signal_strength'])
        user_density.append(packet_features['user_density'])
        bandwidth.append(packet_features['bandwidth'])
        packet_loss_rate.append(packet_features['packet_loss_rate'])
        device_type.append(packet_features['device_type'])
        network_slice.append(packet_features['network_slice'])

    # Create a DataFrame with the padded lists
    df = pd.DataFrame({
        'Latency (ms)': latency,
        'Throughput (Mbps)': throughput,
        'Signal Strength (dBm)': signal_strength,
        'User Density (user/km²)': user_density,  # Needs external data
        'Available Bandwidth (MHz)': bandwidth,
        'Packet Loss Rate (%)': packet_loss_rate,
        'Device Type': device_type,
        'Network Slice': network_slice
    })
    
    return df

#function to make predictions
def make_prediction_from_pcap(pcap_file):
    df = process_pcap(pcap_file)
    encoder = LabelEncoder()

    df['Device Type'] = LabelEncoder().fit_transform(df['Device Type'])    

    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values 

    X_scaled = scaler.fit_transform(X)
    encoded_Y = encoder.fit_transform(Y)
    dummy_y = to_categorical(encoded_Y)
    X_scaled_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

    #make predictions with the loaded model
    predictions = cnn_model.predict(X_scaled_reshaped)

    # Convert predictions from one-hot encoding back to labels
    predicted_labels = np.argmax(predictions, axis=1)
    # Map predictions back to network slice categories (eMBB, URLLC, mMTC)
    slice_labels = ['eMBB', 'URLLC', 'mMTC']
    predicted_slices = [slice_labels[label] for label in predicted_labels]
    
    return predicted_slices

pcap_file = r"D:\NND Project\capture.pcap"  # Path to your pcap file
predictions = make_prediction_from_pcap(pcap_file)

print(predictions)