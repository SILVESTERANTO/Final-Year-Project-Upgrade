from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading
import time
import random
import sqlite3
from datetime import datetime
import hashlib
import os
from sklearn.preprocessing import StandardScaler
import json

app = Flask(__name__)
app.secret_key = 'network_intrusion_detection_secret_key_2024'

class NetworkIntrusionDetector:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.simulation_running = False
        self.simulation_thread = None
        self.results_queue = []
        self.alert_threshold = 0.8
        self.simulation_data = None
        self.simulation_labels = None
        self.data_index = 0
        self.load_models()
        self.load_simulation_data()
        
    def load_models(self):
        try:
            self.model = load_model('models/cnn_lstm_intrusion_model.keras')
            with open('models/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            with open('models/label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            print("Models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")
            
    def load_simulation_data(self):
        dataset_files = [
            'dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
            'dataset/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
            'dataset/Friday-WorkingHours-Morning.pcap_ISCX.csv',
            'dataset/Monday-WorkingHours.pcap_ISCX.csv',
            'dataset/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
            'dataset/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
            'dataset/Tuesday-WorkingHours.pcap_ISCX.csv',
            'dataset/Wednesday-workingHours.pcap_ISCX.csv'
        ]
        
        print("Loading CIC-IDS datasets...")
        all_data = []
        for file_path in dataset_files:
            try:
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    df.columns = df.columns.str.strip()
                    print(f"✓ {os.path.basename(file_path)}: {df.shape[0]:,} records")
                    all_data.append(df)
                else:
                    print(f"✗ File not found: {file_path}")
            except Exception as e:
                print(f"✗ Error loading {file_path}: {e}")
        
        if all_data:
            print("Combining datasets...")
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_data = combined_data.replace([np.inf, -np.inf], np.nan)
            combined_data = combined_data.fillna(0)
            
            label_col = 'Label'
            features = combined_data.drop(columns=[label_col])
            labels = combined_data[label_col]
            
            # Create more balanced sample for better demo
            sample_size = 15000
            
            # More balanced distribution for diverse simulation
            target_distribution = {
                'BENIGN': 0.40,           # 40% normal (6,000)
                'DoS Hulk': 0.15,         # 15% DoS Hulk (2,250)
                'PortScan': 0.15,         # 15% PortScan (2,250)
                'DDoS': 0.10,             # 10% DDoS (1,500)
                'DoS GoldenEye': 0.05,    # 5% DoS GoldenEye (750)
                'FTP-Patator': 0.05,      # 5% FTP attacks (750)
                'SSH-Patator': 0.05,      # 5% SSH attacks (750)
                'Web Attack': 0.05        # 5% Web attacks (750)
            }
            
            balanced_samples = []
            balanced_labels = []
            
            for attack_type, target_ratio in target_distribution.items():
                target_count = int(sample_size * target_ratio)
                
                if attack_type == 'Web Attack':
                    # Get all web attack types
                    mask = labels.str.contains('Web Attack', na=False, case=False)
                else:
                    # Exact match for other attacks
                    mask = (labels == attack_type)
                
                available_data = combined_data[mask]
                
                if len(available_data) >= target_count:
                    # Sample exactly the target count
                    sampled_data = available_data.sample(n=target_count, replace=False)
                    balanced_samples.append(sampled_data.drop(columns=[label_col]))
                    balanced_labels.extend(sampled_data[label_col].tolist())
                    print(f"   ✓ {attack_type}: {target_count:,} records")
                elif len(available_data) > 0:
                    # Use all available data and note the shortfall
                    balanced_samples.append(available_data.drop(columns=[label_col]))
                    balanced_labels.extend(available_data[label_col].tolist())
                    print(f"   ⚠ {attack_type}: {len(available_data):,} records (wanted {target_count:,})")
                else:
                    print(f"   ✗ {attack_type}: No data found")
            
            if balanced_samples:
                self.simulation_data = pd.concat(balanced_samples, ignore_index=True)
                self.simulation_labels = pd.Series(balanced_labels)
                
                # Shuffle the data for random order during simulation
                shuffle_indices = np.random.permutation(len(self.simulation_data))
                self.simulation_data = self.simulation_data.iloc[shuffle_indices].reset_index(drop=True)
                self.simulation_labels = self.simulation_labels.iloc[shuffle_indices].reset_index(drop=True)
            else:
                # Fallback to random sampling if balancing fails
                print("   Falling back to random sampling...")
                sample_indices = np.random.choice(len(features), min(sample_size, len(features)), replace=False)
                self.simulation_data = features.iloc[sample_indices]
                self.simulation_labels = labels.iloc[sample_indices]
            
            print(f"✓ Simulation data ready: {self.simulation_data.shape[0]:,} records from {len(all_data)} files")
            print(f"✓ Attack distribution:")
            for attack, count in self.simulation_labels.value_counts().head().items():
                print(f"   - {attack}: {count:,}")
        else:
            print("✗ No datasets found - using fallback simulation")
            self.simulation_data = None
            self.simulation_labels = None
            
    def preprocess_data(self, data):
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)
        
        df.columns = df.columns.str.strip()
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            if col not in df.columns:
                continue
            try:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
            except:
                df[col] = 0
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df_numeric = df[numeric_columns]
        
        if df_numeric.shape[1] != 78:
            if df_numeric.shape[1] < 78:
                for i in range(78 - df_numeric.shape[1]):
                    df_numeric[f'feature_{i}'] = 0
            else:
                df_numeric = df_numeric.iloc[:, :78]
        
        try:
            scaled_data = self.scaler.transform(df_numeric)
            return scaled_data
        except Exception as e:
            print(f"Scaling error: {e}")
            return df_numeric.values
        
    def create_sequences(self, data, seq_length=5):
        if len(data) < seq_length:
            data = np.tile(data, (seq_length, 1))[:seq_length]
        
        sequences = []
        for i in range(len(data) - seq_length + 1):
            sequences.append(data[i:i + seq_length])
        
        if len(sequences) == 0:
            sequences = [data[:seq_length]]
            
        return np.array(sequences)
    
    def predict(self, network_data):
        try:
            processed_data = self.preprocess_data(network_data)
            sequences = self.create_sequences(processed_data)
            
            predictions = self.model.predict(sequences, verbose=0)
            
            avg_prediction = np.mean(predictions, axis=0)
            predicted_class = np.argmax(avg_prediction)
            confidence = np.max(avg_prediction)
            
            attack_type = self.label_encoder.classes_[predicted_class]
            
            return {
                'attack_type': attack_type,
                'confidence': float(confidence),
                'is_attack': attack_type != 'BENIGN',
                'probabilities': {class_name: float(prob) for class_name, prob in 
                               zip(self.label_encoder.classes_, avg_prediction)}
            }
        except Exception as e:
            return {
                'attack_type': 'ERROR',
                'confidence': 0.0,
                'is_attack': False,
                'error': str(e)
            }

detector = NetworkIntrusionDetector()

def init_db():
    conn = sqlite3.connect('network_intrusion.db')
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  email TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS detections
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  attack_type TEXT NOT NULL,
                  confidence REAL NOT NULL,
                  source_ip TEXT,
                  dest_ip TEXT,
                  alert_sent BOOLEAN DEFAULT FALSE)''')
    
    conn.commit()
    conn.close()

def send_alert_email(detection_data):
    try:
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        sender_email = "ayushtiwari.creatorslab@gmail.com"
        sender_password = "tecx bcym vxdz dtni"
        receiver_email = "ayush.qriocity@gmail.com"
        
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = receiver_email
        message["Subject"] = f"SECURITY ALERT: {detection_data['attack_type']} Detected"
        
        body = f"""
        NETWORK INTRUSION DETECTED
        
        Attack Type: {detection_data['attack_type']}
        Confidence: {detection_data['confidence']:.2%}
        Timestamp: {detection_data['timestamp']}
        Source IP: {detection_data.get('source_ip', 'Unknown')}
        Destination IP: {detection_data.get('dest_ip', 'Unknown')}
        
        Immediate action may be required to secure your network.
        
        Network Intrusion Detection System
        """
        
        message.attach(MIMEText(body, "plain"))
        
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        text = message.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
        
        return True
    except Exception as e:
        print(f"Email sending failed: {e}")
        return False

def simulate_network_traffic():
    if detector.simulation_data is None:
        print("No simulation data available - using fallback")
        while detector.simulation_running:
            try:
                result = {
                    'id': len(detector.results_queue) + 1,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'attack_type': 'BENIGN',
                    'confidence': 0.95,
                    'is_attack': False,
                    'source_ip': f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
                    'dest_ip': f"10.0.{random.randint(1, 255)}.{random.randint(1, 255)}"
                }
                
                detector.results_queue.append(result)
                if len(detector.results_queue) > 100:
                    detector.results_queue.pop(0)
                
                time.sleep(2)
            except Exception as e:
                print(f"Fallback simulation error: {e}")
                time.sleep(1)
        return
    
    data_size = len(detector.simulation_data)
    print(f"Starting simulation with {data_size} records")
    
    while detector.simulation_running:
        try:
            if detector.data_index >= data_size:
                detector.data_index = 0
                print("Restarting simulation data cycle")
            
            network_record = detector.simulation_data.iloc[detector.data_index].to_dict()
            true_label = detector.simulation_labels.iloc[detector.data_index]
            detector.data_index += 1
            
            prediction = detector.predict(network_record)
            
            result = {
                'id': len(detector.results_queue) + 1,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'attack_type': prediction['attack_type'],
                'confidence': prediction['confidence'],
                'is_attack': prediction['is_attack'],
                'source_ip': f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
                'dest_ip': f"10.0.{random.randint(1, 255)}.{random.randint(1, 255)}",
                'true_label': true_label
            }
            
            detector.results_queue.append(result)
            if len(detector.results_queue) > 100:
                detector.results_queue.pop(0)
            
            # Database storage
            try:
                conn = sqlite3.connect('network_intrusion.db')
                c = conn.cursor()
                c.execute('''INSERT INTO detections 
                            (attack_type, confidence, source_ip, dest_ip, alert_sent)
                            VALUES (?, ?, ?, ?, ?)''',
                         (result['attack_type'], result['confidence'], 
                          result['source_ip'], result['dest_ip'], False))
                
                if result['is_attack'] and result['confidence'] > detector.alert_threshold:
                    if send_alert_email(result):
                        c.execute('UPDATE detections SET alert_sent = TRUE WHERE id = ?', 
                                 (c.lastrowid,))
                
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"Database error: {e}")
            
            time.sleep(random.uniform(1, 2))
            
        except Exception as e:
            print(f"Simulation error: {e}")
            time.sleep(1)
            
            prediction = detector.predict(network_packet)
            
            result = {
                'id': len(detector.results_queue) + 1,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'attack_type': prediction['attack_type'],
                'confidence': prediction['confidence'],
                'is_attack': prediction['is_attack'],
                'source_ip': f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
                'dest_ip': f"10.0.{random.randint(1, 255)}.{random.randint(1, 255)}"
            }
            
            detector.results_queue.append(result)
            if len(detector.results_queue) > 100:
                detector.results_queue.pop(0)
            
            conn = sqlite3.connect('network_intrusion.db')
            c = conn.cursor()
            c.execute('''INSERT INTO detections 
                        (attack_type, confidence, source_ip, dest_ip, alert_sent)
                        VALUES (?, ?, ?, ?, ?)''',
                     (result['attack_type'], result['confidence'], 
                      result['source_ip'], result['dest_ip'], False))
            
            if result['is_attack'] and result['confidence'] > detector.alert_threshold:
                if send_alert_email(result):
                    c.execute('UPDATE detections SET alert_sent = TRUE WHERE id = ?', 
                             (c.lastrowid,))
            
            conn.commit()
            conn.close()
            
            time.sleep(random.uniform(1, 3))
            
        except Exception as e:
            print(f"Simulation error: {e}")
            time.sleep(1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        conn = sqlite3.connect('network_intrusion.db')
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE email = ? AND password = ?', 
                 (email, hashed_password))
        user = c.fetchone()
        conn.close()
        
        if user:
            session['user_id'] = user[0]
            session['email'] = user[1]
            return jsonify({'success': True, 'redirect': '/dashboard'})
        else:
            return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        try:
            conn = sqlite3.connect('network_intrusion.db')
            c = conn.cursor()
            c.execute('INSERT INTO users (email, password) VALUES (?, ?)', 
                     (email, hashed_password))
            conn.commit()
            conn.close()
            return jsonify({'success': True})
        except sqlite3.IntegrityError:
            return jsonify({'success': False, 'error': 'Email already exists'}), 400
    
    return render_template('signup.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/start_simulation', methods=['POST'])
def start_simulation():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    if not detector.simulation_running:
        detector.simulation_running = True
        detector.simulation_thread = threading.Thread(target=simulate_network_traffic)
        detector.simulation_thread.start()
        return jsonify({'success': True, 'message': 'Simulation started'})
    else:
        return jsonify({'success': False, 'message': 'Simulation already running'})

@app.route('/stop_simulation', methods=['POST'])
def stop_simulation():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    detector.simulation_running = False
    if detector.simulation_thread and detector.simulation_thread.is_alive():
        detector.simulation_thread.join()
    return jsonify({'success': True, 'message': 'Simulation stopped'})

@app.route('/get_results')
def get_results():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    return jsonify({
        'results': detector.results_queue[-20:],
        'simulation_status': detector.simulation_running,
        'total_detections': len(detector.results_queue)
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.get_json()
        result = detector.predict(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/set_threshold', methods=['POST'])
def set_threshold():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    threshold = float(request.json.get('threshold', 0.8))
    detector.alert_threshold = max(0.1, min(1.0, threshold))
    return jsonify({'success': True, 'threshold': detector.alert_threshold})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    init_db()
    
    if not os.path.exists('models'):
        os.makedirs('models')
        print("Please place your trained models in the 'models' folder:")
        print("- cnn_lstm_intrusion_model.keras")
        print("- scaler.pkl")
        print("- label_encoder.pkl")
    
    print("=" * 60)
    print("🚀 SecureNet AI - Network Intrusion Detection System")
    print("🤖 CNN-LSTM Model Ready | 📊 CIC-IDS Data Loaded")
    print("🌐 Dashboard: http://127.0.0.1:5000")
    print("=" * 60)
    
    app.run(debug=False, host='0.0.0.0', port=5000)