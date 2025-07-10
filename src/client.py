import socket
import json
import pandas as pd
import joblib
import os
import csv
from together import Together

HOST = 'localhost'
PORT = 9999

model = joblib.load("/Users/amin/Documents/University/Term 6/AI in Cyber Security/Final Project/anomaly-detection-project-xotlet/src/anomaly_model.joblib")

def pre_process_data(data):
    # Convert data to DataFrame for model prediction
    df = pd.DataFrame([data])
    # One-hot encode 'protocol' column, drop first to match training
    df_encoded = pd.get_dummies(df, columns=['protocol'], drop_first=True)
    # Ensure the column exists (in case protocol is 'TCP')
    if 'protocol_UDP' not in df_encoded.columns:
        df_encoded['protocol_UDP'] = False
    # Reorder columns to match training
    df_encoded = df_encoded[['src_port', 'dst_port', 'packet_size', 'duration_ms', 'protocol_UDP']]
    return df_encoded

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    buffer = ""
    print("Client connected to server.\n")

    while True:
        chunk = s.recv(1024).decode()
        if not chunk:
            break
        buffer += chunk

        while '\n' in buffer:
            line, buffer = buffer.split('\n', 1)
            try:
                data = json.loads(line)
                print(f'Data Received:\n{data}\n')

                X = pre_process_data(data)
                prediction = model.predict(X)[0]
                # Get anomaly score (the lower, the more abnormal)
                confidence_score = model.decision_function(X)[0]
                if prediction == -1:
                    # Anomaly detected
                    print(f"🚨 Anomaly Detected! (Confidence score: {confidence_score:.4f}) Sending to LLM for explanation...")

                    # Prepare to log anomaly
                    log_path = os.path.join(os.path.dirname(__file__), 'anomalies_log.csv')
                    log_headers = ['src_port', 'dst_port', 'packet_size', 'duration_ms', 'protocol', 'confidence_score', 'llm_explanation']

                    TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "")
                    if TOGETHER_API_KEY == "":
                        print("TOGETHER_API_KEY is not set")
                        exit()
                    client = Together(api_key=TOGETHER_API_KEY)
                    user_prompt = (
                        f"Network sensor reading: {data}\n"
                        "Identify the type of anomaly and provide a brief explanation for its possible cause."
                    )
                    messages = [
                        {"role": "system", "content": "You are a cybersecurity assistant that labels and explains network anomalies."},
                        {"role": "user", "content": user_prompt}
                    ]
                    try:
                        response = client.chat.completions.create(
                            model="meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
                            messages=messages,
                            stream=False,
                        )
                        llm_reply = response.choices[0].message.content.strip() if response.choices[0].message.content else "No response from LLM."
                    except Exception as e:
                        llm_reply = f"LLM call failed: {e}"

                    print(f"\n🚨 Anomaly Detected!\nData: {data}\nConfidence Score: {confidence_score:.4f}\nLLM Explanation: {llm_reply}\n")

                    # Log anomaly to CSV
                    log_row = [
                        data.get('src_port'),
                        data.get('dst_port'),
                        data.get('packet_size'),
                        data.get('duration_ms'),
                        data.get('protocol'),
                        confidence_score,
                        llm_reply
                    ]
                    file_exists = os.path.isfile(log_path)
                    with open(log_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        if not file_exists:
                            writer.writerow(log_headers)
                        writer.writerow(log_row)
                else:
                    print("normal")
                    # Log normal point to normal_log.csv
                    normal_log_path = os.path.join(os.path.dirname(__file__), 'normal_log.csv')
                    normal_log_headers = ['src_port', 'dst_port', 'packet_size', 'duration_ms', 'protocol', 'confidence_score']
                    normal_log_row = [
                        data.get('src_port'),
                        data.get('dst_port'),
                        data.get('packet_size'),
                        data.get('duration_ms'),
                        data.get('protocol'),
                        confidence_score
                    ]
                    file_exists = os.path.isfile(normal_log_path)
                    with open(normal_log_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        if not file_exists:
                            writer.writerow(normal_log_headers)
                        writer.writerow(normal_log_row)

            except json.JSONDecodeError:
                print("Error decoding JSON.")
