import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA

# Path to the anomaly log
anomaly_log_path = os.path.join(os.path.dirname(__file__), 'anomalies_log.csv')
# Path to the normal log (user should create this if they want to compare)
normal_log_path = os.path.join(os.path.dirname(__file__), 'normal_log.csv')

# Read anomalies
anomalies = pd.read_csv(anomaly_log_path)

# Try to read normal data if available
if os.path.exists(normal_log_path):
    normal = pd.read_csv(normal_log_path)
    has_normal = True
else:
    normal = None
    has_normal = False

plt.figure(figsize=(8, 6))

# Plot normal points if available
if has_normal:
    sns.scatterplot(
        data=normal,
        x='packet_size',
        y='duration_ms',
        color='gray',
        label='Normal',
        s=40,
        alpha=0.5
    )

# Plot anomalies
sns.scatterplot(
    data=anomalies,
    x='packet_size',
    y='duration_ms',
    hue='confidence_score',
    palette='coolwarm',
    style=None,
    s=80,
    legend='brief',
    label='Anomaly'
)

plt.title('Packet Size vs Duration: Normal vs Anomaly')
plt.xlabel('Packet Size')
plt.ylabel('Duration (ms)')
plt.legend(title='Legend')
plt.tight_layout()
plt.show()

# --- PCA Visualization ---
# Combine normal and anomaly data for PCA
pca_features = ['src_port', 'dst_port', 'packet_size', 'duration_ms']

anomalies['is_anomaly'] = 1
if has_normal:
    normal['is_anomaly'] = 0
    combined = pd.concat([normal, anomalies], ignore_index=True)
else:
    combined = anomalies.copy()

# Fill missing values if any
combined = combined.fillna(0)

# Run PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(combined[pca_features])
combined['PC1'] = X_pca[:, 0]
combined['PC2'] = X_pca[:, 1]

plt.figure(figsize=(8, 6))
if has_normal:
    sns.scatterplot(
        data=combined[combined['is_anomaly'] == 0],
        x='PC1', y='PC2',
        color='gray', label='Normal', s=40, alpha=0.5
    )
sns.scatterplot(
    data=combined[combined['is_anomaly'] == 1],
    x='PC1', y='PC2',
    hue='confidence_score',
    palette='coolwarm',
    s=80,
    legend='brief',
    label='Anomaly'
)
plt.title('PCA Projection: Normal vs Anomaly')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Legend')
plt.tight_layout()
plt.show()
