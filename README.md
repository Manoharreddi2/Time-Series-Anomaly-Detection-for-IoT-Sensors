🧠 Time Series Anomaly Detection for IoT Sensors
📋 Project Overview

This project builds an AI-based anomaly detection system for IoT sensor data in manufacturing.
It identifies unusual readings that may indicate equipment faults or maintenance needs using:

Isolation Forest (statistical method)

LSTM Autoencoder (deep learning method)

⚙️ How to Run the Code
1. Install Dependencies

Run the following in your terminal or notebook:

pip install numpy pandas matplotlib seaborn scikit-learn statsmodels tensorflow


If your system doesn’t support TensorFlow GPU, install the CPU-only version:

pip install tensorflow-cpu

2. Run the Script
python TimeSeries_Anomaly_Detection_IoT_Assignment.py


or, if you’re using Jupyter Notebook,
open the script as a notebook and run all cells sequentially.

3. Outputs

Plots: EDA visuals and anomaly detection graphs will appear inline or in your window.

Results file: Model evaluation summary is saved to

./models/summary_results.json


Trained model files (if applicable): saved under ./models/

4. Optional: Use a Real Dataset

By default, synthetic data is generated with fake anomalies.
To use your own CSV sensor data:

Open the script.

Change these lines near the top:

DATA_SOURCE = 'csv'
PATH_TO_CSV = 'your_dataset.csv'


Make sure your CSV has a timestamp column and at least one sensor column.

🧾 Project Files
File	Description
TimeSeries_Anomaly_Detection_IoT_Assignment.py	Main Python script (end-to-end pipeline)
models/summary_results.json	Evaluation metrics and results
README.md	How to run and project overview
(optional) your_dataset.csv	Real IoT sensor data



🧩 Key Features

Automatic data cleaning and feature creation

Two anomaly detection approaches

Performance comparison (Precision, Recall, F1-score)

Visualizations of detected anomalies
