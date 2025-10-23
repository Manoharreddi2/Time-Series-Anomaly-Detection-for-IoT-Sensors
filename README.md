🧠 Time Series Anomaly Detection (IoT Project)
📖 Project Summary

This project detects anomalies (unusual readings) in IoT sensor data using both machine learning and deep learning.
It helps identify when machines behave abnormally — which could mean a fault or maintenance need.

🧩 Methods Used

🤖 Isolation Forest – Finds outlier readings using unsupervised learning.

🧬 LSTM Autoencoder – Learns normal patterns and detects when data doesn’t match them.

⚙️ How to Run
🪄 Step 1: Install Required Libraries

 Run this in your terminal or Jupyter Notebook:

 pip install numpy pandas matplotlib seaborn scikit-learn statsmodels tensorflow


If TensorFlow doesn’t install properly, try:

pip install tensorflow-cpu

🚀 Step 2: Run the Code

 If using a Python file:

 python TimeSeries_Anomaly_Detection_IoT_Assignment.py


If using a Jupyter Notebook:

Open the .ipynb file

Click Run All

📈 Step 3: View Results

 Graphs will show normal vs anomalous data points

 Model performance (Precision, Recall, F1-score) will be displayed

 Results and charts may be saved in the project folder


📊 Outputs You’ll See

🧹 Cleaned and preprocessed sensor data

📉 Time series plots showing anomalies

📊 Model comparison metrics (Precision, Recall, F1-score)

💡 Use Cases

This project can be used for:

🏭 Predictive Maintenance

⚙️ Equipment Health Monitoring

🧰 Faulty Sensor Detection
