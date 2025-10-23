🧠 Time Series Anomaly Detection (IoT Project)
📖 Project Summary

This project detects anomalies (unusual readings) in IoT sensor data using machine learning and deep learning.
It helps in finding faulty machines or abnormal behavior in a factory setup.

🧩 Methods Used

Isolation Forest – A machine learning model to find outliers.

LSTM Autoencoder – A deep learning model that learns normal patterns and flags anything different.

⚙️ How to Run
Step 1: Install Required Libraries

Run this in your terminal or Jupyter Notebook:

pip install numpy pandas matplotlib seaborn scikit-learn statsmodels tensorflow


If TensorFlow doesn’t install, try:

pip install tensorflow-cpu

Step 2: Run the Code

If using Python file:

python TimeSeries_Anomaly_Detection_IoT_Assignment.py


If using Jupyter Notebook:

Open the file

Click Run All

Step 3: View Results

Graphs will show normal and anomalous data points.

A summary file (like model scores) will be saved in the models/ folder.


📊 Output

Cleaned and visualized sensor data

Anomalies highlighted in graphs

Precision, recall, and F1-score shown for models

✅ Example Use

You can use this for:

Predictive maintenance

Detecting faulty sensors

Monitoring equipment health
