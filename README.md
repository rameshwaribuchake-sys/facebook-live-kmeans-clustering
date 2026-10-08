Facebook Live K-Means Clustering Dashboard

📌 Project Overview
This project applies K-Means Clustering to the Facebook Live dataset (Live.csv) to group records into meaningful clusters based on their engagement-related features.
An interactive Streamlit dashboard is also provided to visualize the clusters and predict the cluster for a new record.

🎯 Objectives
Perform data preprocessing on the Facebook Live dataset.
Encode the categorical status_type feature.
Scale the features using MinMaxScaler.
Apply K-Means clustering.
Use 4 clusters (K = 4) based on the project analysis.
Visualize cluster distribution and feature information.
Provide an interactive Streamlit dashboard.
Predict the cluster of a new input record.

🛠️ Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Streamlit
Joblib
Jupyter Notebook

📂 Project Structure
facebook-live-kmeans-clustering/
│
├── app.py
├── Live.csv
├── kmeans_clustering_artifacts.pkl
├── requirements.txt


📊 Dataset
The project uses the Facebook Live dataset stored in Live.csv.
The dataset contains Facebook Live post information, including the status_type categorical feature and numerical engagement-related features.

🔄 Methodology
1. Data Loading
The Live.csv dataset is loaded using Pandas.
2. Data Preprocessing
Unnecessary columns are removed before clustering.
The categorical status_type column is converted into numerical form using LabelEncoder.
3. Feature Scaling
The features are normalized using MinMaxScaler so that the features are on a comparable scale.
4. K-Means Clustering
The K-Means algorithm is applied to group similar records.
The final dashboard uses:
Number of Clusters (K) = 4
Random State = 0
5. Cluster Analysis
The resulting clusters are analyzed using:
Cluster distribution
Cluster percentages
Status type vs cluster
Numerical feature summary
6. Dashboard
A Streamlit dashboard provides an interactive interface for exploring the clustering results and predicting the cluster of a new record.

📈 Dashboard Features
The dashboard includes:

📊 Total records

🔢 Number of clusters

📋 Cluster distribution

📌 Cluster percentage

📉 Status type vs cluster analysis

📑 Numerical feature summary

🔍 New record cluster prediction

🎛️ Cluster filtering

📋 Filtered data table

🤖 Model File

The project uses one combined PKL file:
kmeans_clustering_artifacts.pkl

This file contains:
K-Means clustering model
MinMaxScaler
LabelEncoder
Feature columns
Preprocessing information
Using one combined PKL file makes the dashboard deployment simpler.

⚙️ Installation
Clone the repository:
git clone https://github.com/YOUR_USERNAME/facebook-live-kmeans-clustering.git

Move into the project directory:
cd facebook-live-kmeans-clustering

Install the required libraries:
pip install -r requirements.txt

▶️ Run the Dashboard
Run the following command:
streamlit run app.py
The Streamlit dashboard will open in your browser.

📁 Required Files
Make sure these files are present in the same folder:
app.py
Live.csv
kmeans_clustering_artifacts.pkl
requirements.txt


🔮 Future Scope
Deploy the dashboard online using Streamlit Community Cloud.
Add more interactive visualizations.
Compare K-Means with other clustering algorithms.
Add automated cluster profiling.
Improve dashboard UI and user interaction.

👩‍💻 Project
Facebook Live K-Means Clustering
Developed using Python, Machine Learning and Streamlit.
