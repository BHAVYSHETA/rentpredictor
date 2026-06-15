# 🏠 House Rent Prediction System

A Machine Learning project that predicts the monthly rent of a house based on various property features such as area, number of bedrooms, bathrooms, furnishing status, parking availability, city, and more.

## 📌 Project Overview

The goal of this project is to build a predictive model that estimates house rent prices using historical housing data. This helps property owners, tenants, and real estate businesses make data-driven decisions.

## 🚀 Features

* Data Cleaning and Preprocessing
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Machine Learning Model Training
* Model Evaluation and Comparison
* Rent Price Prediction
* Model Saving and Loading using Joblib

## 📊 Dataset Features

| Feature       | Description                              |
| ------------- | ---------------------------------------- |
| Area          | Size of property in square feet          |
| Bedrooms      | Number of bedrooms                       |
| Bathrooms     | Number of bathrooms                      |
| Floor         | Floor number                             |
| AgeOfProperty | Property age in years                    |
| Furnishing    | Furnished / Semi-Furnished / Unfurnished |
| Parking       | Parking availability                     |
| City          | Property location                        |
| Rent          | Target variable                          |

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Joblib
* Jupyter Notebook

## 📂 Project Structure

```text
House-Rent-Prediction/
│
├── data/
│   └── house_rent_dataset.csv
│
├── notebooks/
│   └── EDA.ipynb
│
├── models/
│   └── rent_predictor.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/house-rent-prediction.git
```

Move into the project folder:

```bash
cd house-rent-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Project

Train the model:

```bash
python train.py
```

Make predictions:

```bash
python predict.py
```

## 📈 Model Evaluation

Evaluation Metrics Used:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

## 💡 Future Improvements

* Deploy using Flask
* Deploy using Streamlit
* Add more city-specific features
* Improve prediction accuracy
* Integrate real-time housing data

## 👨‍💻 Author

**Bhavy Sheta**

Aspiring AI & Machine Learning Engineer

## ⭐ Support

If you found this project useful, consider giving it a star on GitHub.
