import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import numpy as np
from sklearn.pipeline import Pipeline
import joblib

class rentprediction():
    
    def __init__(self, path):
        
        self.path = path
        self.features = None
        
        self.target = 'Rent'
        
        self.df = None
        self.df_scaled = None
        
        self.X = None
        self.y = None
        
        self.X_train = None
        self.X_test = None
        
        self.y_train = None
        self.y_test = None
        
        self.le = None
        self.scaler = None
        self.model = None
        
        self.y_pred = None
        
    def load_data(self):
        self.df = pd.read_csv(self.path)
        
    def study_data(self):
        print(self.df.head())
        print("Shape of the data")
        print(f"Rows: {self.df.shape[0]}, Columns: {self.df.shape[1]}")
        print("Summary statistics of data")
        print(self.df.describe(include='all'))
        print("Sum of all null values of each row")
        print(self.df.isnull().sum())
        print(self.df.info())
        
    def preprocess_data(self):
        
        self.df = pd.get_dummies(
        self.df,
        columns=['Furnishing', 'Parking', 'City'],
        dtype=int
        )
        
        self.features = ['Area', 'Bedrooms', 'Bathrooms', 'Floor', 'AgeOfProperty', 'Furnishing_Furnished',
        'Furnishing_Semi-Furnished', 'Furnishing_Unfurnished', 'Parking_No', 'Parking_Yes', 'City_Ahmedabad',
        'City_Bangalore', 'City_Chennai', 'City_Delhi', 'City_Hyderabad', 'City_Pune'
        ]
        
    def split_data(self):
        
        self.X = self.df[self.features]
        self.y = self.df[self.target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
      
    def cross_validation(self):
          
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            # ('model', LinearRegression()), 
            # ('model', DecisionTreeRegressor()), 
            # ('model', RandomForestRegressor()), 
            ('model', XGBRegressor()), 
        ])     
             
        scores = cross_val_score(pipeline, self.X, self.y, cv=5, scoring='r2')
        
        print("\nCross Validation Auc score")
        print(scores)
        
        print(f"\nAverage r2 score: {scores.mean():.4f}")
        
    def feature_importance(self):
        feature_importance = pd.DataFrame({
            'Feature': self.features,
            'feature_importance': self.model.feature_importances_,
        })
        
        print(feature_importance.sort_values(by='feature_importance', ascending=False))
    
    def scale_data(self):
        
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
    def train_model(self):
        # self.model = LinearRegression()
        # self.model = DecisionTreeRegressor()
        # self.model = RandomForestRegressor()
        self.model = XGBRegressor()
        self.model.fit(self.X_train, self.y_train)
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler
        }, "saved_models/rent_model.pkl")
        
        self.y_pred = self.model.predict(self.X_test)
        
    def load_model(self):

        saved_data = joblib.load(
            "saved_models/rent_model.pkl"
        )

        self.model = saved_data["model"]
        self.scaler = saved_data["scaler"]    
    def evaluate(self):
        
        mae = mean_absolute_error(self.y_test, self.y_pred)
        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, self.y_pred)
        
        print(f"MAE : {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE   : {rmse:.4f}")
        print(f"R2 Score : {r2:.4f}")

    def plot(self):
                
        plt.figure(figsize=(8,5))
        sns.scatterplot(x='Area', y='Rent', data=self.df)
        plt.title("Area vs Rent")
        plt.xlabel("Area")
        plt.ylabel("Rent")

        plt.show()
        
        
    def predict_rent(self, area, bedrooms, bathrooms, floor, ageofproperty, furnishing, parking, city):

            if furnishing.lower() == 'furnished':
                furnishing = 'Furnished'
            elif furnishing.lower() == 'unfurnished':
                furnishing = 'Unfurnished'
            elif furnishing.lower() == 'semi-furnished':
                furnishing = 'Semi-Furnished'
            else:
                print("Invalid Input")

            if parking.lower() == 'yes':
                parking = 'Yes'
            elif parking.lower() == 'no':
                parking = 'No'
            else:
                print("Invalid Input")

            if city.lower() == 'delhi':
                city = 'Delhi'
            elif city.lower() == 'bangalore':
                city = 'Bangalore'
            elif city.lower() == 'pune':
                city = 'Pune'
            elif city.lower() == 'hyderabad':
                city = 'Hyderabad'
            elif city.lower() == 'ahmedabad':
                city = 'Ahmedabad'
            elif city.lower() == 'chennai':
                city = 'Chennai'
            else:
                print("Invalid Input")

            user_info = {}
            for col in self.features:
                if col not in user_info:
                    user_info[col] = 0
            
            user_info['Area'] = area
            user_info['Bedrooms'] = bedrooms
            user_info['Bathrooms'] = bathrooms
            user_info['Floor'] = floor
            user_info['AgeOfProperty'] = ageofproperty

            user_info[f"Furnishing_{furnishing}"] = 1
            user_info[f"Parking_{parking}"] = 1
            user_info[f"City_{city}"] = 1
            
            user_info = pd.DataFrame([user_info])

            user_info = user_info[self.features]

            user_info = self.scaler.transform(user_info)
            prediction = self.model.predict(user_info)[0]

            return{
                "Prediction": f"₹{prediction:,.0f}"    
            }
            

        

