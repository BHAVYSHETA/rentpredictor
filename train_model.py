from models.rentpredicter import rentprediction

rent_app = rentprediction("data/house_rent_dataset.csv")

rent_app.load_data()
rent_app.preprocess_data()
rent_app.split_data()
rent_app.scale_data()

rent_app.load_model()

print("Model Trained Successfully!")