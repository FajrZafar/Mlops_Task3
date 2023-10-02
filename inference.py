import pickle

# Load the model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)



def predict(input_data):
    prediction = model.predict([input_data])
    return prediction.tolist()

# Example usage
# Replace this with your actual input data and prediction logic
input_data = [[5.1, 3.5, 1.4, 0.2]]  # Example input
prediction = predict(input_data)
print("Prediction:", prediction)
