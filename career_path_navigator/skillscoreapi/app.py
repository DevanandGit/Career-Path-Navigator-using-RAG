from fastapi import FastAPI
import numpy as np
import tensorflow as tf
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder

# Load Trained Model
model = tf.keras.models.load_model("skill_analysis_model.h5")

# Label Encoder (same as used during training)
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['Beginner', 'Intermediate', 'Advanced'])

app = FastAPI()

# Define Input Format
class StudentResponse(BaseModel):
    Q1: int  # 0 or 1
    Q2: int  # 0 or 1
    Q3: int  # 0, 1, or 2
    Q4: int  # 0 or 1
    Q5: int  # 1, 2, or 3

# Prediction Endpoint
@app.post("/predict-skill")
def predict_skill(student: StudentResponse):
    # Convert input to numpy array
    input_data = np.array([[student.Q1, student.Q2, student.Q3, student.Q4, student.Q5]])
    
    # Get model prediction
    prediction = model.predict(input_data)
    skill_index = np.argmax(prediction)  # Get highest probability class
    skill_label = label_encoder.classes_[skill_index]  # Convert index to skill level
    
    return {"predicted_skill": skill_label}
