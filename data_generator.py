import pandas as pd
import numpy as np
import random
from faker import Faker

fake = Faker()

# Define the questions structure
questions = {
    "Coding": [
        "Do you enjoy solving problems through programming? (Yes/No)",
        "How comfortable are you with learning new programming languages? (Low/Medium/High)",
        "Have you ever developed a project from scratch? (Yes/No)",
        "Do you prefer working with data or building user interfaces? (Data/UI)",
        "How would you rate your debugging skills? (Beginner/Intermediate/Advanced)"
    ],
    "Business": [
        "Do you enjoy working with numbers and financial analysis? (Yes/No)",
        "How comfortable are you with managing teams and projects? (Low/Medium/High)",
        "Have you ever started a small business or side hustle? (Yes/No)",
        "Do you have experience with market research and analysis? (Yes/No)",
        "How would you rate your leadership skills? (Beginner/Intermediate/Advanced)"
    ],
    "Commerce": [
        "Are you interested in working with accounts and financial transactions? (Yes/No)",
        "How comfortable are you with creating reports and summaries? (Low/Medium/High)",
        "Have you worked with inventory management or logistics? (Yes/No)",
        "Do you have a strong understanding of taxation and business laws? (Yes/No)",
        "How would you rate your understanding of financial markets? (Beginner/Intermediate/Advanced)"
    ],
    "Law": [
        "Do you enjoy reading and interpreting legal documents? (Yes/No)",
        "How comfortable are you with public speaking and presenting arguments? (Low/Medium/High)",
        "Have you ever worked or interned in a legal firm? (Yes/No)",
        "Do you have an interest in international law or human rights? (Yes/No)",
        "How would you rate your understanding of the Indian Penal Code or Civil Law? (Beginner/Intermediate/Advanced)"
    ],
    "Medicine": [
        "Are you interested in helping people improve their health? (Yes/No)",
        "How comfortable are you with studying human biology and anatomy? (Low/Medium/High)",
        "Have you ever volunteered in a healthcare setting? (Yes/No)",
        "Do you have experience with diagnosing or treating illnesses? (Yes/No)",
        "How would you rate your knowledge of medical procedures or treatments? (Beginner/Intermediate/Advanced)"
    ],
    "Architecture": [
        "Do you enjoy designing and planning spaces or buildings? (Yes/No)",
        "How comfortable are you with using design software (AutoCAD, Revit, etc.)? (Low/Medium/High)",
        "Have you ever created architectural models or blueprints? (Yes/No)",
        "Do you have knowledge of construction materials and methods? (Yes/No)",
        "How would you rate your creativity in designing spaces? (Beginner/Intermediate/Advanced)"
    ],
    "Journalism": [
        "Are you passionate about writing and storytelling? (Yes/No)",
        "How comfortable are you with researching and investigating news topics? (Low/Medium/High)",
        "Have you ever worked as a reporter or editor for a publication? (Yes/No)",
        "Do you enjoy working under deadlines and in fast-paced environments? (Yes/No)",
        "How would you rate your ability to communicate effectively in writing? (Beginner/Intermediate/Advanced)"
    ],
    "Social Service": [
        "Do you feel passionate about helping others and making a difference? (Yes/No)",
        "How comfortable are you with working in diverse communities? (Low/Medium/High)",
        "Have you volunteered for NGOs or social organizations? (Yes/No)",
        "Are you interested in advocating for underprivileged groups? (Yes/No)",
        "How would you rate your problem-solving skills when working in social settings? (Beginner/Intermediate/Advanced)"
    ]
}

# Create a scoring system for each question type
def score_yes_no(answer):
    return 1 if answer == "Yes" else 0

def score_low_med_high(answer):
    mapping = {"Low": 0, "Medium": 1, "High": 2}
    return mapping.get(answer, 0)

def score_data_ui(answer):
    return 1 if answer == "Data" else 0.5  # Assuming Data is more technical

def score_skill_level(answer):
    mapping = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
    return mapping.get(answer, 0)

# Updated scoring system with proper weightage
def generate_dataset(num_entries=1000):
    data = []
    interests = list(questions.keys())
    
    for _ in range(num_entries):
        interest = random.choice(interests)
        responses = []
        scores = []
        
        # Generate base skill level (0-5) with more high-scoring examples
        base_skill = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.05, 0.1, 0.15, 0.2, 0.25, 0.25])
        
        for question in questions[interest]:
            if "(Yes/No)" in question:
                # Higher probability of Yes for skilled users
                answer = "Yes" if random.random() < (0.2 + base_skill/10) else "No"
                score = 1 if answer == "Yes" else 0
            elif "(Low/Medium/High)" in question:
                # Skew towards higher levels for skilled users
                rand = random.random() * (base_skill + 1)
                answer = "High" if rand > 3 else "Medium" if rand > 1.5 else "Low"
                score = 2 if answer == "High" else 1 if answer == "Medium" else 0
            elif "(Data/UI)" in question:
                answer = random.choice(["Data", "UI"])
                score = 1 if answer == "Data" else 0.8
            elif "(Beginner/Intermediate/Advanced)" in question:
                rand = random.random() * (base_skill + 1)
                answer = "Advanced" if rand > 3 else "Intermediate" if rand > 1.5 else "Beginner"
                score = 2 if answer == "Advanced" else 1 if answer == "Intermediate" else 0
            
            responses.append(answer)
            scores.append(score)
        
        # Calculate overall score (0-5) with proper weighting
        total_score = min(5, max(0, round(base_skill + np.random.normal(0, 0.3), 1)))
        
        entry = {
            "user_id": fake.uuid4(),
            "interest": interest,
            "calculated_skill_score": total_score,
            **{f"Q{i+1}_score": score for i, score in enumerate(scores)},
            **{f"Q{i+1}_response": resp for i, resp in enumerate(responses)}
        }
        
        data.append(entry)
    
    return pd.DataFrame(data)

# Generate the dataset
dataset = generate_dataset(1000)
dataset.to_csv("user_skill_assessment_dataset_v2.csv", index=False)