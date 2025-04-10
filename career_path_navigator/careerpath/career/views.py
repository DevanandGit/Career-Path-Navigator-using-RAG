from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse, JsonResponse
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
import joblib
import os
from .rag_app import get_course_recommendation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "skill_assessment_model.pkl")
model = joblib.load(MODEL_PATH)

if model:
    print("Model connected successfully")
else:
    print("Model connection failed")

# Scoring functions to convert responses to numerical scores
def score_yes_no(answer):
    return 1 if answer.lower() == "yes" else 0

def score_low_med_high(answer):
    mapping = {"low": 0, "medium": 1, "high": 2}
    return mapping.get(answer.lower(), 0)

def score_data_ui(answer):
    return 1 if answer.lower() == "data" else 0.5

def score_skill_level(answer):
    mapping = {"beginner": 0, "intermediate": 1, "advanced": 2}
    return mapping.get(answer.lower(), 0)

user_sessions = {}
@csrf_exempt
def whatsapp_webhook(request):
    if request.method == "POST":
        incoming_message = request.POST.get('Body', '').strip()
        sender = request.POST.get('From', '')
        response = MessagingResponse()

        qualifications = [
            "PCMC", "PCMB", "PCB WITHOUT MATHS", 
            "PLUS-TWO BIO INFORMATICS", "PLUS-TWO PSYCHOLOGY", 
            "HUMANITIES", "COMMERCE"
        ]

        interests_with_questions = [
            "Coding", "Business", "Commerce", "Law", 
            "Medicine", "Architecture", "Journalism", "Social Service"
        ]

        interests_without_questions = [
            "Sports", "Designer", "Hotel Management", "Arts"
        ]

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

        if incoming_message.lower() in ['hello', 'helo', 'hi', 'hey']:
            qualification_options = "\n".join([f"- {q}" for q in qualifications])  
            response.message(f"Welcome! Please enter your qualification from the options below:\n{qualification_options}")
            user_sessions[sender] = {"step": "qualification", "qualifications": []}  # Track user progress
        
        elif sender in user_sessions:
            user_data = user_sessions[sender]

            if user_data["step"] == "qualification":
                if incoming_message.upper() in qualifications:
                    user_data["qualifications"].append(incoming_message.upper())  
                    response.message(f"Got it! Do you want to add another qualification? If yes, enter it. If not, type 'done'.")

                elif incoming_message.lower() == "done":
                    interest_options = "\n".join([f"- {i}" for i in interests_with_questions + interests_without_questions])  
                    response.message(f"Great! Now, select your interest from the following:\n{interest_options}")
                    user_data["step"] = "interest"

                else:
                    response.message("Invalid qualification. Please select from the provided options or type 'done' if finished.")

            elif user_data["step"] == "interest":
                if incoming_message.title() in interests_with_questions:
                    user_data["interest"] = incoming_message.title()
                    user_data["answers"] = []
                    user_data["question_index"] = 0
                    response.message(f"{questions[user_data['interest']][0]}")
                    user_data["step"] = "assessment"

                elif incoming_message.title() in interests_without_questions:
                    user_data["interest"] = incoming_message.title()
                    response.message("Rate yourself in the interest you selected (1-10):")
                    user_data["step"] = "rating"

                else:
                    response.message("Invalid interest. Please choose from the provided options.")

            elif user_data["step"] == "assessment":
                interest = user_data["interest"]
                q_index = user_data["question_index"]
                current_question = questions[interest][q_index]
                
                # Validate and score the response
                try:
                    if "(Yes/No)" in current_question:
                        if incoming_message.lower() not in ["yes", "no"]:
                            raise ValueError("Please answer with Yes or No")
                        score = score_yes_no(incoming_message)
                    elif "(Low/Medium/High)" in current_question:
                        if incoming_message.lower() not in ["low", "medium", "high"]:
                            raise ValueError("Please answer with Low, Medium, or High")
                        score = score_low_med_high(incoming_message)
                    elif "(Data/UI)" in current_question:
                        if incoming_message.lower() not in ["data", "ui"]:
                            raise ValueError("Please answer with Data or UI")
                        score = score_data_ui(incoming_message)
                    elif "(Beginner/Intermediate/Advanced)" in current_question:
                        if incoming_message.lower() not in ["beginner", "intermediate", "advanced"]:
                            raise ValueError("Please answer with Beginner, Intermediate, or Advanced")
                        score = score_skill_level(incoming_message)
                    
                    # Store both response and score
                    user_data["answers"].append({
                        "response": incoming_message,
                        "score": score,
                        "question": current_question
                    })
                    user_data["question_index"] += 1

                    if user_data["question_index"] < len(questions[interest]):
                        response.message(f"{questions[interest][user_data['question_index']]}")
                    else:
                        # Prepare input for the model
                        scores = [ans["score"] for ans in user_data["answers"]]
                        model_input = {
                            f"Q{i+1}_score": score 
                            for i, score in enumerate(scores)
                        }
                        
                        # Get prediction from model
                        predicted_score = model.predict([list(model_input.values())])[0]
                        predicted_score = max(0, min(5, predicted_score))  # Clip to 0-5 range
                        
                        # Convert score to skill level
                        if predicted_score < 2:
                            skill_text = "Beginner"
                        elif predicted_score < 4:
                            skill_text = "Intermediate"
                        else:
                            skill_text = "Advanced"
                            
                        response.message(f"Based on your responses, your skill level in {interest} is: {skill_text} (Score: {predicted_score:.1f}/5).")
                        user_data["step"] = "completed"
                        user_data["skill_text"] = skill_text
                        user_data["predicted_score"] = predicted_score

                except ValueError as e:
                    response.message(str(e) + f"\n\n{current_question}")

            elif user_data["step"] == "rating":
                if incoming_message.isdigit() and 1 <= int(incoming_message) <= 10:
                    rating = int(incoming_message)
                    # Map rating to skill level
                    if rating <= 3:
                        skill_text = "Beginner"
                        predicted_score = 1.0
                    elif rating <= 7:
                        skill_text = "Intermediate"
                        predicted_score = 3.0
                    else:
                        skill_text = "Advanced"
                        predicted_score = 4.5
                        
                    response.message(f"Thank you! You rated yourself {rating}/10 in {user_data['interest']}.")
                    user_data["step"] = "completed"
                    user_data["skill_text"] = skill_text
                    user_data["predicted_score"] = predicted_score
                else:
                    response.message("Please enter a valid number between 1 and 10.")
            
            if user_data["step"] == "completed":
                qualification = user_data["qualifications"][-1]  # Take the last entered qualification
                interest = user_data["interest"]
                skill_text = user_data.get("skill_text", "Beginner")
                predicted_score = user_data.get("predicted_score", 1.0)

                # Fetch career recommendations using RAG
                recommendations = get_course_recommendation(
                    qualification, 
                    interest, 
                    predicted_score  # Pass the actual score instead of just level
                )

                if recommendations:
                    message = "Based on your profile, here are some career recommendations:\n"
                    message += recommendations
                    response.message(message)
                else:
                    response.message("Sorry, we couldn't find a suitable career path for you.")

                # Clear the session or keep it for further interaction
                # user_sessions.pop(sender, None)

        return HttpResponse(str(response), content_type="application/xml")
    return HttpResponse("Invalid request method", status=400)



# View to send a WhatsApp message
def send_message(request):
    # You can define the phone number and message content here
    to_phone_number = '+recepient number'  # Example recipient (replace with a valid number)
    message_body = 'Hello, this is a message from the Django bot!'

    # Call the function to send the message
    send_whatsapp_message(to_phone_number, message_body)

    # Return a response indicating the message was sent
    return JsonResponse({'status': 'Message sent successfully'})

# Function to send WhatsApp message using Twilio
def send_whatsapp_message(to, body):
    account_sid = 'twilio sid'  # Replace with your Twilio SID
    auth_token = ' twilio auth token '  # Replace with your Twilio Auth Token
    from_whatsapp = 'whatsapp:+testnumber'  # Add 'whatsapp:' prefix


    client = Client(account_sid, auth_token)

    message = client.messages.create(
        body=body,
        from_=from_whatsapp,
        to=f'whatsapp:{to}'
    )
    return message.sid
