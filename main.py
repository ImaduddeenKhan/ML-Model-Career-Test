import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # DON'T CHANGE THIS !!!

from flask import Flask, request, jsonify, render_template, send_file
import pickle
import pandas as pd
import numpy as np
import json
from datetime import datetime
import uuid
import io
from fpdf import FPDF
import random

app = Flask(__name__)

# Load the trained model and encoders
MODEL_PATH = 'final_model.pkl'
FEATURE_ENCODER_PATH = 'feature_encoder.pkl'
LABEL_ENCODER_PATH = 'processed_data/label_encoder.pkl'

# Load the model and encoders
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(FEATURE_ENCODER_PATH, 'rb') as f:
    feature_encoder = pickle.load(f)

with open(LABEL_ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

# Define the questions
QUESTIONS = [
    {"id": "Interest_Work_Type", "text": "What type of work interests you the most?", "options": ["Creative", "Technical", "Helping", "Organizing"]},
    {"id": "New_Ideas_or_Problem_Solving", "text": "Do you prefer coming up with new ideas or solving existing problems?", "options": ["New Ideas", "Problem Solving", "Both Equally"]},
    {"id": "Team_or_Alone", "text": "Do you prefer working in a team or alone?", "options": ["Team", "Alone", "Both"]},
    {"id": "Boring_Task_Response", "text": "When faced with a boring task, what do you usually do?", "options": ["Find ways to make it interesting", "Just get it done quickly", "Procrastinate", "Ask someone else to do it"]},
    {"id": "Logic_Q1", "text": "If all A are B, and all B are C, then all A are C. Is this logical?", "options": ["Yes", "No", "Sometimes"]},
    {"id": "Verbal_Q1", "text": "Which word doesn't belong: Apple, Banana, Carrot, Orange?", "options": ["Apple", "Banana", "Carrot", "Orange"]},
    {"id": "Free_Time_Choice", "text": "What do you prefer to do in your free time?", "options": ["Read/Learn", "Create/Build", "Socialize", "Exercise", "Relax"]},
    {"id": "Friend_In_Trouble", "text": "If a friend is in trouble, what's your first response?", "options": ["Offer advice", "Listen", "Take action to help", "Give them space"]},
    {"id": "Tech_Liking", "text": "Do you enjoy using technology?", "options": ["Yes", "No", "Sometimes"]}
]

# Additional questions for adaptive questioning
ADAPTIVE_QUESTIONS = [
    {"id": "adaptive_1", "text": "How do you handle stress?", "options": ["Take a break", "Push through", "Seek help", "Analyze the cause"]},
    {"id": "adaptive_2", "text": "What's more important to you in a career?", "options": ["Money", "Passion", "Work-life balance", "Growth opportunities"]},
    {"id": "adaptive_3", "text": "How do you approach learning new skills?", "options": ["Structured courses", "Self-teaching", "Learning by doing", "Mentorship"]},
    {"id": "adaptive_4", "text": "What environment do you work best in?", "options": ["Quiet and organized", "Busy and collaborative", "Flexible and changing", "Structured with clear goals"]},
    {"id": "adaptive_5", "text": "How do you make important decisions?", "options": ["Analyze all options", "Go with gut feeling", "Seek advice", "Consider pros and cons"]},
    {"id": "adaptive_6", "text": "What motivates you the most?", "options": ["Recognition", "Personal achievement", "Helping others", "Learning new things"]},
    {"id": "adaptive_7", "text": "How do you handle criticism?", "options": ["Embrace it as feedback", "Get defensive", "Ignore it", "Analyze it carefully"]},
    {"id": "adaptive_8", "text": "What's your approach to planning?", "options": ["Detailed plans", "General direction", "Spontaneous", "Adaptable framework"]},
    {"id": "adaptive_9", "text": "How do you prefer to communicate?", "options": ["Written", "Verbal", "Visual", "Combination"]},
    {"id": "adaptive_10", "text": "What role do you naturally take in a group?", "options": ["Leader", "Supporter", "Innovator", "Mediator"]}
]

# Career facts and insights
CAREER_FACTS = {
    "Analyst": [
        "Analysts typically spend 60% of their time cleaning and organizing data rather than analyzing it.",
        "The demand for data analysts is projected to grow 25% by 2030.",
        "Analysts with programming skills earn 30% more than those without.",
        "Most analysts report that curiosity is their most valuable trait."
    ],
    "Artist": [
        "Professional artists spend an average of 50% of their time on business tasks, not creating art.",
        "Artists who collaborate across disciplines tend to have more sustainable careers.",
        "The most financially successful artists typically have strong entrepreneurial skills.",
        "Artists report higher job satisfaction than many traditional professions despite lower average income."
    ],
    "Athlete": [
        "Professional athletes spend more time on recovery and prevention than actual training.",
        "Most professional athletes develop a second career by their mid-30s.",
        "Athletes with business education earn 40% more in post-sport careers.",
        "Mental training is considered equally important as physical training by elite athletes."
    ],
    "Creator": [
        "Content creators spend an average of 5 hours on production for every 1 hour of content.",
        "Successful creators typically master one platform before expanding to others.",
        "Consistency is rated as more important than talent by top creators.",
        "Most professional creators have 3-5 revenue streams, not just one."
    ],
    "Designer": [
        "Designers spend about 30% of their time communicating with stakeholders, not designing.",
        "The most valued designers can explain the 'why' behind their design decisions.",
        "Design thinking is increasingly being adopted by non-design professions.",
        "Designers who understand business metrics advance more quickly in their careers."
    ],
    "Doctor": [
        "Doctors spend nearly 50% of their time on documentation rather than patient care.",
        "Physicians who maintain hobbies report 30% less burnout.",
        "Most doctors change their specialty focus at least once in their career.",
        "Doctors with communication training have significantly fewer malpractice claims."
    ],
    "Engineer": [
        "Engineers spend about 40% of their time communicating, not building or coding.",
        "Problem definition skills are rated more valuable than problem-solving skills by senior engineers.",
        "Engineers who can explain technical concepts to non-technical people advance faster.",
        "Most engineering innovations come from cross-disciplinary collaboration."
    ],
    "Entrepreneur": [
        "Successful entrepreneurs fail an average of 3.8 times before succeeding.",
        "Most entrepreneurs report that managing people is harder than managing the business.",
        "Entrepreneurs who start with co-founders have a 30% higher success rate than solo founders.",
        "The average entrepreneur works on their idea for 1.5 years before launching."
    ],
    "Journalist": [
        "Journalists typically pitch 5 stories for every one that gets approved.",
        "Most journalists develop expertise in 2-3 specific subject areas throughout their career.",
        "Journalists who can analyze data earn 25% more than those who cannot.",
        "The ability to build trust with sources is rated as the most important skill by editors."
    ],
    "Lawyer": [
        "Lawyers spend only about 30% of their time on actual legal analysis.",
        "Lawyers with technical backgrounds are among the highest paid in the profession.",
        "Most legal professionals change practice areas at least twice in their career.",
        "Negotiation skills correlate more strongly with career advancement than legal knowledge."
    ],
    "Leader": [
        "Leaders spend approximately 70% of their time communicating.",
        "The most effective leaders ask questions twice as often as they give directions.",
        "Leaders who regularly seek feedback are rated 30% more effective by their teams.",
        "Decision-making speed is more strongly correlated with success than decision quality."
    ],
    "Musician": [
        "Professional musicians spend more time on business and promotion than performing.",
        "Musicians with diverse income streams have careers lasting 40% longer than those relying solely on performance.",
        "Most successful musicians collaborate across genres throughout their career.",
        "Musicians who teach report greater career longevity and satisfaction."
    ],
    "Politician": [
        "Politicians spend about 70% of their time fundraising and networking.",
        "Those who master both data analysis and storytelling are most effective at policy change.",
        "Most successful politicians have previous experience in community leadership roles.",
        "Politicians who maintain connections outside politics report better decision-making."
    ],
    "Scientist": [
        "Scientists spend about 60% of their time securing funding and writing reports.",
        "Cross-disciplinary scientists publish more impactful research than specialists.",
        "Most breakthrough discoveries come from questioning established assumptions.",
        "Scientists who can communicate to non-experts have more successful careers."
    ],
    "Teacher": [
        "Teachers make approximately 1,500 educational decisions every school day.",
        "Teachers who regularly update their methods report 40% higher job satisfaction.",
        "The most effective teachers spend more time on student feedback than lesson planning.",
        "Teachers with industry experience bring real-world relevance that improves student outcomes."
    ]
}

# Helper functions for generating career insights
def get_personality_type(answers):
    # Simple mapping based on key answers
    if answers.get('Interest_Work_Type') == 'Creative':
        return "Creative Thinker"
    elif answers.get('Interest_Work_Type') == 'Technical':
        return "Analytical Problem-Solver"
    elif answers.get('Interest_Work_Type') == 'Helping':
        return "Supportive Collaborator"
    elif answers.get('Interest_Work_Type') == 'Organizing':
        return "Structured Organizer"
    else:
        return "Versatile Adapter"

def get_dos_and_donts(personality_type):
    dos_donts = {
        "Creative Thinker": {
            "dos": ["Focus on innovative careers", "Seek environments that value originality", "Collaborate with diverse teams"],
            "donts": ["Avoid rigid, highly structured roles", "Don't suppress your unique perspective", "Avoid environments that resist change"]
        },
        "Analytical Problem-Solver": {
            "dos": ["Pursue roles requiring logical thinking", "Seek data-driven environments", "Develop technical and analytical skills"],
            "donts": ["Avoid purely subjective fields", "Don't rush decisions without analysis", "Avoid roles without intellectual challenge"]
        },
        "Supportive Collaborator": {
            "dos": ["Seek people-centered careers", "Develop strong communication skills", "Work in collaborative environments"],
            "donts": ["Avoid isolated work environments", "Don't neglect your own needs", "Avoid highly competitive cultures"]
        },
        "Structured Organizer": {
            "dos": ["Pursue roles requiring attention to detail", "Develop project management skills", "Seek environments with clear processes"],
            "donts": ["Avoid chaotic work environments", "Don't take on too many tasks at once", "Avoid roles requiring constant improvisation"]
        },
        "Versatile Adapter": {
            "dos": ["Seek varied roles with diverse responsibilities", "Develop a broad skill set", "Look for dynamic environments"],
            "donts": ["Avoid overly specialized roles", "Don't stay in stagnant positions", "Avoid rigid hierarchies"]
        }
    }
    return dos_donts.get(personality_type, {"dos": [], "donts": []})

def generate_personal_insight(answers):
    # Generate a personalized insight based on the combination of answers
    insights = [
        f"Your preference for {answers.get('Interest_Work_Type', 'diverse')} work suggests you thrive when your core values align with your daily tasks.",
        f"The way you approach {answers.get('Boring_Task_Response', 'challenges')} reveals your natural problem-solving style.",
        f"Your preference for working {answers.get('Team_or_Alone', 'with others')} indicates how you process information and make decisions.",
        f"Your response to helping friends shows you value {answers.get('Friend_In_Trouble', 'supportive')} approaches in relationships.",
        f"The way you spend free time on {answers.get('Free_Time_Choice', 'activities')} reflects what truly energizes you."
    ]
    return random.sample(insights, 2)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/questions', methods=['GET'])
def get_questions():
    return jsonify(QUESTIONS)

@app.route('/api/question', methods=['GET'])
def get_next_question():
    # Get the question index from the request
    question_index = request.args.get('index', 0, type=int)
    previous_answers = request.args.get('answers', '{}')
    
    try:
        previous_answers = json.loads(previous_answers)
    except:
        previous_answers = {}
    
    # If we've gone through all standard questions, provide an adaptive question
    if question_index >= len(QUESTIONS):
        # In a real implementation, this would use Gemini API to generate adaptive questions
        # For now, we'll use our predefined adaptive questions
        adaptive_index = (question_index - len(QUESTIONS)) % len(ADAPTIVE_QUESTIONS)
        
        # Simple adaptive logic - choose questions based on previous answers
        if previous_answers.get('Interest_Work_Type') == 'Creative':
            adaptive_index = (adaptive_index + 2) % len(ADAPTIVE_QUESTIONS)
        elif previous_answers.get('Team_or_Alone') == 'Team':
            adaptive_index = (adaptive_index + 1) % len(ADAPTIVE_QUESTIONS)
            
        return jsonify({
            "question": ADAPTIVE_QUESTIONS[adaptive_index],
            "is_last": question_index >= len(QUESTIONS) + 4  # Limit to 5 adaptive questions
        })
    
    # Return the current question
    return jsonify({
        "question": QUESTIONS[question_index],
        "is_last": False
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    # Get the answers from the request
    data = request.json
    answers = data.get('answers', {})
    
    # Prepare the input for the model
    input_data = {}
    
    # Fill in the standard questions
    for question in QUESTIONS:
        question_id = question['id']
        if question_id in answers:
            input_data[question_id] = answers[question_id]
        else:
            # If a question is missing, use a default value
            input_data[question_id] = question['options'][0]
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Encode the input
    input_encoded = feature_encoder.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_encoded)[0]
    prediction_proba = model.predict_proba(input_encoded)[0]
    
    # Get the top 3 predictions
    top_indices = prediction_proba.argsort()[-3:][::-1]
    top_careers = [label_encoder.classes_[i] for i in top_indices]
    top_probabilities = [float(prediction_proba[i]) for i in top_indices]
    
    # Generate personality insights
    personality_type = get_personality_type(answers)
    dos_donts = get_dos_and_donts(personality_type)
    personal_insights = generate_personal_insight(answers)
    
    # Get career facts
    career_facts = []
    for career in top_careers:
        if career in CAREER_FACTS:
            career_facts.append(random.choice(CAREER_FACTS[career]))
    
    # Create a unique session ID for this prediction
    session_id = str(uuid.uuid4())
    
    # Store the results (in a real app, this would go to a database)
    result = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "personality_type": personality_type,
        "dos": dos_donts["dos"],
        "donts": dos_donts["donts"],
        "top_careers": [
            {"career": career, "probability": prob} 
            for career, prob in zip(top_careers, top_probabilities)
        ],
        "career_facts": career_facts,
        "personal_insights": personal_insights
    }
    
    return jsonify(result)

@app.route('/api/generate-pdf', methods=['POST'])
def generate_pdf():
    # Get the prediction results from the request
    data = request.json
    
    # Create a PDF
    pdf = FPDF()
    pdf.add_page()
    
    # Set font
    pdf.set_font("Helvetica", size=12)
    
    # Add title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(200, 10, "Career Guidance Report", ln=True, align='C')
    pdf.ln(10)
    
    # Add personality type
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(200, 10, f"Personality Type: {data.get('personality_type', 'Unknown')}", ln=True)
    pdf.ln(5)
    
    # Add Do's and Don'ts
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(200, 10, "Do's:", ln=True)
    pdf.set_font("Helvetica", "", 12)
    for do_item in data.get('dos', []):
        pdf.cell(200, 10, f"• {do_item}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(200, 10, "Don'ts:", ln=True)
    pdf.set_font("Helvetica", "", 12)
    for dont_item in data.get('donts', []):
        pdf.cell(200, 10, f"• {dont_item}", ln=True)
    pdf.ln(10)
    
    # Add top careers
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(200, 10, "Top Career Matches:", ln=True)
    pdf.ln(5)
    
    for i, career_data in enumerate(data.get('top_careers', [])):
        career = career_data.get('career', 'Unknown')
        probability = career_data.get('probability', 0)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(200, 10, f"{i+1}. {career} ({probability:.1%})", ln=True)
    pdf.ln(10)
    
    # Add career facts
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(200, 10, "Career Insights:", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "", 12)
    for fact in data.get('career_facts', []):
        pdf.multi_cell(0, 10, f"• {fact}")
    pdf.ln(5)
    
    # Add personal insights
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(200, 10, "Personal Insights:", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "", 12)
    for insight in data.get('personal_insights', []):
        pdf.multi_cell(0, 10, f"• {insight}")
    
    # Add footer
    pdf.ln(15)
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(200, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')
    
    # Save the PDF to a bytes buffer
    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)
    
    # Return the PDF as a downloadable file
    return send_file(
        pdf_buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name='career_guidance_report.pdf'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
