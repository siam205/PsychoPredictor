from django.shortcuts import render
import joblib
import pandas as pd
from .models import PredictionRecord

# Load the model once when the server starts
model = joblib.load('ml_model/mental_health_model.pkl')

def home(request):
    return render(request, 'index.html')

def predict(request):
    if request.method == 'POST':
        # Create a dictionary with all the features from the form
        data = {
            'Age': request.POST.get('Age'),
            'Gender': request.POST.get('Gender'),
            'self_employed': request.POST.get('self_employed'),
            'family_history': request.POST.get('family_history'),
            'work_interfere': request.POST.get('work_interfere'),
            'no_employees': request.POST.get('no_employees'),
            'remote_work': request.POST.get('remote_work'),
            'tech_company': request.POST.get('tech_company', 'Yes'),
            'benefits': request.POST.get('benefits', 'Yes'),
            'care_options': request.POST.get('care_options', 'Not sure'),
            'wellness_program': request.POST.get('wellness_program', 'No'),
            'seek_help': request.POST.get('seek_help', 'No'),
            'anonymity': request.POST.get('anonymity', "Don't know"),
            'leave': request.POST.get('leave', 'Somewhat easy'),
            'mental_health_consequence': request.POST.get('mental_health_consequence', 'No'),
            'phys_health_consequence': request.POST.get('phys_health_consequence', 'No'),
            'coworkers': request.POST.get('coworkers', 'Some of them'),
            'supervisor': request.POST.get('supervisor', 'Yes'),
            'mental_health_interview': request.POST.get('mental_health_interview', 'No'),
            'phys_health_interview': request.POST.get('phys_health_interview', 'Maybe'),
            'mental_vs_physical': request.POST.get('mental_vs_physical', 'Yes'),
            'obs_consequence': request.POST.get('obs_consequence', 'No')
        }
        
        input_df = pd.DataFrame([data])
        input_df['Age'] = pd.to_numeric(input_df['Age'])

        prediction_result = model.predict(input_df)[0]
        confidence_proba = model.predict_proba(input_df)[0]
        confidence = confidence_proba[1] * 100

        # --- THIS IS THE CORRECTED PART ---
        # We now explicitly map the dictionary keys (uppercase) to the model fields (lowercase)
        PredictionRecord.objects.create(
            age=data['Age'],
            gender=data['Gender'],
            self_employed=data['self_employed'],
            family_history=data['family_history'],
            work_interfere=data['work_interfere'],
            no_employees=data['no_employees'],
            remote_work=data['remote_work'],
            # ... all other fields would be mapped here as well
            prediction_result=prediction_result,
            confidence_score=confidence
        )
        
        return render(request, 'result.html', {'prediction': prediction_result, 'confidence': confidence})
            
    return render(request, 'index.html')