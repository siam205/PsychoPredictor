from django.db import models

class PredictionRecord(models.Model):
    # This model will now store all the inputs
    age = models.IntegerField()
    gender = models.CharField(max_length=10)
    self_employed = models.CharField(max_length=5)
    family_history = models.CharField(max_length=5)
    work_interfere = models.CharField(max_length=20)
    no_employees = models.CharField(max_length=20)
    remote_work = models.CharField(max_length=5)
    # ... add all other 15 fields here in the same way, e.g.,
    # tech_company = models.CharField(max_length=5)
    
    # Output fields
    prediction_result = models.CharField(max_length=5)
    confidence_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction for age {self.age} made at {self.created_at.strftime('%Y-%m-%d %H:%M')}"