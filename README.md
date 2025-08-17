This is a simple web project using some ML concepts. The datasets are collected from Kaggle.

The project's methodology involved first comparing a few different classification models, 
including Logistic Regression, Support Vector Classifier, and Gradient Boosting Classifier, to find the best-performing one.
The Random Forest model was selected for the final pipeline for a few reasons, as explained in 
the notebook and confirmed by its implementation in the training script:
Robustness: The model can handle both numerical and categorical data types, 
which is essential for this dataset. The train_model.py script explicitly prepares both types of data for the model.
Preventing Overfitting: A Random Forest model creates multiple decision trees and averages 
their results to produce a final prediction. This process helps to reduce the risk of overfitting, ensuring the model performs well on new, unseen data.
High Performance: The notebook's goal was to choose the best-performing model, and the train_model.py 
file shows the Random Forest Classifier was selected and fine-tuned with specific parameters (n_estimators=200, max_depth=20).

We used Python, Django for making this project. 

To run this project:
1. You need to install python 3.6 or more advanced versions.
2. If you're using VS code to run this project make sure you have virtual environment set up done.
3. Now open the project folder and open command prompt and write: venv\Scripts\activate .
4. You can now run this project by using: python manage.py runserver .
