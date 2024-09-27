import os
import pandas as pd
import pickle
import re
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from pypdf import PdfReader

app = Flask(__name__)

# Load models
word_vector = pickle.load(open("tfidf.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# Mapping category IDs to job titles
category_mapping = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and Fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}

# Helper function to clean resume text
def clean_resume(txt):
    clean_text = re.sub('http\S+\s', ' ', txt)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+\s', ' ', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', ' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text

# Route to categorize resumes
@app.route('/categorize_resumes', methods=['POST'])
def categorize_resumes():
    uploaded_files = request.files.getlist('resumes_upload')
    output_directory = request.form['output_directory']

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    categorized_resumes = []

    for uploaded_file in uploaded_files:
        if uploaded_file.filename.endswith('.pdf'):
            filename = secure_filename(uploaded_file.filename)
            reader = PdfReader(uploaded_file)
            page = reader.pages[0]
            text = page.extract_text()
            cleaned_resume = clean_resume(text)

            input_features = word_vector.transform([cleaned_resume])
            prediction_id = model.predict(input_features)[0]
            category_name = category_mapping.get(prediction_id, "Unknown")

            # Create category folder and save the file
            category_folder = os.path.join(output_directory, category_name)
            if not os.path.exists(category_folder):
                os.makedirs(category_folder)

            target_path = os.path.join(category_folder, filename)
            with open(target_path, "wb") as f:
                f.write(uploaded_file.read())

            categorized_resumes.append({
                'filename': filename,
                'category': category_name
            })

    return render_template('index.html', categorized_resumes=categorized_resumes)

# Main route
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
