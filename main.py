from flask import Flask, request, render_template
import os
import docx2txt
import PyPDF2
import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename
from pypdf import PdfReader

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['CATEGORIZED_FOLDER'] = 'categorized_resumes/'

# Load models for categorization
word_vector = pickle.load(open(r"PROJECT_MAIN/tfidf.pkl", "rb"))
model = pickle.load(open(r"PROJECT_MAIN/model.pkl", "rb"))

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

# Extract text from different file formats
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        return ""

# Helper function to clean resume text for categorization
def clean_resume(txt):
    clean_text = re.sub('http\S+\s', ' ', txt)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+\s', ' ', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', ' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text

# Categorize resumes into predefined categories
def categorize_resumes(uploaded_files, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    results = []
    
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

            category_folder = os.path.join(output_directory, category_name)
            if not os.path.exists(category_folder):
                os.makedirs(category_folder)

            target_path = os.path.join(category_folder, filename)
            with open(target_path, "wb") as f:
                f.write(uploaded_file.read())

            results.append({'filename': filename, 'category': category_name})

    results_df = pd.DataFrame(results)
    return results_df

@app.route("/")
def index():
    return render_template('index.html')

# Resume matcher route
@app.route('/matcher', methods=['POST'])
def matcher():
    if request.method == 'POST':
        job_description = request.form['job_description']
        resume_files = request.files.getlist('resumes')

        resumes = []
        for resume_file in resume_files:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
            resume_file.save(filename)
            resumes.append(extract_text(filename))

        if not resumes or not job_description:
            return render_template('index.html', message="Please upload resumes and enter a job description.")

        # Vectorize job description and resumes
        vectorizer = TfidfVectorizer().fit_transform([job_description] + resumes)
        vectors = vectorizer.toarray()

        # Calculate cosine similarities
        job_vector = vectors[0]
        resume_vectors = vectors[1:]
        similarities = cosine_similarity([job_vector], resume_vectors)[0]

        # Get top 5 resumes and their similarity scores
        top_indices = similarities.argsort()[-5:][::-1]
        top_resumes = [resume_files[i].filename for i in top_indices]
        similarity_scores = [round(similarities[i], 2) for i in top_indices]

        return render_template('index.html', message="Top matching resumes:", top_resumes=top_resumes)

    return render_template('index.html')

# Resume categorizer route
@app.route('/categorize_resumes', methods=['POST'])
def categorize():
    uploaded_files = request.files.getlist('resumes_upload')
    output_directory = request.form['output_directory']

    if not uploaded_files or not output_directory:
        return render_template('index.html', message="Please upload resumes and specify an output directory.")

    categorized_resumes = categorize_resumes(uploaded_files, output_directory)
    return render_template('index.html', categorized_resumes=categorized_resumes.to_dict(orient='records'))

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['CATEGORIZED_FOLDER']):
        os.makedirs(app.config['CATEGORIZED_FOLDER'])
    app.run(debug=True)
