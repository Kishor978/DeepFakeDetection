from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from predict import vids

app = Flask(__name__)

# Define the allowed extensions for uploaded files
ALLOWED_EXTENSIONS = {'mp4'}

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle the form submission and perform prediction"""
    if request.method == 'POST':
        # Check if the file part is present in the request
        if 'file' not in request.files:
            return 'No file part'

        # Get the uploaded file and selected model type
        uploaded_file = request.files['file']
        model_type = request.form['model_type']

        # Check if a file was selected
        if uploaded_file.filename == '':
            return 'No selected file'

        # Check if the file has an allowed extension
        if not allowed_file(uploaded_file.filename):
            return 'Invalid file type'

        # Save the file to a temporary path
        filename = secure_filename(uploaded_file.filename)
        file_path = 'sample_prediction_data/' + filename
        uploaded_file.save(file_path)

        # Call the selected model for prediction
        if model_type == 'autoencoder':
            prediction = vids(file_path, net="ed")
        elif model_type == 'variational':
            prediction = vids(file_path, net="vae")
        else:
            return 'Invalid model type'

        # Render the prediction result on the home page
        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
