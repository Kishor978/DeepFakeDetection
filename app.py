from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from predict import vids
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        # Get the uploaded file
        uploaded_file = request.files['file']
        
        # If the user does not select a file, the browser submits an empty part without filename
        if uploaded_file.filename == '':
            return 'No selected file'
        
        if uploaded_file:
            # Save the file to a temporary path
            filename = secure_filename(uploaded_file.filename)
            file_path = 'sample_prediction_data' + filename
            uploaded_file.save(file_path)
            
        # Call your deepfake detection model for prediction
        # Replace the following line with your model prediction logic
        prediction = vids(file_path)
        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
