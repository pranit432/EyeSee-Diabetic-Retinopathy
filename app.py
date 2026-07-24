import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
from cloudant.client import Cloudant

# ---------------------------------------------------------------------------
# 1. Initialize Flask App & Secret Key
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = "dr_prediction_secret_key"  # Required for managing user sessions

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ---------------------------------------------------------------------------
# 2. Load Saved Model
# ---------------------------------------------------------------------------
# Update path if your model file name is different (e.g., 'model.h5' or 'dr_xception.h5')
MODEL_PATH = "diabetic_retinopathy_xception.h5" 
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Class labels mapping for Diabetic Retinopathy
CLASSES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

# ---------------------------------------------------------------------------
# 3. Cloudant Database Connection Setup
# ---------------------------------------------------------------------------
API_KEY = "IHxulWxvSy_TNtIjAqihkadCZ9xAfSO7ftC8Zw3VnJ6I"
ACCOUNT_NAME = "f8258608-6c52-45dd-b8b0-e3680202a56a-bluemix"
DB_NAME = "diabetic-retinopathy-db"

# Establish persistent Cloudant connection
client = Cloudant.iam(ACCOUNT_NAME, API_KEY, connect=True)

if DB_NAME in client.all_dbs():
    my_database = client[DB_NAME]
else:
    my_database = client.create_database(DB_NAME)

# ---------------------------------------------------------------------------
# 4. Web Routes
# ---------------------------------------------------------------------------

# Home / Index Page
@app.route('/')
def home():
    return render_template('index.html')


# User Registration Route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # Construct query to check if user already exists
        query = {'_id': email}
        docs = my_database.get_query_result(query)

        # Check if user document already exists
        if len(docs.all()) == 0:
            user_data = {
                '_id': email,
                'username': username,
                'password': password
            }
            my_database.create_document(user_data)
            return render_template('login.html', msg="Registration successful! Please login.")
        else:
            return render_template('register.html', msg="User already registered. Please login.")
            
    return render_template('register.html')


# User Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = request.form['email']
        passw = request.form['password']

        query = {'_id': user_id}
        docs = my_database.get_query_result(query)
        doc_list = docs.all()

        if len(doc_list) == 0:
            return render_template('login.html', msg="User ID not found. Please register first.")
        else:
            stored_user = doc_list[0]
            if stored_user['password'] == passw:
                session['user'] = user_id
                return redirect(url_for('predict_page'))
            else:
                return render_template('login.html', msg="Incorrect credentials provided.")
                
    return render_template('login.html')


# Prediction Dashboard & Processing Route
@app.route('/prediction', methods=['GET', 'POST'])
def predict_page():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('prediction.html', msg="No file uploaded.")

        file = request.files['image']
        if file.filename == '':
            return render_template('prediction.html', msg="No image selected.")

        if file:
            # Save uploaded image locally
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Image Preprocessing using Keras (Xception architecture expects 299x299)
            img = image.load_img(file_path, target_size=(299, 299))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # Perform prediction
            preds = model.predict(x)
            pred_class_index = np.argmax(preds, axis=1)[0]
            result = CLASSES[pred_class_index]

            return render_template('prediction.html', prediction=result, filename=file.filename)

    return render_template('prediction.html')


# User Logout Route
@app.route('/logout')
def logout():
    session.pop('user', None)
    return render_template('logout.html')


# ---------------------------------------------------------------------------
# 5. Main Application Entry Point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True, port=5000)