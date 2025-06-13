from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import hashlib

# Initialize Flask app
app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)
app.secret_key = 'your_secret_key_here'  # Replace with a secure key for production

# Define disease classes and detailed information
disease_classes = ['Bacterial Leaf Blight', 'Brown Spot', 'Leaf Smut', 'Rice Blast']
disease_info = {
    'Bacterial Leaf Blight': {
        'description': 'A bacterial disease caused by Xanthomonas oryzae pv. oryzae, affecting rice plants, especially in warm and humid conditions.',
        'symptoms': [
            'Water-soaked lesions on leaves that turn grayish-white',
            'Leaf wilting and drying',
            'Yellowing along the veins'
        ],
        'management': [
            'Apply copper-based bactericides and remove infected plant debris.',
            'Ensure proper water management to avoid waterlogging.',
            'Use resistant rice varieties if available.',
            'Practice crop rotation with non-host plants.'
        ],
        'video': 'https://www.youtube.com/embed/VbfCNg9CiWQ'  # Example embed URL
    },
    'Brown Spot': {
        'description': 'A fungal disease caused by Cochliobolus miyabeanus, common in nutrient-deficient soils and humid environments.',
        'symptoms': [
            'Small, brown to grayish spots with a yellow halo on leaves',
            'Spots may merge, causing larger dead areas',
            'Reduced photosynthesis and grain yield'
        ],
        'management': [
            'Use a fungicide like mancozeb and improve field drainage.',
            'Apply balanced fertilizers to strengthen plant immunity.',
            'Remove and destroy infected plant residues after harvest.',
            'Avoid dense planting to improve air circulation.'
        ],
        'video': 'https://www.youtube.com/embed/I-P0fXwu1tI'  # Example embed URL
    },
    'Leaf Smut': {
        'description': 'A fungal disease caused by Entyloma oryzae, typically occurring late in the rice growing season.',
        'symptoms': [
            'Small, black spots or smudges on leaves',
            'Spots may enlarge and turn grayish-white',
            'Minimal impact on yield but affects grain quality'
        ],
        'management': [
            'Apply sulfur-based fungicides and avoid overhead irrigation.',
            'Maintain proper field sanitation by removing crop debris.',
            'Use certified disease-free seeds for planting.',
            'Monitor fields regularly for early detection.'
        ],
        'video': 'https://www.youtube.com/embed/U_TlggNfWVs'  # Placeholder (replace with real URL)
    },
    'Rice Blast': {
        'description': 'A highly destructive fungal disease caused by Magnaporthe oryzae, affecting all parts of the rice plant.',
        'symptoms': [
            'Diamond-shaped lesions with gray centers on leaves',
            'Necrosis of leaf veins and panicles',
            'Severe cases lead to complete crop failure'
        ],
        'management': [
            'Use systemic fungicides like tricyclazole and plant resistant varieties.',
            'Avoid excessive nitrogen fertilization.',
            'Implement timely irrigation to prevent drought stress.',
            'Burn infected crop residues to reduce spore spread.'
        ],
        'video': 'https://www.youtube.com/embed/ZCsKmEXC4Y8'  # Example embed URL
    }
}

# In-memory user profile (only one user at a time)
user_profile = None

# Hash password for security
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Prediction function (simplified for demo)
def simple_predict(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mean_hue = np.mean(hsv_image[:, :, 0])
    mean_saturation = np.mean(hsv_image[:, :, 1])
    
    if mean_hue < 50 and mean_saturation > 100:
        return 'Bacterial Leaf Blight'
    elif mean_hue < 100 and mean_saturation < 80:
        return 'Brown Spot'
    elif mean_hue > 100 and mean_saturation < 50:
        return 'Leaf Smut'
    else:
        return 'Rice Blast'

# Routes
@app.route('/')
def home():
    return render_template('login.html')  # Renders the updated login.html

@app.route('/index')
def index():
    if 'email' not in session:
        return redirect(url_for('home'))
    return render_template('index.html')  # Ensure index.html exists

@app.route('/about')
def about():
    if 'email' not in session:
        return redirect(url_for('home'))
    return render_template('about.html')  # Ensure about.html exists

@app.route('/contact')
def contact():
    if 'email' not in session:
        return redirect(url_for('home'))
    return render_template('contact.html')  # Ensure contact.html exists

@app.route('/login', methods=['POST'])
def login():
    global user_profile
    email = request.form.get('email')
    password = request.form.get('password')
    hashed_password = hash_password(password)
    
    if user_profile and user_profile['email'] == email and user_profile['password'] == hashed_password:
        session['email'] = email
        return redirect(url_for('index'))
    else:
        return jsonify({'error': 'Invalid credentials or no user registered'}), 401

@app.route('/register', methods=['POST'])
def register():
    global user_profile
    if user_profile:
        return jsonify({'error': 'Another user is already registered. Only one user allowed at a time.'}), 400
    
    name = request.form.get('names')
    email = request.form.get('emailCreate')
    password = request.form.get('passwordCreate')
    
    if not all([name, email, password]):
        missing = [field for field, value in [('names', name), ('emailCreate', email), ('passwordCreate', password)] if not value]
        return jsonify({'error': f'Missing fields: {", ".join(missing)}'}), 400
    
    hashed_password = hash_password(password)
    user_profile = {
        'name': name,
        'email': email,
        'password': hashed_password
    }
    return redirect(url_for('home'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'email' not in session:
        return jsonify({'error': 'Please log in first'}), 401
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    image = Image.open(file)
    image = image.resize((224, 224))
    image_array = np.array(image)
    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]

    predicted_disease = simple_predict(image_array)
    disease_data = disease_info[predicted_disease]

    return jsonify({
        'disease': predicted_disease,
        'description': disease_data['description'],
        'symptoms': disease_data['symptoms'],
        'management': disease_data['management'],
        'video_suggestion': disease_data['video']
    })

@app.route('/logout')
def logout():
    global user_profile
    session.pop('email', None)
    user_profile = None
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)