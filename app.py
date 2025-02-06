from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from googletrans import Translator
import numpy as np
import joblib
from io import BytesIO
import google.generativeai as genai
from dotenv import load_dotenv
import os
from config import (
    MODEL_CONFIG, CLASS_NAMES, LANGUAGE_CODES, 
    FERTILIZER_NAMES, CROP_EMOJI_MAPPING, EMOJI_MAPPING
)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize services
translator = Translator()
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))


# Load all models
def load_models():
    try:
        return {
            'disease_model': tf.keras.models.load_model(MODEL_CONFIG['DISEASE_MODEL_PATH']),
            'crop_model': joblib.load(MODEL_CONFIG['CROP_MODEL_PATH']),
            'label_encoder': joblib.load(MODEL_CONFIG['LABEL_ENCODER_PATH']),
            'fertilizer_model': joblib.load(MODEL_CONFIG['FERTILIZER_MODEL_PATH']),
            'scaler': joblib.load(MODEL_CONFIG['SCALER_PATH'])
        }
    except Exception as e:
        print(f"Error loading models: {e}")
        raise

models = load_models()

# Route handlers
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cropdisease')
def cropdisease():
    return render_template('crop-disease-prediction.html')

@app.route('/croprecommendation')
def croprecommendation():
    return render_template('crop-recommendation.html')

@app.route('/fertilizerrecommendation')
def fertilizerrecommendation():
    return render_template('fertilizer-recommendation.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Process image
        image = load_img(BytesIO(file.read()), target_size=MODEL_CONFIG['IMG_SIZE'])
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)

        # Make prediction
        predictions = models['disease_model'].predict(image_array)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        # Get disease information
        gen_model = genai.GenerativeModel("gemini-1.5-flash")
        class_name = CLASS_NAMES[predicted_class]
        precautions = gen_model.generate_content(
            f"Provide 3-4 bullet points listing only the precautions and cures for {class_name}. "
            "Do not include any additional information or explanations."
        )

        # Create HTML-safe precautions
        precautions_text = precautions.text.replace("\n", "<br>")
        result_text = f"The predicted crop disease is {class_name} with {confidence*100:.2f}% confidence.\n\nRecommended Precautions:"

        return jsonify({
            "predicted_class": class_name,
            "confidence": float(confidence),
            "precautions": precautions_text
        })

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    if request.method == 'POST':
        try:
            # Get form data
            input_data = get_crop_input_data(request.form)
            
            # Make prediction
            prediction = models['crop_model'].predict(input_data)
            crop_name = models['label_encoder'].inverse_transform(prediction)[0]

            # Crop emoji mapping
            CROP_EMOJI_MAPPING = {
                'Rice': '🌾',
                'Maize (Corn)': '🌽',
                'Chickpea': '🫘',
                'Kidney Beans': '🫘',
                'Pigeon Peas': '🫘',
                'Moth Beans': '🫘',
                'Mung Bean': '🫘',
                'Black Gram': '🫘',
                'Lentil': '🫘',
                'Pomegranate': '🍎',
                'Banana': '🍌',
                'Mango': '🥭',
                'Grapes': '🍇',
                'Watermelon': '🍉',
                'Muskmelon': '🍈',
                'Apple': '🍎',
                'Orange': '🍊',
                'Papaya': '🥭',
                'Coconut': '🥥',
                'Cotton': '🌱',
                'Jute': '🪢',
                'Coffee': '☕'
            }

            # Amazon links for each crop
            CROP_LINKS = {
                'Rice': [
                    ('Seeds', 'https://www.amazon.in/s?k=rice+seeds'),
                    ('Fertilizer', 'https://www.amazon.in/s?k=rice+fertilizer'),
                    ('Harvesting Tools', 'https://www.amazon.in/s?k=rice+harvesting+tools')
                ],
                'Maize (Corn)': [
                    ('Seeds', 'https://www.amazon.in/s?k=maize+seeds'),
                    ('Fertilizer', 'https://www.amazon.in/s?k=corn+fertilizer'),
                    ('Planter', 'https://www.amazon.in/s?k=corn+planter+machine')
                ],
                'Chickpea': [
                    ('Seeds', 'https://www.amazon.in/s?k=chickpea+seeds'),
                    ('Fertilizer', 'https://www.amazon.in/s?k=chickpea+fertilizer'),
                    ('Harvester', 'https://www.amazon.in/s?k=chickpea+harvester')
                ],
                'Kidney Beans': [
                    ('Seeds', 'https://www.amazon.in/s?k=kidney+beans+seeds'),
                    ('Fertilizer', 'https://www.amazon.in/s?k=kidney+beans+fertilizer'),
                    ('Storage', 'https://www.amazon.in/s?k=bean+storage+containers')
                ],
                'Pigeon Peas': [
                    ('Seeds', 'https://www.amazon.in/s?k=pigeon+peas+seeds'),
                    ('Fertilizer', 'https://www.amazon.in/s?k=pigeon+peas+fertilizer'),
                    ('Processing', 'https://www.amazon.in/s?k=pulse+processing+machine')
                ],
                'Moth Beans': [
                    ('Seeds', 'https://www.amazon.in/s?k=moth+beans+seeds'),
                    ('Fertilizer', 'https://www.amazon.in/s?k=moth+beans+fertilizer'),
                    ('Sprouting Kit', 'https://www.amazon.in/s?k=bean+sprouting+kit')
                ],
                'Mung Bean': [
                    ('Seeds', 'https://www.amazon.in/s?k=mung+bean+seeds'),
                    ('Fertilizer', 'https://www.amazon.in/s?k=mung+bean+fertilizer'),
                    ('Sprouters', 'https://www.amazon.in/s?k=sprouting+kits')
                ],
                'Black Gram': [
                    ('Seeds', 'https://www.amazon.in/s?k=black+gram+seeds'),
                    ('Fertilizer', 'https://www.amazon.in/s?k=black+gram+fertilizer'),
                    ('Harvester', 'https://www.amazon.in/s?k=gram+harvesting+machine')
                ],
                'Lentil': [
                    ('Seeds', 'https://www.amazon.in/s?k=lentil+seeds'),
                    ('Fertilizer', 'https://www.amazon.in/s?k=lentil+fertilizer'),
                    ('Processing', 'https://www.amazon.in/s?k=lentil+processing+machine')
                ],
                'Pomegranate': [
                    ('Plants', 'https://www.amazon.in/s?k=pomegranate+plants'),
                    ('Fertilizer', 'https://www.amazon.in/s?k=pomegranate+fertilizer'),
                    ('Juicer', 'https://www.amazon.in/s?k=pomegranate+juicer')
                ],
                'Banana': [
                    ('Plants', 'https://www.amazon.in/s?k=banana+plants'),
                    ('Fertilizer', 'https://www.amazon.in/s?k=banana+fertilizer'),
                    ('Harvesting Knife', 'https://www.amazon.in/s?k=banana+harvesting+knife')
                ],
                'Mango': [
                    ('Plants', 'https://www.amazon.in/s?k=mango+plants'),
                    ('Fertilizer', 'https://www.amazon.in/s?k=mango+fertilizer'),
                    ('Grafting Tool', 'https://www.amazon.in/s?k=mango+grafting+tool')
                ],
                'Grapes': [
                    ('Vines', 'https://www.amazon.in/s?k=grape+vines'),
                    ('Fertilizer', 'https://www.amazon.in/s?k=grape+fertilizer'),
                    ('Wine Making', 'https://www.amazon.in/s?k=wine+making+kit')
                ],
                'Watermelon': [
                    ('Seeds', 'https://www.amazon.in/s?k=watermelon+seeds'),
                    ('Fertilizer', 'https://www.amazon.in/s?k=watermelon+fertilizer'),
                    ('Seed Spitter', 'https://www.amazon.in/s?k=watermelon+seed+spitter')
                ],
                'Muskmelon': [
                    ('Seeds', 'https://www.amazon.in/s?k=muskmelon+seeds'),
                    ('Fertilizer', 'https://www.amazon.in/s?k=muskmelon+fertilizer'),
                    ('Trellis', 'https://www.amazon.in/s?k=melon+trellis')
                ],
                'Apple': [
                    ('Trees', 'https://www.amazon.in/s?k=apple+trees'),
                    ('Fertilizer', 'https://www.amazon.in/s?k=apple+tree+fertilizer'),
                    ('Picker', 'https://www.amazon.in/s?k=fruit+picker')
                ],
                'Orange': [
                    ('Trees', 'https://www.amazon.in/s?k=orange+trees'),
                    ('Fertilizer', 'https://www.amazon.in/s?k=orange+tree+fertilizer'),
                    ('Juicer', 'https://www.amazon.in/s?k=orange+juicer')
                ],
                'Papaya': [
                    ('Seeds', 'https://www.amazon.in/s?k=papaya+seeds'),
                    ('Fertilizer', 'https://www.amazon.in/s?k=papaya+fertilizer'),
                    ('Ripening Cover', 'https://www.amazon.in/s?k=fruit+ripening+cover')
                ],
                'Coconut': [
                    ('Seedlings', 'https://www.amazon.in/s?k=coconut+seedlings'),
                    ('Fertilizer', 'https://www.amazon.in/s?k=coconut+fertilizer'),
                    ('Climber', 'https://www.amazon.in/s?k=coconut+tree+climber')
                ],
                'Cotton': [
                    ('Seeds', 'https://www.amazon.in/s?k=cotton+seeds'),
                    ('Fertilizer', 'https://www.amazon.in/s?k=cotton+fertilizer'),
                    ('Ginning', 'https://www.amazon.in/s?k=cotton+ginning+machine')
                ],
                'Jute': [
                    ('Seeds', 'https://www.amazon.in/s?k=jute+seeds'),
                    ('Fertilizer', 'https://www.amazon.in/s?k=jute+fertilizer'),
                    ('Processing', 'https://www.amazon.in/s?k=jute+processing+machine')
                ],
                'Coffee': [
                    ('Plants', 'https://www.amazon.in/s?k=coffee+plants'),
                    ('Fertilizer', 'https://www.amazon.in/s?k=coffee+fertilizer'),
                    ('Grinder', 'https://www.amazon.in/s?k=coffee+grinder')
                ]
            }

            DEFAULT_LINKS = [
                ('Seeds', 'https://www.amazon.in/s?k=agriculture+seeds'),
                ('Fertilizer', 'https://www.amazon.in/s?k=organic+fertilizer'),
                ('Tools', 'https://www.amazon.in/s?k=farming+tools')
            ]

            # Get resources
            emoji = CROP_EMOJI_MAPPING.get(crop_name, '🌱')
            amazon_links = CROP_LINKS.get(crop_name, DEFAULT_LINKS)
            
            # Format prediction
            prediction_text = f"{emoji} Recommended Crop: {crop_name} {emoji}"

            return render_template(
                'crop-recommendation.html',
                prediction_text=prediction_text,
                amazon_links=amazon_links,
                crop_emoji=emoji
            )
        except Exception as e:
            return render_template('crop-recommendation.html',
                                prediction_text=f"Error: {str(e)}")

@app.route('/fertilizer', methods=['GET', 'POST'])
def fertilizer():
    if request.method == 'POST':
        try:
            FERTILIZER_NAMES = {
                0: "Urea",
                1: "DAP",
                2: "MOP",
                3: "Complex",
                4: "SSP",
                5: "Ammonium Sulphate",
                6: "Gypsum"
            }

            FERTILIZER_LINKS = {
                "Urea": "https://www.amazon.in/s?k=urea+fertilizer",
                "DAP": "https://www.amazon.in/s?k=dap+fertilizer",
                "MOP": "https://www.amazon.in/s?k=potash+fertilizer",
                "Complex": "https://www.amazon.in/s?k=npk+fertilizer",
                "SSP": "https://www.amazon.in/s?k=ssp+fertilizer",
                "Ammonium Sulphate": "https://www.amazon.in/s?k=ammonium+sulphate",
                "Gypsum": "https://www.amazon.in/s?k=gypsum+fertilizer",
                "Organic": "https://www.amazon.in/s?k=organic+fertilizer"
            }

            # Process input data
            input_features = get_fertilizer_input_data(request.form)
            input_scaled = models['scaler'].transform(input_features)
            
            # Make prediction
            prediction = models['fertilizer_model'].predict(input_scaled)[0]
            fertilizer_name = FERTILIZER_NAMES.get(prediction, "Unknown")
            
            # Get Amazon link
            amazon_link = FERTILIZER_LINKS.get(
                fertilizer_name, 
                "https://www.amazon.in/s?k=agriculture+fertilizer"
            )

            return render_template('fertilizer-recommendation.html', 
                                result=f'Recommended Fertilizer: {fertilizer_name}',
                                amazon_link=amazon_link)
        except Exception as e:
            return render_template('fertilizer-recommendation.html', 
                                result=f'Error: {str(e)}')
    
    return render_template('fertilizer-recommendation.html', result=None)

@app.route('/translate', methods=['POST'])
def translate_content():
    try:
        data = request.get_json()
        text = data.get('text')
        target_lang = data.get('language')
        
        if not text or not target_lang:
            return jsonify({'error': 'Missing text or language parameter'}), 400
            
        if target_lang not in LANGUAGE_CODES:
            return jsonify({'error': 'Unsupported language'}), 400
            
        translated_text = translate_text(text, LANGUAGE_CODES[target_lang])
        return jsonify({'translated_text': translated_text})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


gen_model2 = genai.GenerativeModel("gemini-1.5-flash")
@app.route("/chat")
def chat_page():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    
    if not user_input:
        return jsonify({"response": "Please provide a message."})
    
    try:
        response = gen_model2.generate_content(user_input)
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})

# Helper functions
def translate_text(text, target_lang):
    try:
        translation = translator.translate(text, dest=target_lang)
        return translation.text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def create_result_html(result_text, precautions):
    return f"""
        <div class="prediction-result">
            <p id="prediction-text">{result_text}</p>
            <ul style="margin-top: 10px; margin-bottom: 15px;">
                {''.join([f'<li>{precaution}</li>' for precaution in precautions])}
            </ul>
            <button onclick="readAloud('prediction-text')" class="read-aloud-btn">
                <i class="fas fa-volume-up"></i> Read Results
            </button>
            <button onclick="stopReading()" class="stop-reading-btn">
                <i class="fas fa-stop"></i> Stop Reading
            </button>
        </div>
    """

def get_crop_input_data(form_data):
    return np.array([[
        float(form_data['nitrogen']),
        float(form_data['phosphorus']),
        float(form_data['potassium']),
        float(form_data['temperature']),
        float(form_data['humidity']),
        float(form_data['ph']),
        float(form_data['rainfall'])
    ]])

def get_fertilizer_input_data(form_data):
    return np.array([[
        float(form_data['temperature']),
        float(form_data['humidity']),
        float(form_data['moisture']),
        int(form_data['soil_type']),
        int(form_data['crop_type']),
        float(form_data['nitrogen']),
        float(form_data['phosphorus']),
        float(form_data['potassium'])
    ]])

def format_crop_prediction(crop_name):
    emoji = CROP_EMOJI_MAPPING.get(crop_name, '')
    return f"The most suitable crop for agriculture in the given conditions is: {emoji} **{crop_name}** {emoji}."

if __name__ == '__main__':
    app.run(debug=os.getenv('FLASK_ENV') == 'development')