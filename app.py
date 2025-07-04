# # # # # # # app.py
# # # # # # #
# # # # # # # This script creates a Flask web application to serve our trained
# # # # # # # Tuberculosis detection model. It provides an interface for users
# # # # # # # to upload an image and get a prediction.

# # # # # # import os
# # # # # # import torch
# # # # # # from torch import nn
# # # # # # from flask import Flask, request, jsonify, render_template
# # # # # # from PIL import Image
# # # # # # import logging

# # # # # # # Import our custom modules
# # # # # # from src.model import HybridTBNet
# # # # # # from src.data_setup import get_data_transforms

# # # # # # # --- Configuration ---
# # # # # # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # # # # # # Initialize the Flask app
# # # # # # app = Flask(__name__)

# # # # # # # --- Model and Class Name Loading ---
# # # # # # # Setup paths
# # # # # # PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# # # # # # MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "hybrid_tb_net_v1.pth")

# # # # # # # Setup device
# # # # # # device = "cuda" if torch.cuda.is_available() else "cpu"

# # # # # # # Load the model architecture
# # # # # # model = HybridTBNet(input_channels=1, cnn_output_channels=128).to(device)

# # # # # # # Load the trained model weights
# # # # # # try:
# # # # # #     model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# # # # # #     model.eval()
# # # # # #     logging.info(f"Model loaded successfully from {MODEL_PATH}")
# # # # # # except FileNotFoundError:
# # # # # #     logging.error(f"Model file not found at {MODEL_PATH}. Please ensure the model is trained and saved correctly.")
# # # # # #     model = None # Set model to None if it fails to load

# # # # # # # Define class names
# # # # # # class_names = ["Normal", "Tuberculosis"]

# # # # # # # Get the transformation pipeline for a single image
# # # # # # _, test_transform = get_data_transforms()

# # # # # # def predict_image(image: Image.Image) -> tuple:
# # # # # #     """
# # # # # #     Takes a PIL image, preprocesses it, and returns the model's prediction and confidence.
    
# # # # # #     Args:
# # # # # #         image (Image.Image): The input image to be classified.
    
# # # # # #     Returns:
# # # # # #         tuple: A tuple containing the predicted class name and the confidence score.
# # # # # #     """
# # # # # #     if model is None:
# # # # # #         raise RuntimeError("Model is not loaded. Cannot perform prediction.")

# # # # # #     # Preprocess the image
# # # # # #     # The transform pipeline expects a PIL image
# # # # # #     image_tensor = test_transform(image).unsqueeze(0).to(device)

# # # # # #     # Make a prediction
# # # # # #     with torch.inference_mode():
# # # # # #         logits = model(image_tensor).squeeze()
# # # # # #         # Convert logits to probability using sigmoid
# # # # # #         probability = torch.sigmoid(logits)
    
# # # # # #     # The output of sigmoid is the probability of the positive class ("Tuberculosis")
# # # # # #     # If prob > 0.5, it's TB. Otherwise, it's Normal.
# # # # # #     confidence = probability.item() if probability > 0.5 else 1 - probability.item()
# # # # # #     predicted_class_index = (probability > 0.5).long().item()
# # # # # #     predicted_class_name = class_names[predicted_class_index]
    
# # # # # #     return predicted_class_name, confidence

# # # # # # # --- Flask Routes ---

# # # # # # @app.route('/', methods=['GET'])
# # # # # # def home():
# # # # # #     """
# # # # # #     Renders the main HTML page.
# # # # # #     """
# # # # # #     return render_template('index.html')

# # # # # # @app.route('/predict', methods=['POST'])
# # # # # # def predict():
# # # # # #     """
# # # # # #     Handles the image upload and returns the prediction as JSON.
# # # # # #     """
# # # # # #     if 'file' not in request.files:
# # # # # #         return jsonify({'error': 'No file part in the request'}), 400
    
# # # # # #     file = request.files['file']
    
# # # # # #     if file.filename == '':
# # # # # #         return jsonify({'error': 'No file selected'}), 400
        
# # # # # #     if file:
# # # # # #         try:
# # # # # #             # Open the image file
# # # # # #             image = Image.open(file.stream).convert("RGB")
            
# # # # # #             # Get prediction
# # # # # #             prediction, confidence = predict_image(image)
            
# # # # # #             # Return the result
# # # # # #             return jsonify({
# # # # # #                 'prediction': prediction,
# # # # # #                 'confidence': confidence
# # # # # #             })
            
# # # # # #         except Exception as e:
# # # # # #             logging.error(f"An error occurred during prediction: {e}")
# # # # # #             return jsonify({'error': 'Failed to process the image. Please try again.'}), 500

# # # # # # if __name__ == '__main__':
# # # # # #     # To run the app: flask run
# # # # # #     # Or for development: flask --app app --debug run
# # # # # #     app.run(debug=True)

# # # # # # app.py
# # # # # #
# # # # # # This script creates a Flask web application to serve our trained
# # # # # # Tuberculosis detection model and a helpful chatbot.

# # # # # import os
# # # # # import torch
# # # # # from torch import nn
# # # # # from flask import Flask, request, jsonify, render_template
# # # # # from PIL import Image
# # # # # import logging
# # # # # import requests # Added for chatbot API calls

# # # # # # Import our custom modules
# # # # # from src.model import HybridTBNet
# # # # # from src.data_setup import get_data_transforms

# # # # # # --- Configuration ---
# # # # # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # # # # # Initialize the Flask app
# # # # # app = Flask(__name__)

# # # # # # --- Chatbot Configuration ---
# # # # # OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
# # # # # OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# # # # # # --- Model and Class Name Loading ---
# # # # # # Setup paths
# # # # # PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# # # # # MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "hybrid_tb_net_v1.pth")

# # # # # # Setup device
# # # # # device = "cuda" if torch.cuda.is_available() else "cpu"

# # # # # # Load the model architecture
# # # # # model = HybridTBNet(input_channels=1, cnn_output_channels=128).to(device)

# # # # # # Load the trained model weights
# # # # # try:
# # # # #     model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# # # # #     model.eval()
# # # # #     logging.info(f"Model loaded successfully from {MODEL_PATH}")
# # # # # except FileNotFoundError:
# # # # #     logging.error(f"Model file not found at {MODEL_PATH}. Please ensure the model is trained and saved correctly.")
# # # # #     model = None # Set model to None if it fails to load

# # # # # # Define class names
# # # # # class_names = ["Normal", "Tuberculosis"]

# # # # # # Get the transformation pipeline for a single image
# # # # # _, test_transform = get_data_transforms()

# # # # # def predict_image(image: Image.Image) -> tuple:
# # # # #     """
# # # # #     Takes a PIL image, preprocesses it, and returns the model's prediction and confidence.
# # # # #     """
# # # # #     if model is None:
# # # # #         raise RuntimeError("Model is not loaded. Cannot perform prediction.")

# # # # #     image_tensor = test_transform(image).unsqueeze(0).to(device)

# # # # #     with torch.inference_mode():
# # # # #         logits = model(image_tensor).squeeze()
# # # # #         probability = torch.sigmoid(logits)
    
# # # # #     confidence = probability.item() if probability > 0.5 else 1 - probability.item()
# # # # #     predicted_class_index = (probability > 0.5).long().item()
# # # # #     predicted_class_name = class_names[predicted_class_index]
    
# # # # #     return predicted_class_name, confidence

# # # # # # --- Flask Routes ---

# # # # # @app.route('/', methods=['GET'])
# # # # # def home():
# # # # #     """Renders the main landing page."""
# # # # #     return render_template('index.html')

# # # # # @app.route('/detector', methods=['GET'])
# # # # # def detector():
# # # # #     """Renders the TB Detector page."""
# # # # #     return render_template('detector.html')

# # # # # @app.route('/predict', methods=['POST'])
# # # # # def predict():
# # # # #     """Handles the image upload and returns the prediction as JSON."""
# # # # #     if 'file' not in request.files:
# # # # #         return jsonify({'error': 'No file part in the request'}), 400
    
# # # # #     file = request.files['file']
    
# # # # #     if file.filename == '':
# # # # #         return jsonify({'error': 'No file selected'}), 400
        
# # # # #     if file:
# # # # #         try:
# # # # #             image = Image.open(file.stream).convert("RGB")
# # # # #             prediction, confidence = predict_image(image)
# # # # #             return jsonify({'prediction': prediction, 'confidence': confidence})
# # # # #         except Exception as e:
# # # # #             logging.error(f"An error occurred during prediction: {e}")
# # # # #             return jsonify({'error': 'Failed to process the image. Please try again.'}), 500

# # # # # @app.route('/chat', methods=['POST'])
# # # # # def chat():
# # # # #     """Handles chatbot messages by proxying to OpenRouter API."""
# # # # #     if not OPENROUTER_API_KEY:
# # # # #         return jsonify({'error': 'API key for chatbot is not configured.'}), 500

# # # # #     user_message = request.json.get('message')
# # # # #     if not user_message:
# # # # #         return jsonify({'error': 'No message provided.'}), 400

# # # # #     headers = {
# # # # #         "Authorization": f"Bearer {OPENROUTER_API_KEY}",
# # # # #         "Content-Type": "application/json"
# # # # #     }
    
# # # # #     # A simple system prompt to guide the chatbot
# # # # #     system_prompt = "You are a helpful assistant providing information about Tuberculosis. Be concise and clear. Do not provide medical advice."

# # # # #     data = {
# # # # #         "model": "mistralai/mistral-7b-instruct:free", # Using a capable free model
# # # # #         "messages": [
# # # # #             {"role": "system", "content": system_prompt},
# # # # #             {"role": "user", "content": user_message}
# # # # #         ]
# # # # #     }

# # # # #     try:
# # # # #         response = requests.post(OPENROUTER_API_URL, headers=headers, json=data)
# # # # #         response.raise_for_status() # Raise an exception for bad status codes
# # # # #         bot_response = response.json()['choices'][0]['message']['content']
# # # # #         return jsonify({'reply': bot_response})
# # # # #     except requests.exceptions.RequestException as e:
# # # # #         logging.error(f"Error calling OpenRouter API: {e}")
# # # # #         return jsonify({'error': 'Failed to get a response from the chatbot.'}), 502

# # # # # if __name__ == '__main__':
# # # # #     app.run(debug=True)
# # # # # app.py
# # # # #
# # # # # This script creates a Flask web application to serve our trained
# # # # # Tuberculosis detection model and a helpful chatbot.

# # # # import os
# # # # import torch
# # # # from torch import nn
# # # # from flask import Flask, request, jsonify, render_template
# # # # from PIL import Image
# # # # import logging
# # # # import requests # For chatbot API calls

# # # # # Import our custom modules
# # # # from src.model import HybridTBNet
# # # # from src.data_setup import get_data_transforms

# # # # # --- Configuration ---
# # # # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # # # # Initialize the Flask app
# # # # app = Flask(__name__)

# # # # # --- Chatbot Configuration ---
# # # # OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
# # # # OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# # # # # --- Model and Class Name Loading ---
# # # # # Setup paths
# # # # PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# # # # MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "hybrid_tb_net_v1.pth")

# # # # # Setup device
# # # # device = "cuda" if torch.cuda.is_available() else "cpu"

# # # # # Load the model architecture
# # # # model = HybridTBNet(input_channels=1, cnn_output_channels=128).to(device)

# # # # # Load the trained model weights
# # # # try:
# # # #     model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# # # #     model.eval()
# # # #     logging.info(f"Model loaded successfully from {MODEL_PATH}")
# # # # except FileNotFoundError:
# # # #     logging.error(f"Model file not found at {MODEL_PATH}. Please ensure the model is trained and saved correctly.")
# # # #     model = None # Set model to None if it fails to load

# # # # # Define class names
# # # # class_names = ["Normal", "Tuberculosis"]

# # # # # Get the transformation pipeline for a single image
# # # # _, test_transform = get_data_transforms()

# # # # def predict_image(image: Image.Image) -> tuple:
# # # #     """
# # # #     Takes a PIL image, preprocesses it, and returns the model's prediction and confidence.
# # # #     """
# # # #     if model is None:
# # # #         raise RuntimeError("Model is not loaded. Cannot perform prediction.")

# # # #     image_tensor = test_transform(image).unsqueeze(0).to(device)

# # # #     with torch.inference_mode():
# # # #         logits = model(image_tensor).squeeze()
# # # #         probability = torch.sigmoid(logits)
    
# # # #     confidence = probability.item() if probability > 0.5 else 1 - probability.item()
# # # #     predicted_class_index = (probability > 0.5).long().item()
# # # #     predicted_class_name = class_names[predicted_class_index]
    
# # # #     return predicted_class_name, confidence

# # # # # --- Flask Routes ---

# # # # @app.route('/', methods=['GET'])
# # # # def home():
# # # #     """Renders the main Chatbot page."""
# # # #     return render_template('index.html')

# # # # @app.route('/detector', methods=['GET'])
# # # # def detector():
# # # #     """Renders the TB Detector page."""
# # # #     return render_template('detector.html')

# # # # @app.route('/predict', methods=['POST'])
# # # # def predict():
# # # #     """Handles the image upload and returns the prediction as JSON."""
# # # #     if 'file' not in request.files:
# # # #         return jsonify({'error': 'No file part in the request'}), 400
    
# # # #     file = request.files['file']
    
# # # #     if file.filename == '':
# # # #         return jsonify({'error': 'No file selected'}), 400
        
# # # #     if file:
# # # #         try:
# # # #             image = Image.open(file.stream).convert("RGB")
# # # #             prediction, confidence = predict_image(image)
# # # #             return jsonify({'prediction': prediction, 'confidence': confidence})
# # # #         except Exception as e:
# # # #             logging.error(f"An error occurred during prediction: {e}")
# # # #             return jsonify({'error': 'Failed to process the image. Please try again.'}), 500

# # # # @app.route('/chat', methods=['POST'])
# # # # def chat():
# # # #     """Handles chatbot messages by proxying to OpenRouter API."""
# # # #     if not OPENROUTER_API_KEY:
# # # #         return jsonify({'error': 'API key for chatbot is not configured. Please set the OPENROUTER_API_KEY environment variable.'}), 500

# # # #     user_message = request.json.get('message')
# # # #     if not user_message:
# # # #         return jsonify({'error': 'No message provided.'}), 400

# # # #     headers = {
# # # #         "Authorization": f"Bearer {OPENROUTER_API_KEY}",
# # # #         "Content-Type": "application/json"
# # # #     }
    
# # # #     system_prompt = "You are a helpful AI assistant for a Tuberculosis Detection project. Your name is 'Aura'. You provide clear, concise information about Tuberculosis but you MUST refuse to give medical advice, diagnoses, or treatment plans, instead directing the user to consult a healthcare professional. Be friendly and supportive."

# # # #     data = {
# # # #         "model": "mistralai/mistral-7b-instruct:free",
# # # #         "messages": [
# # # #             {"role": "system", "content": system_prompt},
# # # #             {"role": "user", "content": user_message}
# # # #         ]
# # # #     }

# # # #     try:
# # # #         response = requests.post(OPENROUTER_API_URL, headers=headers, json=data)
# # # #         response.raise_for_status()
# # # #         bot_response = response.json()['choices'][0]['message']['content']
# # # #         return jsonify({'reply': bot_response})
# # # #     except requests.exceptions.RequestException as e:
# # # #         logging.error(f"Error calling OpenRouter API: {e}")
# # # #         return jsonify({'error': 'Failed to get a response from the chatbot.'}), 502

# # # # if __name__ == '__main__':
# # # #     app.run(debug=True)
# # # # app.py
# # # #
# # # # This script creates a Flask web application to serve our trained
# # # # Tuberculosis detection model and a helpful chatbot.

# # # import os
# # # import torch
# # # from torch import nn
# # # from flask import Flask, request, jsonify, render_template
# # # from PIL import Image
# # # import logging
# # # import requests # For chatbot API calls
# # # from dotenv import load_dotenv # Import dotenv

# # # # Load environment variables from .env file
# # # load_dotenv()

# # # # Import our custom modules
# # # from src.model import HybridTBNet
# # # from src.data_setup import get_data_transforms

# # # # --- Configuration ---
# # # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # # # Initialize the Flask app
# # # app = Flask(__name__)

# # # # --- Chatbot Configuration ---
# # # OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
# # # OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# # # # --- Model and Class Name Loading ---
# # # # Setup paths
# # # PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# # # MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "hybrid_tb_net_v1.pth")

# # # # Setup device
# # # device = "cuda" if torch.cuda.is_available() else "cpu"

# # # # Load the model architecture
# # # model = HybridTBNet(input_channels=1, cnn_output_channels=128).to(device)

# # # # Load the trained model weights
# # # try:
# # #     model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# # #     model.eval()
# # #     logging.info(f"Model loaded successfully from {MODEL_PATH}")
# # # except FileNotFoundError:
# # #     logging.error(f"Model file not found at {MODEL_PATH}. Please ensure the model is trained and saved correctly.")
# # #     model = None # Set model to None if it fails to load

# # # # Define class names
# # # class_names = ["Normal", "Tuberculosis"]

# # # # Get the transformation pipeline for a single image
# # # _, test_transform = get_data_transforms()

# # # def predict_image(image: Image.Image) -> tuple:
# # #     """
# # #     Takes a PIL image, preprocesses it, and returns the model's prediction and confidence.
# # #     """
# # #     if model is None:
# # #         raise RuntimeError("Model is not loaded. Cannot perform prediction.")

# # #     image_tensor = test_transform(image).unsqueeze(0).to(device)

# # #     with torch.inference_mode():
# # #         logits = model(image_tensor).squeeze()
# # #         probability = torch.sigmoid(logits)
    
# # #     confidence = probability.item() if probability > 0.5 else 1 - probability.item()
# # #     predicted_class_index = (probability > 0.5).long().item()
# # #     predicted_class_name = class_names[predicted_class_index]
    
# # #     return predicted_class_name, confidence

# # # # --- Flask Routes ---

# # # @app.route('/', methods=['GET'])
# # # def home():
# # #     """Renders the main Chatbot page."""
# # #     return render_template('index.html')

# # # @app.route('/detector', methods=['GET'])
# # # def detector():
# # #     """Renders the TB Detector page."""
# # #     return render_template('detector.html')

# # # @app.route('/predict', methods=['POST'])
# # # def predict():
# # #     """Handles the image upload and returns the prediction as JSON."""
# # #     if 'file' not in request.files:
# # #         return jsonify({'error': 'No file part in the request'}), 400
    
# # #     file = request.files['file']
    
# # #     if file.filename == '':
# # #         return jsonify({'error': 'No file selected'}), 400
        
# # #     if file:
# # #         try:
# # #             image = Image.open(file.stream).convert("RGB")
# # #             prediction, confidence = predict_image(image)
# # #             return jsonify({'prediction': prediction, 'confidence': confidence})
# # #         except Exception as e:
# # #             logging.error(f"An error occurred during prediction: {e}")
# # #             return jsonify({'error': 'Failed to process the image. Please try again.'}), 500

# # # @app.route('/chat', methods=['POST'])
# # # def chat():
# # #     """Handles chatbot messages by proxying to OpenRouter API."""
# # #     if not OPENROUTER_API_KEY:
# # #         return jsonify({'error': 'API key for chatbot is not configured. Please create a .env file and add your OPENROUTER_API_KEY.'}), 500

# # #     user_message = request.json.get('message')
# # #     if not user_message:
# # #         return jsonify({'error': 'No message provided.'}), 400

# # #     headers = {
# # #         "Authorization": f"Bearer {OPENROUTER_API_KEY}",
# # #         "Content-Type": "application/json"
# # #     }
    
# # #     system_prompt = "You are a helpful AI assistant for a Tuberculosis Detection project. Your name is 'Aura'. You provide clear, concise information about Tuberculosis but you MUST refuse to give medical advice, diagnoses, or treatment plans, instead directing the user to consult a healthcare professional. Be friendly and supportive."

# # #     data = {
# # #         "model": "mistralai/mistral-7b-instruct:free",
# # #         "messages": [
# # #             {"role": "system", "content": system_prompt},
# # #             {"role": "user", "content": user_message}
# # #         ]
# # #     }

# # #     try:
# # #         response = requests.post(OPENROUTER_API_URL, headers=headers, json=data)
# # #         response.raise_for_status()
# # #         bot_response = response.json()['choices'][0]['message']['content']
# # #         return jsonify({'reply': bot_response})
# # #     except requests.exceptions.RequestException as e:
# # #         logging.error(f"Error calling OpenRouter API: {e}")
# # #         return jsonify({'error': 'Failed to get a response from the chatbot.'}), 502

# # # if __name__ == '__main__':
# # #     app.run(debug=True)
# # # app.py
# # #
# # # This script creates a Flask web application to serve our trained
# # # Tuberculosis detection model and a helpful chatbot.

# # import os
# # import torch
# # from torch import nn
# # from flask import Flask, request, jsonify, render_template
# # from PIL import Image
# # import logging
# # import requests # For chatbot API calls
# # from dotenv import load_dotenv # Import dotenv

# # # Load environment variables from .env file
# # load_dotenv()

# # # Import our custom modules
# # from src.model import HybridTBNet
# # from src.data_setup import get_data_transforms

# # # --- Configuration ---
# # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # # Initialize the Flask app
# # app = Flask(__name__)

# # # --- Chatbot Configuration ---
# # OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
# # OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# # # --- Model and Class Name Loading ---
# # # Setup paths
# # PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# # MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "hybrid_tb_net_v1.pth")

# # # Setup device
# # device = "cuda" if torch.cuda.is_available() else "cpu"

# # # Load the model architecture
# # model = HybridTBNet(input_channels=1, cnn_output_channels=128).to(device)

# # # Load the trained model weights
# # try:
# #     model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# #     model.eval()
# #     logging.info(f"Model loaded successfully from {MODEL_PATH}")
# # except FileNotFoundError:
# #     logging.error(f"Model file not found at {MODEL_PATH}. Please ensure the model is trained and saved correctly.")
# #     model = None # Set model to None if it fails to load

# # # Define class names
# # class_names = ["Normal", "Tuberculosis"]

# # # Get the transformation pipeline for a single image
# # _, test_transform = get_data_transforms()

# # def predict_image(image: Image.Image) -> tuple:
# #     """
# #     Takes a PIL image, preprocesses it, and returns the model's prediction and confidence.
# #     """
# #     if model is None:
# #         raise RuntimeError("Model is not loaded. Cannot perform prediction.")

# #     image_tensor = test_transform(image).unsqueeze(0).to(device)

# #     with torch.inference_mode():
# #         logits = model(image_tensor).squeeze()
# #         probability = torch.sigmoid(logits)
    
# #     confidence = probability.item() if probability > 0.5 else 1 - probability.item()
# #     predicted_class_index = (probability > 0.5).long().item()
# #     predicted_class_name = class_names[predicted_class_index]
    
# #     return predicted_class_name, confidence

# # # --- Flask Routes ---

# # @app.route('/', methods=['GET'])
# # def home():
# #     """Renders the main Chatbot page."""
# #     return render_template('index.html')

# # @app.route('/detector', methods=['GET'])
# # def detector():
# #     """Renders the TB Detector page."""
# #     return render_template('detector.html')

# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     """Handles the image upload and returns the prediction as JSON."""
# #     if 'file' not in request.files:
# #         return jsonify({'error': 'No file part in the request'}), 400
    
# #     file = request.files['file']
    
# #     if file.filename == '':
# #         return jsonify({'error': 'No file selected'}), 400
        
# #     if file:
# #         try:
# #             image = Image.open(file.stream).convert("RGB")
# #             prediction, confidence = predict_image(image)
# #             return jsonify({'prediction': prediction, 'confidence': confidence})
# #         except Exception as e:
# #             logging.error(f"An error occurred during prediction: {e}")
# #             return jsonify({'error': 'Failed to process the image. Please try again.'}), 500

# # @app.route('/chat', methods=['POST'])
# # def chat():
# #     """Handles chatbot messages by proxying to OpenRouter API."""
# #     if not OPENROUTER_API_KEY:
# #         return jsonify({'error': 'API key for chatbot is not configured. Please create a .env file and add your OPENROUTER_API_KEY.'}), 500

# #     user_message = request.json.get('message')
# #     if not user_message:
# #         return jsonify({'error': 'No message provided.'}), 400

# #     headers = {
# #         "Authorization": f"Bearer {OPENROUTER_API_KEY}",
# #         "Content-Type": "application/json"
# #     }
    
# #     # --- FIX: Updated system prompt for stricter topic relevance ---
# #     system_prompt = """You are an AI assistant named Aura for a Tuberculosis (TB) analysis project. Your ONLY function is to provide information about Tuberculosis.
# # - You MUST answer questions about TB symptoms, transmission, prevention, and general facts.
# # - You MUST use Markdown for formatting your answers (e.g., lists, bold text).
# # - If the user asks for medical advice, a diagnosis, or a treatment plan, you MUST politely refuse and advise them to consult a real healthcare professional.
# # - If the user asks a question NOT related to Tuberculosis, you MUST politely refuse and state that you can only answer questions about TB.
# # - Be friendly, concise, and supportive in your tone."""

# #     data = {
# #         "model": "mistralai/mistral-7b-instruct:free",
# #         "messages": [
# #             {"role": "system", "content": system_prompt},
# #             {"role": "user", "content": user_message}
# #         ]
# #     }

# #     try:
# #         response = requests.post(OPENROUTER_API_URL, headers=headers, json=data)
# #         response.raise_for_status()
# #         bot_response = response.json()['choices'][0]['message']['content']
# #         return jsonify({'reply': bot_response})
# #     except requests.exceptions.RequestException as e:
# #         logging.error(f"Error calling OpenRouter API: {e}")
# #         return jsonify({'error': 'Failed to get a response from the chatbot.'}), 502

# # if __name__ == '__main__':
# #     app.run(debug=True)
# # app.py
# #
# # This script creates a Flask web application to serve our trained
# # Tuberculosis detection model and a helpful chatbot.

# import os
# import torch
# from torch import nn
# from flask import Flask, request, jsonify, render_template
# from PIL import Image
# import logging
# import requests # For chatbot API calls
# from dotenv import load_dotenv # Import dotenv

# # Load environment variables from .env file
# load_dotenv()

# # Import our custom modules
# from src.model import HybridTBNet
# from src.data_setup import get_data_transforms

# # --- Configuration ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Initialize the Flask app
# app = Flask(__name__)

# # --- Chatbot Configuration ---
# OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
# OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# # --- Model and Class Name Loading ---
# # Setup paths
# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "hybrid_tb_net_v1.pth")

# # Setup device
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Load the model architecture
# model = HybridTBNet(input_channels=1, cnn_output_channels=128).to(device)

# # Load the trained model weights
# try:
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
#     model.eval()
#     logging.info(f"Model loaded successfully from {MODEL_PATH}")
# except FileNotFoundError:
#     logging.error(f"Model file not found at {MODEL_PATH}. Please ensure the model is trained and saved correctly.")
#     model = None # Set model to None if it fails to load

# # Define class names
# class_names = ["Normal", "Tuberculosis"]

# # Get the transformation pipeline for a single image
# _, test_transform = get_data_transforms()

# def predict_image(image: Image.Image) -> tuple:
#     """
#     Takes a PIL image, preprocesses it, and returns the model's prediction and confidence.
#     """
#     if model is None:
#         raise RuntimeError("Model is not loaded. Cannot perform prediction.")

#     image_tensor = test_transform(image).unsqueeze(0).to(device)

#     with torch.inference_mode():
#         logits = model(image_tensor).squeeze()
#         probability = torch.sigmoid(logits)
    
#     confidence = probability.item() if probability > 0.5 else 1 - probability.item()
#     predicted_class_index = (probability > 0.5).long().item()
#     predicted_class_name = class_names[predicted_class_index]
    
#     return predicted_class_name, confidence

# # --- Flask Routes ---

# @app.route('/', methods=['GET'])
# def home():
#     """Renders the main Chatbot page."""
#     return render_template('index.html')

# @app.route('/detector', methods=['GET'])
# def detector():
#     """Renders the TB Detector page."""
#     return render_template('detector.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     """Handles the image upload and returns the prediction as JSON."""
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part in the request'}), 400
    
#     file = request.files['file']
    
#     if file.filename == '':
#         return jsonify({'error': 'No file selected'}), 400
        
#     if file:
#         try:
#             image = Image.open(file.stream).convert("RGB")
#             prediction, confidence = predict_image(image)
#             return jsonify({'prediction': prediction, 'confidence': confidence})
#         except Exception as e:
#             logging.error(f"An error occurred during prediction: {e}")
#             return jsonify({'error': 'Failed to process the image. Please try again.'}), 500

# @app.route('/chat', methods=['POST'])
# def chat():
#     """Handles chatbot messages by proxying to OpenRouter API."""
#     if not OPENROUTER_API_KEY:
#         return jsonify({'error': 'API key for chatbot is not configured. Please create a .env file and add your OPENROUTER_API_KEY.'}), 500

#     user_message = request.json.get('message')
#     if not user_message:
#         return jsonify({'error': 'No message provided.'}), 400

#     headers = {
#         "Authorization": f"Bearer {OPENROUTER_API_KEY}",
#         "Content-Type": "application/json"
#     }
    
#     # Updated system prompt for stricter topic relevance and behavior
#     system_prompt = """You are Aura, a specialized AI assistant for a Tuberculosis (TB) analysis project. Your knowledge is strictly limited to Tuberculosis.

# **Core Directives:**
# 1.  **Topic Relevance:** Your ONLY function is to answer questions directly related to Tuberculosis (e.g., symptoms, transmission, prevention, general facts).
# 2.  **Refusal Protocol:** If a user asks a question about ANY other topic, including other medical conditions, general knowledge, or personal questions, you MUST respond with a polite refusal and nothing more. A good refusal is: "I'm sorry, but as Aura, my expertise is strictly limited to Tuberculosis. I cannot provide information on other topics. How can I help you with TB?" Do NOT answer the off-topic question.
# 3.  **Medical Advice:** You are NOT a medical professional. If a user asks for a diagnosis, treatment plan, or personal medical advice, you MUST refuse and strongly advise them to consult a qualified healthcare provider.
# 4.  **Formatting:** Always use Markdown for clear formatting (lists, bolding, etc.).
# """

#     data = {
#         "model": "mistralai/mistral-7b-instruct:free",
#         "messages": [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_message}
#         ]
#     }

#     try:
#         response = requests.post(OPENROUTER_API_URL, headers=headers, json=data)
#         response.raise_for_status()
#         bot_response = response.json()['choices'][0]['message']['content']
#         return jsonify({'reply': bot_response})
#     except requests.exceptions.RequestException as e:
#         logging.error(f"Error calling OpenRouter API: {e}")
#         return jsonify({'error': 'Failed to get a response from the chatbot.'}), 502

# if __name__ == '__main__':
#     app.run(debug=True)
# app.py
#
# This script creates a Flask web application to serve our trained
# Tuberculosis detection model and a helpful chatbot.

import os
import torch
from torch import nn
from flask import Flask, request, jsonify, render_template
from PIL import Image
import logging
import requests # For chatbot API calls
from dotenv import load_dotenv # Import dotenv

# Load environment variables from .env file
load_dotenv()

# Import our custom modules
from src.model import HybridTBNet
from src.data_setup import get_data_transforms

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the Flask app
app = Flask(__name__)

# --- Chatbot Configuration ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# --- Model and Class Name Loading ---
# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "hybrid_tb_net_v1.pth")

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model architecture
model = HybridTBNet(input_channels=1, cnn_output_channels=128).to(device)

# Load the trained model weights
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    logging.info(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    logging.error(f"Model file not found at {MODEL_PATH}. Please ensure the model is trained and saved correctly.")
    model = None # Set model to None if it fails to load

# Define class names
class_names = ["Normal", "Tuberculosis"]

# Get the transformation pipeline for a single image
_, test_transform = get_data_transforms()

def predict_image(image: Image.Image) -> tuple:
    """
    Takes a PIL image, preprocesses it, and returns the model's prediction and confidence.
    """
    if model is None:
        raise RuntimeError("Model is not loaded. Cannot perform prediction.")

    image_tensor = test_transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(image_tensor).squeeze()
        probability = torch.sigmoid(logits)
    
    confidence = probability.item() if probability > 0.5 else 1 - probability.item()
    predicted_class_index = (probability > 0.5).long().item()
    predicted_class_name = class_names[predicted_class_index]
    
    return predicted_class_name, confidence

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def home():
    """Renders the main Chatbot page."""
    return render_template('index.html')

@app.route('/detector', methods=['GET'])
def detector():
    """Renders the TB Detector page."""
    return render_template('detector.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the image upload and returns the prediction as JSON."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    if file:
        try:
            image = Image.open(file.stream).convert("RGB")
            prediction, confidence = predict_image(image)
            return jsonify({'prediction': prediction, 'confidence': confidence})
        except Exception as e:
            logging.error(f"An error occurred during prediction: {e}")
            return jsonify({'error': 'Failed to process the image. Please try again.'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Handles chatbot messages by proxying to OpenRouter API."""
    if not OPENROUTER_API_KEY:
        return jsonify({'error': 'API key for chatbot is not configured. Please create a .env file and add your OPENROUTER_API_KEY.'}), 500

    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'error': 'No message provided.'}), 400

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Updated system prompt for stricter topic relevance and behavior
    system_prompt = """You are Aura, a specialized AI assistant for a Tuberculosis (TB) analysis project. Your knowledge is strictly limited to Tuberculosis.

**Core Directives:**
1.  **Topic Relevance:** Your ONLY function is to answer questions directly related to Tuberculosis (e.g., symptoms, transmission, prevention, general facts).
2.  **Refusal Protocol:** If a user asks a question about ANY other topic, including other medical conditions, general knowledge, or personal questions, you MUST respond with a polite refusal and nothing more. A good refusal is: "I'm sorry, but as Aura, my expertise is strictly limited to Tuberculosis. I cannot provide information on other topics. How can I help you with TB?" Do NOT answer the off-topic question.
3.  **Medical Advice:** You are NOT a medical professional. If a user asks for a diagnosis, treatment plan, or personal medical advice, you MUST refuse and strongly advise them to consult a qualified healthcare provider.
4.  **Formatting:** Always use Markdown for clear formatting (lists, bolding, etc.).
"""

    data = {
        "model": "mistralai/mistral-7b-instruct:free",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    }

    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=data)
        response.raise_for_status()
        bot_response = response.json()['choices'][0]['message']['content']
        return jsonify({'reply': bot_response})
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling OpenRouter API: {e}")
        return jsonify({'error': 'Failed to get a response from the chatbot.'}), 502

if __name__ == '__main__':
    app.run(debug=True)
