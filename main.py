import pandas as pd
from flask import Flask, jsonify, request
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from decouple import config
from flask_cors import CORS
# from flasgger import Swagger, swag_from
import bcrypt
import json
import os

app = Flask(__name__)
CORS(app)

app.config['JWT_SECRET_KEY'] = config('JWT_SECRET_KEY', default='secretkey123')
# app.config['SWAGGER'] = {
#     'title': 'SSStudio API',
#     'uiversion': 3
# }
jwt = JWTManager(app)
# swagger = Swagger(app)

users = {}

@app.route('/register', methods=['POST'])
def register():
    req = request.get_json()
    username = req.get('username')
    password = req.get('password')

    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400
    if username in users:
        return jsonify({"error": "Username already exists"}), 409

    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    users[username] = hashed
    return jsonify({"msg": "User registered"}), 201

@app.route('/login', methods=['POST'])
def login():
    req = request.get_json()
    username = req.get('username')
    password = req.get('password')

    hashed = users.get(username)
    if not hashed or not bcrypt.checkpw(password.encode('utf-8'), hashed):
        return jsonify({"error": "Invalid credentials"}), 401

    access_token = create_access_token(identity=username)
    return jsonify({'token': access_token}), 200

@app.route('/ping', methods=['GET'])
def ping():
    return {"status": "ok"}, 200

@app.route('/test', methods=['GET'])
@jwt_required()
def test():
    current_user = get_jwt_identity()
    return jsonify({"msg": current_user}), 200

@app.route('/anonymize', methods=['POST'])
@jwt_required()
def anonymize_data():
    request_data = request.get_json()
    if 'data' not in request_data or 'configs' not in request_data:
        return jsonify({"error": "Missing 'data' or 'configs'"}), 400
    configs = request_data['configs']
    if not isinstance(configs, dict) or 'model' not in configs:
        return jsonify({"error": "'configs' not formatted correctly"}), 400
    model = configs.get('model')
    data = pd.DataFrame(request_data['data'])
    try:
        if model.lower() == 'health':
            from models.healthcare_models import anonymize_health_data
            anonymized_data = anonymize_health_data(data)
        elif model.lower() == 'finance':
            from models.finance_models import anonymize_finance_data
            anonymized_data = anonymize_finance_data(data)
        elif model.lower() == 'education':
            from models.education_models import anonymize_education_data
            anonymized_data = anonymize_education_data(data)
        else:
            return jsonify({"error": "Unsupported model for anonymization"}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred during anonymization: {str(e)}"}), 500
    return jsonify({"anonymized_data": anonymized_data.to_json(orient='records')}), 200

@app.route('/synthesize', methods=['POST'])
@jwt_required()
def synthesize_data():
    request_data = request.get_json()
    if 'data' not in request_data or 'configs' not in request_data:
        return jsonify({"error": "Missing 'data' or 'configs'"}), 400
    configs = request_data['configs']
    if not isinstance(configs, dict) or 'model' not in configs:
        return jsonify({"error": "'configs' not formatted correctly"}), 400
    model = configs.get('model')
    data = pd.DataFrame(request_data['data'])
    try:
        if model.lower() == 'health':
            from models.healthcare_models import generate_health_data
            synthesized_data = generate_health_data(data, configs.get('num_rows', 50), configs.get('categorical_cols', None), configs.get('epochs', 10))
        elif model.lower() == 'finance':
            from models.finance_models import generate_finance_data
            synthesized_data = generate_finance_data(data, configs.get('num_rows', 50), configs.get('categorical_cols', None), configs.get('epochs', 10))
        elif model.lower() == 'education':
            from models.education_models import generate_education_data
            synthesized_data = generate_education_data(data, configs.get('num_rows', 50), configs.get('categorical_cols', None), configs.get('epochs', 10))
        else:
            return jsonify({"error": "Unsupported model for synthesis"}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred during synthesis: {str(e)}"}), 500
    return jsonify({"synthesized_data": synthesized_data.to_json(orient='records')}), 200

@app.route('/balance', methods=['POST'])
@jwt_required()
def balance_data():
    request_data = request.get_json()
    if 'data' not in request_data or 'configs' not in request_data:
        return jsonify({"error": "Missing 'data' or 'configs'"}), 400
    configs = request_data['configs']
    if not isinstance(configs, dict) or 'model' not in configs:
        return jsonify({"error": "'configs' not formatted correctly"}), 400
    model = configs.get('model')
    data = pd.DataFrame(request_data['data'])
    try:
        if model.lower() == 'health':
            from models.healthcare_models import balance_health_data
            synthesized_data = balance_health_data(data)
        elif model.lower() == 'finance':
            if 'target_column' not in configs:
                return jsonify({"error": "Missing 'target_column' in configs"}), 400
            from models.finance_models import balance_finance_data
            synthesized_data = balance_finance_data(data, configs['target_column'])
        elif model.lower() == 'education':
            if 'target_column' not in configs:
                return jsonify({"error": "Missing 'target_column' in configs"}), 400
            from models.education_models import balance_education_data
            synthesized_data = balance_education_data(data, configs['target_column'])
        else:
            return jsonify({"error": "Unsupported model for balancing"}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred during balancing: {str(e)}"}), 500
    return jsonify({"synthesized_data": synthesized_data.to_json(orient='records')}), 200
@app.route('/')
def home():
    return jsonify({"message": "Synthetic Studio Backend is running"}), 200
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

