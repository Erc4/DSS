from flask import Flask, request, jsonify
from SAW import SAW
import numpy as np

app = Flask(__name__)

@app.route('/saw/normalize', methods=['POST'])
def normalize():
    data = request.json
    decision_matrix = data['decision_matrix']
    weights = data['weights']
    
    saw_instance = SAW(np.array(decision_matrix), np.array(weights))
    normalized_matrix = saw_instance.normalize_matrix()
    rounded_matrix = np.round(normalized_matrix, 2).tolist()
    scores = saw_instance.calculate_scores()

    return jsonify({"normalized_matrix": rounded_matrix, "scores": scores.tolist()})

@app.route('/saw/file', methods=['POST'])
def solve_with_file():
    file = request.files['parametros.txt']
    file_content = file.read().decode('utf-8').splitlines()
    saw_instance = SAW.from_file(file_content) 
    scores = saw_instance.calculate_scores()
    normalized_matrix = saw_instance.normalize_matrix() # Leer el archivo, decodificar y dividir en líneas
    #scores = SAW.from_file(file_content)  # Llamar al método sobrecargado
    
    # Convertir los scores a una lista
    #scores_list = scores.tolist() if isinstance(scores, np.ndarray) else scores

    return jsonify({
                    "scores": scores.tolist(),
                    "normalized matrix": normalized_matrix.tolist()})

if __name__ == '__main__':
    app.run(debug=True)   