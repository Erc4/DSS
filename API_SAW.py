from flask import Flask, request, jsonify
from saw import SAW
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


if __name__ == '__main__':
    app.run(debug=True)   