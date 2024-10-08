import numpy as np

class SAW:
    def __init__(self, decision_matrix, weights):
        self.decision_matrix = np.array(decision_matrix)
        self.weights = np.array(weights)

    def normalize_matrix(self):
        normalized_matrix = np.zeros_like(self.decision_matrix, dtype=float)
        for i in range(self.decision_matrix.shape[1]):
            if self.weights[i] >= 0:  # Criterio de beneficio
                normalized_matrix[:, i] = self.decision_matrix[:, i] / np.max(self.decision_matrix[:, i])
                #print(f"Normalizando criterio de beneficio (Criterio {i + 1}): {normalized_matrix[:, i]}")  # Mostrar la normalización
            else:  # Criterio de costo
                normalized_matrix[:, i] = np.min(self.decision_matrix[:, i]) / self.decision_matrix[:, i]
                #print(f"Normalizando criterio de costo (Criterio {i + 1}): {normalized_matrix[:, i]}")  # Mostrar la normalización
        return normalized_matrix
    
    def calculate_scores(self):
        normalized_matrix = self.normalize_matrix()
        #print(f"Matriz normalizada:\n{normalized_matrix}")  # Mostrar la matriz normalizada
        
        weighted_matrix = normalized_matrix * np.abs(self.weights)  # Multiplicación de la matriz normalizada por los pesos
        #print(f"Matriz ponderada (normalizada * pesos):\n{weighted_matrix}")  # Mostrar la matriz ponderada
        
        scores = np.sum(weighted_matrix, axis=1)  # Sumar los valores ponderados para cada alternativa
        #print(f"Scores (suma de la matriz ponderada): {scores}")  # Mostrar los scores finales
        return scores
    
    @staticmethod   
    def from_parameters(decision_matrix, weights):
        saw_instance = SAW(np.array(decision_matrix), np.array(weights))
        return saw_instance.calculate_scores()
    
    @staticmethod
    def from_file(file_content):
        # Leer el contenido del archivo y convertirlo en una matriz y un arreglo de pesos
        lines = [list(map(float, line.strip().split(','))) for line in file_content]
        decision_matrix = lines[:-1]  # Matriz de decisiones
        weights = lines[-1]  # La última fila contiene los pesos
        return SAW(decision_matrix, weights)   