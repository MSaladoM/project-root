import unittest
import yaml
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.data_loader import load_and_preprocess_data
from src.model_trainer import train_and_save_model

class TestMLOpsPipeline(unittest.TestCase):
    
    def setUp(self):
        # Esta función se ejecuta antes de cada prueba para cargar la configuración
        with open("config/params.yaml", "r") as file:
            self.config = yaml.safe_load(file)
            
    def test_load_and_preprocess_data_not_empty(self):
        """1. Verificar que load_and_preprocess_data no devuelve datos vacíos."""
        X_train, X_test, y_train, y_test = load_and_preprocess_data(self.config)
        
        # Validar que los tamaños sean mayores a 0
        self.assertGreater(len(X_train), 0, "Error: X_train está vacío")
        self.assertGreater(len(X_test), 0, "Error: X_test está vacío")
        
        # Validar que X e y tengan la misma cantidad de filas
        self.assertEqual(len(X_train), len(y_train), "Error: Desajuste de filas en Train")
        self.assertEqual(len(X_test), len(y_test), "Error: Desajuste de filas en Test")

    def test_train_and_save_model_metrics(self):
        """2. Verificar que train_and_save_model devuelve un diccionario con las 3 métricas."""
        # Creamos datos falsos genéricos para aislar la prueba del modelo
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Ejecutamos la función del Rol 2
        metrics = train_and_save_model(X_train, y_train, X_test, y_test, self.config)
        
        # Validar que devuelve un diccionario
        self.assertIsInstance(metrics, dict, "Error: La función no devolvió un diccionario")
        
        # Validar que las 3 llaves exactas existen
        self.assertIn("accuracy", metrics, "Error: Falta la métrica accuracy")
        self.assertIn("recall", metrics, "Error: Falta la métrica recall")
        self.assertIn("f1_score", metrics, "Error: Falta la métrica f1_score")

if __name__ == "__main__":
    unittest.main()