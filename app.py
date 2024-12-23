from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Définir un modèle de données pour la requête
class Prediction(BaseModel):
    feature1: float
    feature2: float
    feature3: float

# Initialisation de l'application FastAPI
app = FastAPI()

# Charger le modèle de prédiction (assurez-vous que le fichier 'model.pk' est accessible)
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Définir l'endpoint pour la prédiction
@app.post("/predict")
def predict(request: Prediction):
    # Préparer les données d'entrée pour le modèle
    features = np.array([[request.feature1, request.feature2, request.feature3]])

    # Faire la prédiction
    prediction = model.predict(features)
    prediction = prediction[0]

    # Retourner la prédiction sous forme de dictionnaire
    return {"prediction": prediction}
