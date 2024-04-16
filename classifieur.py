import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Charger le dataset
digits = load_digits()
X, y = digits.data, digits.target

# Préparation des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation des données
def normalize_data(X):
    """
    Normalise les données pour avoir une moyenne de 0 et un écart-type de 1.
    Évite la division par zéro en remplaçant l'écart-type de zéro par un (pour éviter la division par zéro).

    Paramètres :
    - X : les données à normaliser.

    Retourne :
    - Les données normalisées.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # Éviter la division par zéro
    std_replaced = np.where(std == 0, 1, std)
    X_normalized = (X - mean) / std_replaced
    return X_normalized


X_train_scaled = normalize_data(X_train)
X_test_scaled = normalize_data(X_test)
#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)

# Ajout du biais
X_train_scaled = np.hstack([np.ones((X_train_scaled.shape[0], 1)), X_train_scaled])
X_test_scaled = np.hstack([np.ones((X_test_scaled.shape[0], 1)), X_test_scaled])

# Définition des fonctions nécessaires
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    epsilon = 1e-5
    cost = -(1/m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
    return cost

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    
    for i in range(iterations):
        gradient = X.T.dot(sigmoid(X.dot(theta)) - y) / m
        theta -= alpha * gradient
        cost_history[i] = compute_cost(X, y, theta)
    
    return theta, cost_history

# Entraînement du modèle et évaluation
num_classes = len(np.unique(y_train))
theta_initial = np.zeros((X_train_scaled.shape[1], num_classes))
iterations = 1000
alpha = 0.01

for i in range(num_classes):
    y_train_multiclass = (y_train == i).astype(int)
    theta_optimal, cost_history = gradient_descent(X_train_scaled, y_train_multiclass, theta_initial[:,i], alpha, iterations)
    theta_initial[:,i] = theta_optimal

# Prédiction sur l'ensemble de test
def predict_classes(X, all_theta):
    preds = sigmoid(X.dot(all_theta))
    return np.argmax(preds, axis=1)

y_pred = predict_classes(X_test_scaled, theta_initial)
accuracy = np.mean(y_pred == y_test)
print(f'Accuracy: {accuracy:.4f}')

# Sélection de quelques échantillons de l'ensemble de test
sample_indices = [0, 1, 2, 3, 4]  # Choisir des indices d'échantillons
sample_images = X_test_scaled[sample_indices]
sample_labels = y_test[sample_indices]

# Prédiction des classes pour les échantillons choisis
sample_predictions = predict_classes(sample_images, theta_initial)

# Affichage des vraies étiquettes et des prédictions
print("Vraies étiquettes : ", sample_labels)    
print("Prédictions : ", sample_predictions)

# Afficher les images des échantillons avec matplotlib (si souhaité)
import matplotlib.pyplot as plt 

fig, axes = plt.subplots(1, len(sample_indices), figsize=(10, 3))
for i, ax in enumerate(axes):
    ax.imshow(sample_images[i][1:].reshape(8, 8), cmap='gray')  # Le [1:] est pour enlever la colonne de biais ajoutée
    ax.set_title(f"Vrai: {sample_labels[i]}\nPrédit: {sample_predictions[i]}")
    ax.axis('off')
plt.show()


### 2 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Création du modèle de régression logistique
log_reg = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs', multi_class='multinomial')

# Entraînement du modèle
log_reg.fit(X_train_scaled, y_train)

# Prédiction sur l'ensemble de test
y_pred_sklearn = log_reg.predict(X_test_scaled)

# Calcul de la précision
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
print(f'Précision avec scikit-learn: {accuracy_sklearn:.4f}')

# Prédiction des classes pour les mêmes échantillons choisis avec scikit-learn
# Assurez-vous d'utiliser les données non modifiées, sans colonne de biais ajoutée manuellement
sample_images_sklearn = X_test_scaled[sample_indices, :]  # Utiliser X_test_scaled directement

# Prédiction avec scikit-learn
sample_predictions_sklearn = log_reg.predict(sample_images_sklearn)

# Affichage des vraies étiquettes et des prédictions de scikit-learn
print("Vraies étiquettes : ", sample_labels)
print("Prédictions avec scikit-learn: ", sample_predictions_sklearn)

# Afficher les images des échantillons avec les prédictions de scikit-learn
fig, axes = plt.subplots(1, len(sample_indices), figsize=(10, 3))
for i, ax in enumerate(axes):
    ax.imshow(X_test[sample_indices][i].reshape(8, 8), cmap='gray')  # Utiliser X_test directement pour afficher les images
    ax.set_title(f"Vrai: {sample_labels[i]}\nPrédit: {sample_predictions_sklearn[i]}")
    ax.axis('off')
plt.show()
