import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.spatial.distance import cdist
import pickle
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

class IncrementalKernelClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, bandwidth=1.0, kernel='gaussian', cache_predictions=True):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.cache_predictions = cache_predictions

        # Dane treningowe
        self.X_train = None
        self.y_train = None
        self.classes_ = None

        # Cache dla przyspieszenia
        self.prediction_cache = {}  # {(x1, x2): [prob_class0, prob_class1]}
        self.grid_cache = {}  # Cache dla regularnej siatki punktów

        # Statystyki klasy (dla szybszego dostępu)
        self.class_points = defaultdict(list)  # {class_label: [points]}
        self.class_counts = defaultdict(int)

    def gaussian_kernel(self, distances):
        """Jądro Gaussa"""
        return np.exp(-0.5 * (distances / self.bandwidth) ** 2)

    def fit(self, X, y):
        """Pierwotne trenowanie modelu"""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.classes_ = np.unique(y)

        # Buduj cache punktów dla każdej klasy
        self._rebuild_class_cache()

        # Opcjonalnie: wstępnie oblicz predykcje dla siatki punktów
        if self.cache_predictions:
            self._precompute_grid_predictions()

        return self

    def _rebuild_class_cache(self):
        """Odbudowuje cache punktów dla każdej klasy"""
        self.class_points.clear()
        self.class_counts.clear()

        for i, class_label in enumerate(self.y_train):
            if class_label not in self.class_points:
                self.class_points[class_label] = []
            self.class_points[class_label].append(self.X_train[i])
            self.class_counts[class_label] += 1

        # Konwertuj listy na numpy arrays
        for class_label in self.class_points:
            self.class_points[class_label] = np.array(self.class_points[class_label])

    def _precompute_grid_predictions(self, grid_resolution=50):
        """Wstępnie oblicza predykcje dla regularnej siatki punktów"""
        if self.X_train is None or len(self.X_train) == 0:
            return

        # Utwórz siatkę punktów w przestrzeni danych
        x1_min, x1_max = self.X_train[:, 0].min() - 1, self.X_train[:, 0].max() + 1
        x2_min, x2_max = self.X_train[:, 1].min() - 1, self.X_train[:, 1].max() + 1

        x1_grid = np.linspace(x1_min, x1_max, grid_resolution)
        x2_grid = np.linspace(x2_min, x2_max, grid_resolution)

        for x1 in x1_grid:
            for x2 in x2_grid:
                point = (round(x1, 2), round(x2, 2))
                self.grid_cache[point] = self._compute_density_at_point([x1, x2])

    def _compute_density_at_point(self, x):
        """Oblicza gęstość prawdopodobieństwa w punkcie x"""
        class_densities = []

        for class_label in self.classes_:
            if class_label not in self.class_points or len(self.class_points[class_label]) == 0:
                density = 0
            else:
                # Oblicz odległości do wszystkich punktów klasy
                distances = cdist([x], self.class_points[class_label], metric='euclidean')[0]

                # Zastosuj jądro Gaussa
                kernel_values = self.gaussian_kernel(distances)

                # Gęstość jako średnia z wartości jądra
                density = np.mean(kernel_values)

            class_densities.append(density)

        # Normalizuj prawdopodobieństwa
        total_density = sum(class_densities)
        if total_density > 0:
            return [d / total_density for d in class_densities]
        else:
            return [1 / len(self.classes_)] * len(self.classes_)

    def add_training_point(self, x_new, y_new, update_radius=None):
        """
        Dodaje nowy punkt treningowy i aktualizuje tylko lokalne predykcje

        Args:
            x_new: nowy punkt (wiek, BMI)
            y_new: etykieta klasy (0 lub 1)
            update_radius: promień aktualizacji (domyślnie 3 * bandwidth)
        """
        if update_radius is None:
            update_radius = 3 * self.bandwidth

        x_new = np.array(x_new)

        # Dodaj nowy punkt do danych treningowych
        if self.X_train is None:
            self.X_train = np.array([x_new])
            self.y_train = np.array([y_new])
            self.classes_ = np.unique([y_new])
        else:
            self.X_train = np.vstack([self.X_train, x_new])
            self.y_train = np.append(self.y_train, y_new)
            self.classes_ = np.unique(self.y_train)

        # Aktualizuj cache punktów klasy - POPRAWKA TUTAJ
        if y_new not in self.class_points:
            # Jeśli to pierwsza klasa tego typu, stwórz nowy array
            self.class_points[y_new] = np.array([x_new])
        else:
            # Jeśli klasa już istnieje, dodaj punkt do istniejącego array
            self.class_points[y_new] = np.vstack([self.class_points[y_new], x_new])

        self.class_counts[y_new] += 1

        # Aktualizuj cache predykcji w obszarze wpływu nowego punktu
        self._update_local_predictions(x_new, update_radius)

        print(f"Dodano punkt: {x_new}, klasa: {y_new}")
        print(f"Zaktualizowano predykcje w promieniu {update_radius} od nowego punktu")

    def _update_local_predictions(self, x_new, update_radius):
        """Aktualizuje predykcje w lokalnym obszarze wokół nowego punktu"""
        updated_count = 0

        # Aktualizuj cache grid
        points_to_remove = []
        for point, _ in self.grid_cache.items():
            point_coords = np.array(point)
            distance = np.linalg.norm(point_coords - x_new)

            if distance <= update_radius:
                # Przelicz predykcję dla tego punktu
                self.grid_cache[point] = self._compute_density_at_point(point_coords)
                updated_count += 1

        # Aktualizuj cache predykcji
        points_to_remove = []
        for cached_point, _ in self.prediction_cache.items():
            point_coords = np.array(cached_point)
            distance = np.linalg.norm(point_coords - x_new)

            if distance <= update_radius:
                points_to_remove.append(cached_point)

        # Usuń nieaktualne cache (będą przeliczone przy następnym zapytaniu)
        for point in points_to_remove:
            del self.prediction_cache[point]
            updated_count += 1

        print(f"Zaktualizowano {updated_count} punktów w cache")

    def predict_proba(self, X):
        """Predykcja prawdopodobieństwa z wykorzystaniem cache"""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        probas = []
        for x in X:
            # Sprawdź cache
            cache_key = (round(x[0], 2), round(x[1], 2))

            if cache_key in self.prediction_cache:
                probas.append(self.prediction_cache[cache_key])
            elif cache_key in self.grid_cache:
                probas.append(self.grid_cache[cache_key])
            else:
                # Oblicz nową predykcję
                proba = self._compute_density_at_point(x)

                # Zapisz w cache
                if self.cache_predictions:
                    self.prediction_cache[cache_key] = proba

                probas.append(proba)

        return np.array(probas)

    def predict(self, X):
        """Predykcja klasy"""
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

    def get_influence_area(self, x_point, radius_multiplier=3):
        """
        Zwraca punkty treningowe w obszarze wpływu danego punktu
        """
        influence_radius = radius_multiplier * self.bandwidth
        distances = cdist([x_point], self.X_train, metric='euclidean')[0]
        influenced_indices = np.where(distances <= influence_radius)[0]

        return {
            'points': self.X_train[influenced_indices],
            'labels': self.y_train[influenced_indices],
            'distances': distances[influenced_indices],
            'count': len(influenced_indices)
        }

    def save_model(self, filepath):
        """Zapisz model do pliku"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filepath):
        """Wczytaj model z pliku"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def get_model_stats(self):
        """Zwraca statystyki modelu"""
        return {
            'training_points': len(self.X_train) if self.X_train is not None else 0,
            'classes': list(self.classes_) if self.classes_ is not None else [],
            'class_distribution': dict(self.class_counts),
            'cache_size': len(self.prediction_cache),
            'grid_cache_size': len(self.grid_cache),
            'bandwidth': self.bandwidth
        }

    def evaluate(self, X_test, y_test):
        """
        Ocena modelu na zbiorze testowym z obliczeniem metryk
        """
        try:
            y_pred = self.predict(X_test)
            y_proba = self.predict_proba(X_test)[:, 1]

            # Podstawowe metryki
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            try:
                roc_auc = roc_auc_score(y_test, y_proba)
            except ValueError:
                roc_auc = 0.5  # Domyślna wartość gdy tylko jedna klasa

            conf_matrix = confusion_matrix(y_test, y_pred)

            # Raport klasyfikacji
            class_report = classification_report(
                y_test, y_pred,
                target_names=['Brak cukrzycy', 'Cukrzyca'],
                zero_division=0
            )

            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'confusion_matrix': conf_matrix,
                'classification_report': class_report,
                'y_test': y_test,
                'y_proba': y_proba
            }
        except Exception as e:
            print(f"Błąd podczas ewaluacji: {e}")
            return None


# Przykład użycia
if __name__ == "__main__":
    # Przykładowe dane
    X_initial = np.array([[25, 22], [45, 28], [35, 25], [50, 30]])
    y_initial = np.array([0, 1, 0, 1])

    # Inicjalne trenowanie
    classifier = IncrementalKernelClassifier(bandwidth=5.0)
    classifier.fit(X_initial, y_initial)

    print("=== Model po początkowym trenowaniu ===")
    print(classifier.get_model_stats())

    # Testowa predykcja
    test_point = [40, 26]
    prob_before = classifier.predict_proba([test_point])[0]
    print(f"\nPredykcja dla {test_point} PRZED dodaniem nowego punktu:")
    print(f"P(klasa=0): {prob_before[0]:.3f}, P(klasa=1): {prob_before[1]:.3f}")

    # Dodanie nowego punktu treningowego
    print(f"\n=== Dodawanie nowego punktu [42, 27] z klasą 1 ===")
    classifier.add_training_point([42, 27], 1)

    # Predykcja po dodaniu punktu
    prob_after = classifier.predict_proba([test_point])[0]
    print(f"\nPredykcja dla {test_point} PO dodaniu nowego punktu:")
    print(f"P(klasa=0): {prob_after[0]:.3f}, P(klasa=1): {prob_after[1]:.3f}")

    print(f"\nZmiana prawdopodobieństwa:")
    print(f"Δ P(klasa=0): {prob_after[0] - prob_before[0]:+.3f}")
    print(f"Δ P(klasa=1): {prob_after[1] - prob_before[1]:+.3f}")

    # Sprawdź obszar wpływu
    influence = classifier.get_influence_area([42, 27])
    print(f"\nObszar wpływu nowego punktu [42, 27]:")
    print(f"Liczba wpływających punktów: {influence['count']}")
    print(f"Punkty w obszarze wpływu: {influence['points']}")

    print("\n=== Statystyki modelu po aktualizacji ===")
    print(classifier.get_model_stats())