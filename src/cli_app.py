# cli_app.py
import argparse
import numpy as np
import os
import pickle
from database import DiabetesDatabase
from kernel_classifier import IncrementalKernelClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from datetime import datetime
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


class IncrementalDiabetesCLI:
    def __init__(self, model_path="models/diabetes_model.pkl", db_path="data/diabetes.db"):
        self.db = DiabetesDatabase(db_path)
        self.classifier = None
        self.model_path = model_path
        self.feature_names = None
        self.normalization_params = {}

        # Utwórz katalog dla modeli jeśli nie istnieje
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        # Spróbuj wczytać istniejący model
        self._load_or_create_model()

    def _load_or_create_model(self):
        """Wczytuje istniejący model lub przygotowuje do utworzenia nowego"""
        if os.path.exists(self.model_path):
            try:
                model_data = self._load_model_data()
                self.classifier = model_data['classifier']
                self.normalization_params = model_data.get('normalization_params', {})
                self.feature_names = model_data.get('feature_names', ['age', 'bmi'])

                print(f"Wczytano istniejący model z {self.model_path}")
                print(f"Cechy modelu: {self.feature_names}")
                stats = self.classifier.get_model_stats()
                print(f"Model zawiera {stats['training_points']} punktów treningowych")
            except Exception as e:
                print(f"Błąd wczytywania modelu: {e}")
                self.classifier = None
        else:
            print("Nie znaleziono istniejącego modelu - zostanie utworzony nowy")

    def initial_train(self, bandwidth=2.0, features=['age', 'bmi']):
        """
        Początkowe trenowanie modelu na danych z bazy

        Args:
            bandwidth: parametr jądra Gaussa
            features: lista cech do użycia w modelu
        """
        print("Początkowe trenowanie modelu...")

        # Sprawdź dostępność danych
        try:
            df = self.db.get_training_data(features=features)
        except Exception as e:
            print(f"Błąd pobierania danych: {e}")
            return False

        if len(df) < 5:
            print("Zbyt mało danych do trenowania (minimum 5 rekordów)")
            print("Spróbuj załadować dane używając: python data_loader.py --load-csv your_data.csv")
            return False

        # Przygotuj dane
        X = df[features].values
        y = df['outcome'].values

        print(f"Trenowanie na {len(X)} próbkach")
        print(f"Cechy: {features}")
        print(f"Rozkład klas: {np.bincount(y)}")

        # Oblicz parametry normalizacji
        self.feature_names = features
        self.normalization_params = {}

        for i, feature in enumerate(features):
            self.normalization_params[feature] = {
                'mean': X[:, i].mean(),
                'std': X[:, i].std()
            }

        # Normalizuj dane
        X_normalized = self._normalize_data(X)

        # Stwórz i wytrenuj klasyfikator
        self.classifier = IncrementalKernelClassifier(
            bandwidth=bandwidth,
            cache_predictions=True
        )
        self.classifier.fit(X_normalized, y)

        # Zapisz model
        self._save_model()

        stats = self.classifier.get_model_stats()
        print(f"✅ Model wytrenowany na {stats['training_points']} próbkach")
        print(f"Rozkład klas: {stats['class_distribution']}")
        return True

    def _normalize_data(self, X):
        """Normalizuje dane wejściowe używając zapisanych parametrów"""
        if not self.normalization_params:
            raise ValueError("Brak parametrów normalizacji - model nie został wytrenowany")

        X_norm = X.copy().astype(float)
        for i, feature in enumerate(self.feature_names):
            if feature in self.normalization_params:
                mean = self.normalization_params[feature]['mean']
                std = self.normalization_params[feature]['std']
                X_norm[:, i] = (X[:, i] - mean) / std
            else:
                print(f"Ostrzeżenie: brak parametrów normalizacji dla cechy '{feature}'")

        return X_norm

    def _denormalize_data(self, X_norm):
        """Odwraca normalizację danych"""
        X = X_norm.copy()
        for i, feature in enumerate(self.feature_names):
            if feature in self.normalization_params:
                mean = self.normalization_params[feature]['mean']
                std = self.normalization_params[feature]['std']
                X[:, i] = X_norm[:, i] * std + mean
        return X

    def add_and_update(self, data_dict, save_to_db=True):
        """
        Dodaje nowy punkt i aktualizuje model inkrementalnie

        Args:
            data_dict: słownik z danymi pacjenta (musi zawierać wszystkie cechy modelu + outcome)
            save_to_db: czy zapisać do bazy danych
        """
        if self.classifier is None:
            print("Model nie został jeszcze wytrenowany! Użyj --initial-train")
            return

        # Sprawdź czy mamy wszystkie wymagane cechy
        missing_features = [f for f in self.feature_names if f not in data_dict]
        if missing_features:
            print(f"Brakujące cechy: {missing_features}")
            return

        if 'outcome' not in data_dict:
            print("Brakuje etykiety 'outcome' (0 lub 1)")
            return

        # Zapisz w bazie danych
        if save_to_db:
            try:
                self.db.add_record(**data_dict)
            except Exception as e:
                print(f"Błąd zapisu do bazy: {e}")

        # Przygotuj punkt do normalizacji
        new_point = np.array([[data_dict[f] for f in self.feature_names]])
        new_point_norm = self._normalize_data(new_point)[0]
        outcome = data_dict['outcome']

        print(f"\n=== Dodawanie nowego punktu ===")
        feature_str = ", ".join([f"{f}: {data_dict[f]}" for f in self.feature_names])
        print(f"{feature_str}, Outcome: {outcome}")

        # Sprawdź predykcję PRZED dodaniem
        prob_before = self.classifier.predict_proba([new_point_norm])[0]
        print(f"\nPredykcja PRZED dodaniem:")
        print(f"P(brak cukrzycy): {prob_before[0]:.3f}")
        print(f"P(cukrzyca): {prob_before[1]:.3f}")

        # Dodaj punkt do modelu
        self.classifier.add_training_point(new_point_norm, outcome)

        # Sprawdź predykcję PO dodaniu
        prob_after = self.classifier.predict_proba([new_point_norm])[0]
        print(f"\nPredykcja PO dodaniu:")
        print(f"P(brak cukrzycy): {prob_after[0]:.3f}")
        print(f"P(cukrzyca): {prob_after[1]:.3f}")

        # Sprawdź obszar wpływu
        influence = self.classifier.get_influence_area(new_point_norm)
        print(f"\nObszar wpływu: {influence['count']} punktów")

        # Zapisz zaktualizowany model
        self._save_model()
        print("✅ Model zaktualizowany i zapisany!")

    def predict_diabetes(self, data_dict, show_influence=False):
        """
        Przewiduje ryzyko cukrzycy

        Args:
            data_dict: słownik z cechami pacjenta
            show_influence: czy pokazać punkty wpływające na predykcję
        """
        if self.classifier is None:
            print("Model nie został wytrenowany!")
            return

        # Sprawdź czy mamy wszystkie wymagane cechy
        missing_features = [f for f in self.feature_names if f not in data_dict]
        if missing_features:
            print(f"Brakujące cechy: {missing_features}")
            return

        # Przygotuj dane
        input_point = np.array([[data_dict[f] for f in self.feature_names]])
        input_point_norm = self._normalize_data(input_point)[0]

        # Predykcja
        prediction = self.classifier.predict([input_point_norm])[0]
        probabilities = self.classifier.predict_proba([input_point_norm])[0]

        print(f"\n=== Predykcja ===")
        feature_str = ", ".join([f"{f}: {data_dict[f]}" for f in self.feature_names])
        print(f"Dane: {feature_str}")
        print(f"Ryzyko cukrzycy: {'WYSOKIE' if prediction == 1 else 'NISKIE'}")
        print(f"P(brak cukrzycy): {probabilities[0]:.3f}")
        print(f"P(cukrzyca): {probabilities[1]:.3f}")

        if show_influence:
            influence = self.classifier.get_influence_area(input_point_norm)
            print(f"\nPunkty wpływające na predykcję:")
            print(f"Liczba punktów: {influence['count']}")

            if influence['count'] > 0:
                # Denormalizuj punkty do wyświetlenia
                points_denorm = self._denormalize_data(influence['points'])
                print("Najbliższe punkty treningowe:")
                for i, (point, label, dist) in enumerate(
                        zip(points_denorm[:5], influence['labels'][:5], influence['distances'][:5])):
                    feature_vals = ", ".join([f"{self.feature_names[j]}: {point[j]:.1f}"
                                              for j in range(len(self.feature_names))])
                    print(f"  {i + 1}. {feature_vals}, Klasa: {label}, Odległość: {dist:.3f}")

    def visualize_model(self, save_plot=False, features_to_plot=None):
        """
        Wizualizuje model i dane treningowe (tylko dla modeli 2D)

        Args:
            save_plot: czy zapisać wykres
            features_to_plot: które dwie cechy wyświetlić [feature1, feature2]
        """
        if self.classifier is None:
            print("Model nie został wytrenowany!")
            return

        if len(self.feature_names) < 2:
            print("Wizualizacja wymaga przynajmniej 2 cech")
            return

        # Wybierz cechy do wizualizacji
        if features_to_plot is None:
            features_to_plot = self.feature_names[:2]
        elif len(features_to_plot) != 2:
            print("Do wizualizacji potrzeba dokładnie 2 cech")
            return

        feature_indices = [self.feature_names.index(f) for f in features_to_plot
                           if f in self.feature_names]

        if len(feature_indices) != 2:
            print(f"Nie można znaleźć cech {features_to_plot} w modelu")
            return

        print(f"Wizualizacja dla cech: {features_to_plot}")

        # Pobierz zakresy danych treningowych (zdenormalizowanych)
        if hasattr(self.classifier, 'X_train') and self.classifier.X_train is not None:
            train_points_denorm = self._denormalize_data(self.classifier.X_train)

            # Wybierz odpowiednie kolumny
            x_data = train_points_denorm[:, feature_indices[0]]
            y_data = train_points_denorm[:, feature_indices[1]]

            x_min, x_max = x_data.min() - 5, x_data.max() + 5
            y_min, y_max = y_data.min() - 5, y_data.max() + 5
        else:
            # Domyślne zakresy jeśli brak danych treningowych
            x_min, x_max = 20, 80
            y_min, y_max = 15, 45

        # Utwórz siatkę punktów
        x_range = np.linspace(x_min, x_max, 50)
        y_range = np.linspace(y_min, y_max, 50)
        x_grid, y_grid = np.meshgrid(x_range, y_range)

        # Przygotuj punkty siatki dla predykcji
        grid_points = np.zeros((len(x_range) * len(y_range), len(self.feature_names)))
        grid_points[:, feature_indices[0]] = x_grid.ravel()
        grid_points[:, feature_indices[1]] = y_grid.ravel()

        # Wypełnij pozostałe cechy średnimi wartościami
        for i, feature in enumerate(self.feature_names):
            if i not in feature_indices:
                grid_points[:, i] = self.normalization_params[feature]['mean']

        # Normalizuj i przewiduj
        grid_points_norm = self._normalize_data(grid_points)
        predictions = self.classifier.predict_proba(grid_points_norm)
        prob_diabetes = predictions[:, 1].reshape(x_grid.shape)

        # Rysuj
        plt.figure(figsize=(12, 8))

        # Mapa kolorów dla prawdopodobieństwa
        contour = plt.contourf(x_grid, y_grid, prob_diabetes,
                               levels=20, cmap='RdYlBu_r', alpha=0.7)
        plt.colorbar(contour, label='Prawdopodobieństwo cukrzycy')

        # Dodaj punkty treningowe
        if hasattr(self.classifier, 'X_train') and self.classifier.X_train is not None:
            train_points_denorm = self._denormalize_data(self.classifier.X_train)

            x_train = train_points_denorm[:, feature_indices[0]]
            y_train = train_points_denorm[:, feature_indices[1]]

            # Punkty bez cukrzycy (klasa 0)
            class_0_mask = self.classifier.y_train == 0
            plt.scatter(x_train[class_0_mask], y_train[class_0_mask],
                        c='blue', marker='o', s=50, label='Brak cukrzycy',
                        edgecolors='black', alpha=0.8)

            # Punkty z cukrzycą (klasa 1)
            class_1_mask = self.classifier.y_train == 1
            plt.scatter(x_train[class_1_mask], y_train[class_1_mask],
                        c='red', marker='s', s=50, label='Cukrzyca',
                        edgecolors='black', alpha=0.8)

        plt.xlabel(features_to_plot[0].replace('_', ' ').title())
        plt.ylabel(features_to_plot[1].replace('_', ' ').title())
        plt.title('Klasyfikator Ryzyka Cukrzycy - Mapa Prawdopodobieństwa')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_plot:
            filename = f'diabetes_classifier_{features_to_plot[0]}_{features_to_plot[1]}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Wykres zapisany jako '{filename}'")

        plt.show()

    def _save_model(self):
        """Zapisuje model wraz z parametrami normalizacji"""
        if self.classifier is not None:
            model_data = {
                'classifier': self.classifier,
                'normalization_params': self.normalization_params,
                'feature_names': self.feature_names
            }

            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)

    def _load_model_data(self):
        """Wczytuje dane modelu z pliku"""
        with open(self.model_path, 'rb') as f:
            return pickle.load(f)

    def show_stats(self):
        """Pokazuje statystyki modelu i bazy danych"""
        print("\n=== STATYSTYKI MODELU ===")

        if self.classifier is None:
            print("Model nie został wytrenowany!")
        else:
            stats = self.classifier.get_model_stats()
            print(f"Liczba punktów treningowych: {stats['training_points']}")
            print(f"Klasy: {stats['classes']}")
            print(f"Rozkład klas: {stats['class_distribution']}")
            print(f"Bandwidth: {stats['bandwidth']}")
            print(f"Cechy modelu: {self.feature_names}")
            print(f"Rozmiar cache predykcji: {stats['cache_size']}")
            print(f"Rozmiar cache siatki: {stats['grid_cache_size']}")

        print("\n=== STATYSTYKI BAZY DANYCH ===")
        db_stats = self.db.get_statistics()
        if db_stats.get('total_records', 0) == 0:
            print("Baza danych jest pusta")
        else:
            print(f"Całkowita liczba rekordów: {db_stats['total_records']}")
            print(f"Rozkład wyników: {db_stats['outcome_distribution']}")

            if 'age_stats' in db_stats:
                age_stats = db_stats['age_stats']
                print(f"Wiek - średnia: {age_stats['mean']:.1f}, "
                      f"odchylenie: {age_stats['std']:.1f}, "
                      f"zakres: {age_stats['min']:.1f}-{age_stats['max']:.1f}")

            if 'bmi_stats' in db_stats:
                bmi_stats = db_stats['bmi_stats']
                print(f"BMI - średnia: {bmi_stats['mean']:.1f}, "
                      f"odchylenie: {bmi_stats['std']:.1f}, "
                      f"zakres: {bmi_stats['min']:.1f}-{bmi_stats['max']:.1f}")

    def interactive_mode(self):
        """Tryb interaktywny dla łatwego użytkowania"""
        print("\n=== TRYB INTERAKTYWNY ===")
        print("Dostępne komendy:")
        print("1. train - trenuj nowy model")
        print("2. add - dodaj nowy punkt treningowy")
        print("3. predict - przewiduj ryzyko")
        print("4. visualize - pokaż wizualizację")
        print("5. stats - pokaż statystyki")
        print("6. exit - wyjście")

        while True:
            try:
                command = input("\nWybierz komendę (1-6): ").strip()

                if command == '1' or command.lower() == 'train':
                    self._interactive_train()
                elif command == '2' or command.lower() == 'add':
                    self._interactive_add()
                elif command == '3' or command.lower() == 'predict':
                    self._interactive_predict()
                elif command == '4' or command.lower() == 'visualize':
                    self.visualize_model(save_plot=True)
                elif command == '5' or command.lower() == 'stats':
                    self.show_stats()
                elif command == '6' or command.lower() == 'exit':
                    print("Do widzenia!")
                    break
                else:
                    print("Nieprawidłowa komenda. Wybierz 1-6.")

            except KeyboardInterrupt:
                print("\nProgram przerwany przez użytkownika")
                break
            except Exception as e:
                print(f"Błąd: {e}")

    def _interactive_train(self):
        """Interaktywne trenowanie modelu"""
        print("\nTrenowanie nowego modelu...")

        # Dostępne cechy
        available_features = ['pregnancies', 'glucose', 'blood_pressure',
                              'skin_thickness', 'insulin', 'bmi',
                              'diabetes_pedigree_function', 'age']

        print("Dostępne cechy:", available_features)
        features_input = input("Podaj cechy oddzielone przecinkami (lub Enter dla age,bmi): ").strip()

        if features_input:
            features = [f.strip() for f in features_input.split(',')]
            # Sprawdź poprawność cech
            invalid_features = [f for f in features if f not in available_features]
            if invalid_features:
                print(f"Nieprawidłowe cechy: {invalid_features}")
                return
        else:
            features = ['age', 'bmi']

        bandwidth = input("Bandwidth (Enter dla 2.0): ").strip()
        bandwidth = float(bandwidth) if bandwidth else 2.0

        self.initial_train(bandwidth=bandwidth, features=features)

    def _interactive_add(self):
        """Interaktywne dodawanie punktu"""
        if self.classifier is None:
            print("Najpierw wytrenuj model!")
            return

        print(f"\nDodawanie nowego punktu. Wymagane cechy: {self.feature_names}")
        data_dict = {}

        # Pobierz wartości cech
        for feature in self.feature_names:
            while True:
                try:
                    value = float(input(f"{feature}: "))
                    data_dict[feature] = value
                    break
                except ValueError:
                    print("Podaj liczbę!")

        # Pobierz outcome
        while True:
            try:
                outcome = int(input("Outcome (0 - brak cukrzycy, 1 - cukrzyca): "))
                if outcome in [0, 1]:
                    data_dict['outcome'] = outcome
                    break
                else:
                    print("Outcome musi być 0 lub 1!")
            except ValueError:
                print("Podaj 0 lub 1!")

        self.add_and_update(data_dict)

    def _interactive_predict(self):
        """Interaktywna predykcja"""
        if self.classifier is None:
            print("Najpierw wytrenuj model!")
            return

        print(f"\nPredykcja ryzyka. Wymagane cechy: {self.feature_names}")
        data_dict = {}

        for feature in self.feature_names:
            while True:
                try:
                    value = float(input(f"{feature}: "))
                    data_dict[feature] = value
                    break
                except ValueError:
                    print("Podaj liczbę!")

        show_influence = input("Pokazać wpływające punkty? (y/n): ").lower() == 'y'
        self.predict_diabetes(data_dict, show_influence=show_influence)

    def run_evaluation(self, test_size=0.2, random_state=42):
        """Przeprowadza pełną ewaluację modelu z podziałem na zbiór treningowy/testowy"""
        print("\n=== Rozpoczynam ewaluację modelu ===")

        # Sprawdź czy mamy zdefiniowane cechy
        if not self.feature_names:
            print("⚠️ Brak zdefiniowanych cech! Używam domyślnych ['age', 'bmi']")
            self.feature_names = ['age', 'bmi']

        # Pobierz dane z bazy
        try:
            df = self.db.get_training_data(features=self.feature_names)
        except Exception as e:
            print(f"Błąd pobierania danych: {e}")
            return None

        if len(df) < 10:
            print("⚠️ Zbyt mało danych do ewaluacji (minimum 10 rekordów)")
            return None

        # Przygotuj dane
        try:
            X = df[self.feature_names].values
            y = df['outcome'].values

            # Podział na train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

            # Normalizacja
            if not self.normalization_params:
                print("⚠️ Brak parametrów normalizacji - obliczam na podstawie danych")
                self._calculate_normalization_params(X_train)

            X_train_norm = self._normalize_data(X_train)
            X_test_norm = self._normalize_data(X_test)

            # Trenowanie nowej instancji modelu
            eval_classifier = IncrementalKernelClassifier(
                bandwidth=self.classifier.bandwidth if self.classifier else 2.0,
                cache_predictions=True
            )
            eval_classifier.fit(X_train_norm, y_train)

            # Ewaluacja
            results = eval_classifier.evaluate(X_test_norm, y_test)

            # Zapisz wyniki do pliku
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_report_{timestamp}.txt"

            with open(filename, 'w') as f:
                f.write("=== RAPORT EWALUACJI KLASYFIKATORA CUKRZYCY ===\n\n")
                f.write(f"Data wykonania: {datetime.now()}\n")
                f.write(f"Parametry modelu: bandwidth={eval_classifier.bandwidth}\n")
                f.write(f"Użyte cechy: {self.feature_names}\n\n")

                f.write(f"Liczba próbek treningowych: {len(X_train)}\n")
                f.write(f"Liczba próbek testowych: {len(X_test)}\n\n")

                f.write(f"Dokładność (Accuracy): {results['accuracy']:.4f}\n")
                f.write(f"Precyzja (Precision): {results['precision']:.4f}\n")
                f.write(f"Czułość (Recall): {results['recall']:.4f}\n")
                f.write(f"F1-Score: {results['f1']:.4f}\n")
                f.write(f"AUC-ROC: {results['roc_auc']:.4f}\n\n")

                f.write("Macierz pomyłek:\n")
                f.write(np.array2string(results['confusion_matrix'], separator=', '))
                f.write("\n\n")

                f.write("Raport klasyfikacji:\n")
                f.write(results['classification_report'])

            print(f"✅ Wyniki ewaluacji zapisano w pliku: {filename}")

            # Wizualizacja wyników
            self._plot_evaluation_results(results, filename.replace('.txt', '.png'))

            return results
        except KeyError as ke:
            print(f"⛔ Błąd: Brak kolumny {ke} w danych. Sprawdź feature_names.")
            return None

    def _calculate_normalization_params(self, X):
        """Oblicza parametry normalizacji na podstawie danych"""
        self.normalization_params = {}
        for i, feature in enumerate(self.feature_names):
            self.normalization_params[feature] = {
                'mean': X[:, i].mean(),
                'std': X[:, i].std()
            }

    def _plot_evaluation_results(self, results, filename):
        """Generuje profesjonalną wizualizację wyników ewaluacji"""
        if not results:
            return

        # Konfiguracja stylu wykresów
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 12))

        # Kolory dla spójności wizualnej
        colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#F18F01',
            'danger': '#C73E1D',
            'light': '#F5F5F5',
            'text': '#2C3E50'
        }

        # 1. Macierz pomyłek z dodatkowymi informacjami
        plt.subplot(2, 3, 1)
        cm = results['confusion_matrix']

        # Oblicz percentages dla macierzy pomyłek
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        # Tworzenie macierzy z liczbami i procentami
        annotations = []
        for i in range(cm.shape[0]):
            row_annotations = []
            for j in range(cm.shape[1]):
                row_annotations.append(f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)')
            annotations.append(row_annotations)

        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                    cbar_kws={'label': 'Liczba przypadków'},
                    xticklabels=['Brak cukrzycy', 'Cukrzyca'],
                    yticklabels=['Brak cukrzycy', 'Cukrzyca'])
        plt.title('Macierz pomyłek\n(liczba przypadków i procenty)',
                  fontsize=14, fontweight='bold', color=colors['text'])
        plt.xlabel('Predykcja', fontsize=12, fontweight='bold')
        plt.ylabel('Rzeczywistość', fontsize=12, fontweight='bold')

        # 2. Krzywa ROC z dodatkowymi informacjami
        plt.subplot(2, 3, 2)
        try:
            if 'y_test' in results and 'y_proba' in results:
                fpr, tpr, thresholds = roc_curve(results['y_test'], results['y_proba'])

                plt.plot(fpr, tpr, color=colors['primary'], linewidth=3,
                         label=f'Model (AUC = {results["roc_auc"]:.3f})')
                plt.plot([0, 1], [0, 1], color=colors['danger'], linestyle='--',
                         linewidth=2, label='Losowy klasyfikator (AUC = 0.500)')

                # Dodaj punkt optymalny (najbliższy lewemu górnemu rogowi)
                optimal_idx = np.argmax(tpr - fpr)
                optimal_threshold = thresholds[optimal_idx]
                plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8,
                         label=f'Próg optymalny = {optimal_threshold:.3f}')

                plt.fill_between(fpr, tpr, alpha=0.2, color=colors['primary'])

            else:
                plt.text(0.5, 0.5, "Brak danych do wygenerowania krzywej ROC",
                         ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)
        except Exception as e:
            plt.text(0.5, 0.5, f"Błąd generowania krzywej ROC:\n{str(e)}",
                     ha='center', va='center', fontsize=10, transform=plt.gca().transAxes)

        plt.title('Krzywa ROC', fontsize=14, fontweight='bold', color=colors['text'])
        plt.xlabel('Współczynnik fałszywie pozytywnych (FPR)', fontsize=12)
        plt.ylabel('Współczynnik prawdziwie pozytywnych (TPR)', fontsize=12)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)

        # 3. Wykres słupkowy metryk z wartościami
        plt.subplot(2, 3, 3)
        metrics = ['Dokładność', 'Precyzja', 'Czułość', 'F1-Score', 'AUC-ROC']
        metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        values = [results.get(key, 0) for key in metric_keys]

        # Kolory słupków według wartości (gradient od czerwonego do zielonego)
        bar_colors = []
        for val in values:
            if val >= 0.8:
                bar_colors.append(colors['success'])
            elif val >= 0.6:
                bar_colors.append('#F39C12')  # Pomarańczowy
            else:
                bar_colors.append(colors['danger'])

        bars = plt.bar(metrics, values, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1)

        # Dodaj wartości na słupkach
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.ylim(0, 1.1)
        plt.title('Metryki wydajności modelu', fontsize=14, fontweight='bold', color=colors['text'])
        plt.ylabel('Wartość metryki', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')

        # 4. Rozkład prawdopodobieństw predykcji
        plt.subplot(2, 3, 4)
        if 'y_test' in results and 'y_proba' in results:
            y_test = results['y_test']
            y_proba = results['y_proba']

            # Prawdopodobieństwa dla klasy 0 (brak cukrzycy)
            proba_class_0 = y_proba[y_test == 0]
            # Prawdopodobieństwa dla klasy 1 (cukrzyca)
            proba_class_1 = y_proba[y_test == 1]

            plt.hist(proba_class_0, bins=20, alpha=0.7, color=colors['primary'],
                     label=f'Brak cukrzycy (n={len(proba_class_0)})', density=True)
            plt.hist(proba_class_1, bins=20, alpha=0.7, color=colors['secondary'],
                     label=f'Cukrzyca (n={len(proba_class_1)})', density=True)

            # Dodaj linię na 0.5 (próg decyzyjny)
            plt.axvline(x=0.5, color=colors['danger'], linestyle='--', linewidth=2,
                        label='Próg decyzyjny (0.5)')

            plt.xlabel('Prawdopodobieństwo cukrzycy', fontsize=12)
            plt.ylabel('Gęstość', fontsize=12)
            plt.title('Rozkład prawdopodobieństw predykcji', fontsize=14, fontweight='bold', color=colors['text'])
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, "Brak danych prawdopodobieństw",
                     ha='center', va='center', transform=plt.gca().transAxes)

        # 5. Precision-Recall curve
        plt.subplot(2, 3, 5)
        try:
            if 'y_test' in results and 'y_proba' in results:
                from sklearn.metrics import precision_recall_curve, average_precision_score

                precision_vals, recall_vals, pr_thresholds = precision_recall_curve(
                    results['y_test'], results['y_proba'])
                avg_precision = average_precision_score(results['y_test'], results['y_proba'])

                plt.plot(recall_vals, precision_vals, color=colors['secondary'], linewidth=3,
                         label=f'Model (AP = {avg_precision:.3f})')

                # Baseline (proporcja pozytywnych przykładów)
                baseline = np.mean(results['y_test'])
                plt.axhline(y=baseline, color=colors['danger'], linestyle='--', linewidth=2,
                            label=f'Baseline (AP = {baseline:.3f})')

                plt.fill_between(recall_vals, precision_vals, alpha=0.2, color=colors['secondary'])

                plt.xlabel('Czułość (Recall)', fontsize=12)
                plt.ylabel('Precyzja (Precision)', fontsize=12)
                plt.title('Krzywa Precision-Recall', fontsize=14, fontweight='bold', color=colors['text'])
                plt.legend(loc='lower left')
                plt.grid(True, alpha=0.3)
                plt.xlim([0, 1])
                plt.ylim([0, 1])
            else:
                plt.text(0.5, 0.5, "Brak danych do krzywej P-R",
                         ha='center', va='center', transform=plt.gca().transAxes)
        except Exception as e:
            plt.text(0.5, 0.5, f"Błąd P-R curve:\n{str(e)}",
                     ha='center', va='center', fontsize=10, transform=plt.gca().transAxes)

        # 6. Podsumowanie tekstowe
        plt.subplot(2, 3, 6)
        plt.axis('off')

        # Interpretacja wyników
        interpretation_text = f"""
    INTERPRETACJA WYNIKÓW:

    📊 OGÓLNA WYDAJNOŚĆ:
    • Dokładność: {results.get('accuracy', 0):.1%}
    • Model {'DOBRY' if results.get('accuracy', 0) > 0.8 else 'ŚREDNI' if results.get('accuracy', 0) > 0.6 else 'WYMAGA POPRAWY'}

    🎯 PRECYZJA I CZUŁOŚĆ:
    • Precyzja: {results.get('precision', 0):.1%}
    • Czułość: {results.get('recall', 0):.1%}
    • F1-Score: {results.get('f1', 0):.1%}

    📈 ZDOLNOŚĆ ROZRÓŻNIANIA:
    • AUC-ROC: {results.get('roc_auc', 0):.3f}
    • {'Doskonała' if results.get('roc_auc', 0) > 0.9 else 'Dobra' if results.get('roc_auc', 0) > 0.8 else 'Umiarkowana' if results.get('roc_auc', 0) > 0.7 else 'Słaba'} zdolność klasyfikacji

    ⚠️  BŁĘDY:
    • Fałszywie pozytywne: {results['confusion_matrix'][0, 1] if 'confusion_matrix' in results else 'N/A'}
    • Fałszywie negatywne: {results['confusion_matrix'][1, 0] if 'confusion_matrix' in results else 'N/A'}

    💡 REKOMENDACJE:
    """

        # Dodaj rekomendacje na podstawie wyników
        if results.get('precision', 0) < 0.7:
            interpretation_text += "• Zwiększ precyzję - za dużo fałszywych alarmów\n"
        if results.get('recall', 0) < 0.7:
            interpretation_text += "• Popraw czułość - model pomija przypadki cukrzycy\n"
        if results.get('roc_auc', 0) < 0.8:
            interpretation_text += "• Rozważ dostrojenie parametrów modelu\n"
        if results.get('accuracy', 0) > 0.85:
            interpretation_text += "• Model działa bardzo dobrze!\n"

        plt.text(0.05, 0.95, interpretation_text, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor=colors['light'], alpha=0.8))

        # Ogólne ustawienia wykresu
        plt.suptitle(f'Raport ewaluacji klasyfikatora cukrzycy\n'
                     f'Wygenerowano: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                     fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Zapis wykresu
        try:
            plt.savefig(filename, dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"✅ Wizualizacja wyników zapisana jako: {filename}")

            # Opcjonalnie zapisz też w formacie PDF dla lepszej jakości
            pdf_filename = filename.replace('.png', '.pdf')
            plt.savefig(pdf_filename, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"✅ Wersja PDF zapisana jako: {pdf_filename}")

        except Exception as e:
            print(f"⛔ Błąd zapisu wykresu: {e}")

        plt.close()

    def _calculate_additional_metrics(self, results):
        """Oblicza dodatkowe metryki dla lepszej analizy"""
        try:
            cm = results['confusion_matrix']
            tn, fp, fn, tp = cm.ravel()

            # Dodatkowe metryki
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value

            results.update({
                'specificity': specificity,
                'sensitivity': sensitivity,
                'npv': npv,
                'ppv': ppv,
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn,
                'true_positives': tp
            })

            return results
        except Exception as e:
            print(f"Błąd obliczania dodatkowych metryk: {e}")
            return results

    def visualize_3d_density(self, save_plot=False, grid_resolution=30):
        """
        Wizualizuje funkcję gęstości prawdopodobieństwa w 3D (tylko dla modeli 2D)
        zoptymalizowana wersja

        Args:
            save_plot: czy zapisać wykres
            grid_resolution: rozdzielczość siatki (domyślnie 30 dla wydajności)
        """
        if self.classifier is None:
            print("Model nie został wytrenowany!")
            return

        if len(self.feature_names) != 2:
            print("Wizualizacja 3D wymaga dokładnie 2 cech")
            return

        print(f"Wizualizacja 3D dla cech: {self.feature_names} (rozdzielczość: {grid_resolution}x{grid_resolution})")

        # Pobierz zakresy danych treningowych (zdenormalizowanych)
        if hasattr(self.classifier, 'X_train') and self.classifier.X_train is not None:
            train_points_denorm = self._denormalize_data(self.classifier.X_train)
            x_data = train_points_denorm[:, 0]
            y_data = train_points_denorm[:, 1]
            x_min, x_max = x_data.min() - 5, x_data.max() + 5
            y_min, y_max = y_data.min() - 5, y_data.max() + 5
        else:
            # Domyślne zakresy
            x_min, x_max = 20, 80
            y_min, y_max = 15, 45

        # Utwórz siatkę punktów (mniejsza rozdzielczość)
        x_range = np.linspace(x_min, x_max, grid_resolution)
        y_range = np.linspace(y_min, y_max, grid_resolution)
        x_grid, y_grid = np.meshgrid(x_range, y_range)

        # Przygotuj punkty siatki dla predykcji
        grid_points = np.column_stack([
            x_grid.ravel(),
            y_grid.ravel()
        ])

        # Normalizuj i przewiduj (całą siatką naraz)
        grid_points_norm = self._normalize_data(grid_points)
        predictions = self.classifier.predict_proba(grid_points_norm)
        prob_diabetes = predictions[:, 1].reshape(x_grid.shape)

        # Tworzenie wykresu 3D
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Rysuj powierzchnię z większymi krokami (rstride i cstride)
        surf = ax.plot_surface(x_grid, y_grid, prob_diabetes,
                               cmap='viridis',
                               rstride=2, cstride=2,  # Co drugi punkt
                               edgecolor='none',
                               alpha=0.7)

        # Ogranicz liczbę punktów treningowych do wyświetlenia
        max_points_to_show = 100

        if hasattr(self.classifier, 'X_train') and self.classifier.X_train is not None:
            train_points_denorm = self._denormalize_data(self.classifier.X_train)
            x_train = train_points_denorm[:, 0]
            y_train = train_points_denorm[:, 1]

            # Oblicz prawdopodobieństwa tylko dla punktów treningowych
            z_train = self.classifier.predict_proba(
                self._normalize_data(train_points_denorm)
            )[:, 1]

            # Jeśli jest za dużo punktów, wybierz losową próbkę
            if len(x_train) > max_points_to_show:
                indices = np.random.choice(len(x_train), max_points_to_show, replace=False)
                x_train = x_train[indices]
                y_train = y_train[indices]
                z_train = z_train[indices]
                class_0_mask = self.classifier.y_train[indices] == 0
                class_1_mask = self.classifier.y_train[indices] == 1
            else:
                class_0_mask = self.classifier.y_train == 0
                class_1_mask = self.classifier.y_train == 1

            # Punkty bez cukrzycy (klasa 0)
            ax.scatter(x_train[class_0_mask], y_train[class_0_mask], z_train[class_0_mask],
                       c='green', marker='o', s=20, label='Brak cukrzycy', alpha=0.7)

            # Punkty z cukrzycą (klasa 1)
            ax.scatter(x_train[class_1_mask], y_train[class_1_mask], z_train[class_1_mask],
                       c='red', marker='s', s=20, label='Cukrzyca', alpha=0.7)

        # Opisy osi
        ax.set_xlabel(self.feature_names[0].replace('_', ' ').title())
        ax.set_ylabel(self.feature_names[1].replace('_', ' ').title())
        ax.set_zlabel('P(cukrzyca)')
        ax.set_title('Funkcja gęstości prawdopodobieństwa - 3D')
        ax.legend()

        # Dodanie paska kolorów
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='P(cukrzyca)')

        if save_plot:
            filename = f'diabetes_3d_density_{self.feature_names[0]}_{self.feature_names[1]}.png'
            plt.savefig(filename, dpi=200, bbox_inches='tight')  # Niższe DPI
            print(f"Wykres 3D zapisany jako '{filename}'")

        plt.show()


def parse_data_input(input_str):
    """Parsuje dane wejściowe w formacie klucz=wartość"""
    data_dict = {}
    pairs = input_str.split(',')

    for pair in pairs:
        if '=' not in pair:
            continue
        key, value = pair.split('=', 1)
        key = key.strip()
        value = value.strip()

        try:
            # Spróbuj przekonwertować na float
            data_dict[key] = float(value)
        except ValueError:
            # Jeśli się nie uda, zostaw jako string
            data_dict[key] = value

    return data_dict


def main():
    parser = argparse.ArgumentParser(
        description='Inkrementalny Klasyfikator Ryzyka Cukrzycy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady użycia:

1. Początkowe trenowanie modelu:
   python cli_app.py --initial-train

2. Trenowanie z wybranymi cechami:
   python cli_app.py --initial-train --features age,bmi,glucose

3. Dodanie nowego punktu:
   python cli_app.py --add "age=45,bmi=28.5,outcome=1"

4. Predykcja:
   python cli_app.py --predict "age=35,bmi=25"

5. Tryb interaktywny:
   python cli_app.py --interactive

6. Wizualizacja:
   python cli_app.py --visualize
   

        """)

    parser.add_argument('--initial-train', action='store_true',
                        help='Początkowe trenowanie modelu')
    parser.add_argument('--features', type=str,
                        help='Cechy do użycia w modelu (oddzielone przecinkami)')
    parser.add_argument('--add', type=str, metavar='DATA',
                        help='Dodaj dane w formacie "klucz=wartość,klucz=wartość"')
    parser.add_argument('--predict', type=str, metavar='DATA',
                        help='Przewiduj ryzyko dla danych w formacie "klucz=wartość"')
    parser.add_argument('--predict-detailed', type=str, metavar='DATA',
                        help='Przewiduj z pokazaniem wpływających punktów')
    parser.add_argument('--visualize', action='store_true',
                        help='Wizualizuj model')
    parser.add_argument('--stats', action='store_true',
                        help='Pokaż statystyki modelu')
    parser.add_argument('--interactive', action='store_true',
                        help='Uruchom w trybie interaktywnym')
    parser.add_argument('--bandwidth', type=float, default=2.0,
                        help='Bandwidth dla jądra (domyślnie 2.0)')
    parser.add_argument('--model-path', type=str, default="models/diabetes_model.pkl",
                        help='Ścieżka do pliku modelu')
    parser.add_argument('--db-path', type=str, default="data/diabetes.db",
                        help='Ścieżka do bazy danych')
    parser.add_argument('--evaluate', action='store_true',
                        help='Przeprowadź pełną ewaluację modelu')
    parser.add_argument('--visualize3d', action='store_true',
                        help='Wizualizuj gęstość prawdopodobieństwa w 3D')

    args = parser.parse_args()

    # Utwórz instancję CLI
    cli = IncrementalDiabetesCLI(model_path=args.model_path, db_path=args.db_path)

    if args.interactive:
        cli.interactive_mode()
    elif args.initial_train:
        features = args.features.split(',') if args.features else ['age', 'bmi']
        features = [f.strip() for f in features]  # Usuń białe znaki
        cli.initial_train(bandwidth=args.bandwidth, features=features)
    elif args.add:
        data_dict = parse_data_input(args.add)
        if data_dict:
            cli.add_and_update(data_dict)
        else:
            print("Błąd parsowania danych. Użyj formatu: klucz=wartość,klucz=wartość")
    elif args.predict:
        data_dict = parse_data_input(args.predict)
        if data_dict:
            cli.predict_diabetes(data_dict)
        else:
            print("Błąd parsowania danych. Użyj formatu: klucz=wartość,klucz=wartość")
    elif args.predict_detailed:
        data_dict = parse_data_input(args.predict_detailed)
        if data_dict:
            cli.predict_diabetes(data_dict, show_influence=True)
        else:
            print("Błąd parsowania danych. Użyj formatu: klucz=wartość,klucz=wartość")
    elif args.visualize:
        cli.visualize_model(save_plot=True)
    elif args.stats:
        cli.show_stats()
    elif args.evaluate:
        cli.run_evaluation(test_size=0.3)
    elif args.visualize3d:
        cli.visualize_3d_density(save_plot=True)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()