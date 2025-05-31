# database.py
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path


class DiabetesDatabase:
    def __init__(self, db_path="data/diabetes.db"):
        self.db_path = db_path
        # Utwórz katalog jeśli nie istnieje
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_database()

    def init_database(self):
        """Inicjalizuje bazę danych z pełną strukturą"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pregnancies INTEGER,
                glucose REAL,
                blood_pressure REAL,
                skin_thickness REAL,
                insulin REAL,
                bmi REAL NOT NULL,
                diabetes_pedigree_function REAL,
                age REAL NOT NULL,
                outcome INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def add_record(self, age, bmi, outcome, pregnancies=None, glucose=None,
                   blood_pressure=None, skin_thickness=None, insulin=None,
                   diabetes_pedigree_function=None):
        """
        Dodaje rekord do bazy danych

        Args:
            age: wiek (wymagane)
            bmi: BMI (wymagane)
            outcome: wynik (0/1) (wymagane)
            pozostałe parametry opcjonalne
        """
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """INSERT INTO patients 
               (pregnancies, glucose, blood_pressure, skin_thickness, 
                insulin, bmi, diabetes_pedigree_function, age, outcome) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (pregnancies, glucose, blood_pressure, skin_thickness,
             insulin, bmi, diabetes_pedigree_function, age, outcome)
        )
        conn.commit()
        conn.close()

    def get_training_data(self, features=['age', 'bmi'], target='outcome'):
        """
        Pobiera dane treningowe z bazy

        Args:
            features: lista cech do pobrania (domyślnie age, bmi)
            target: nazwa kolumny docelowej (domyślnie outcome)

        Returns:
            DataFrame z danymi treningowymi
        """
        conn = sqlite3.connect(self.db_path)

        # Sprawdź dostępne kolumny
        available_features = []
        for feature in features:
            cursor = conn.execute(f"PRAGMA table_info(patients)")
            columns = [row[1] for row in cursor.fetchall()]
            if feature in columns:
                available_features.append(feature)
            else:
                print(f"Ostrzeżenie: kolumna '{feature}' nie istnieje w bazie danych")

        if not available_features:
            raise ValueError("Brak dostępnych cech w bazie danych")

        # Buduj zapytanie SQL
        feature_columns = ', '.join(available_features)
        query = f"SELECT {feature_columns}, {target} FROM patients WHERE "

        # Dodaj warunki dla wymaganych kolumn (usuń NULL-e)
        conditions = []
        for feature in available_features:
            conditions.append(f"{feature} IS NOT NULL")
        conditions.append(f"{target} IS NOT NULL")

        query += " AND ".join(conditions)

        df = pd.read_sql_query(query, conn)
        conn.close()

        print(f"Pobrano {len(df)} rekordów z bazy danych")
        print(f"Cechy: {available_features}")
        print(f"Rozkład klasy docelowej: {df[target].value_counts().to_dict()}")

        return df

    def get_full_training_data(self):
        """Pobiera wszystkie dostępne dane treningowe"""
        features = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
                    'insulin', 'bmi', 'diabetes_pedigree_function', 'age']
        return self.get_training_data(features=features)

    def get_basic_training_data(self):
        """Pobiera podstawowe dane treningowe (age, bmi)"""
        return self.get_training_data(features=['age', 'bmi'])

    def get_enhanced_training_data(self):
        """Pobiera rozszerzone dane treningowe (age, bmi, glucose, blood_pressure)"""
        return self.get_training_data(features=['age', 'bmi', 'glucose', 'blood_pressure'])

    def get_statistics(self):
        """Zwraca statystyki bazy danych"""
        conn = sqlite3.connect(self.db_path)

        # Podstawowe statystyki
        cursor = conn.execute("SELECT COUNT(*) FROM patients")
        total_count = cursor.fetchone()[0]

        if total_count == 0:
            conn.close()
            return {"total_records": 0, "message": "Baza danych jest pusta"}

        # Rozkład klasy docelowej
        cursor = conn.execute("SELECT outcome, COUNT(*) FROM patients GROUP BY outcome")
        outcome_dist = {str(outcome): count for outcome, count in cursor.fetchall()}

        # Statystyki opisowe dla kluczowych zmiennych
        df = pd.read_sql_query("SELECT age, bmi, glucose, blood_pressure, outcome FROM patients", conn)
        conn.close()

        stats = {
            "total_records": total_count,
            "outcome_distribution": outcome_dist,
            "age_stats": {
                "mean": df['age'].mean(),
                "std": df['age'].std(),
                "min": df['age'].min(),
                "max": df['age'].max()
            },
            "bmi_stats": {
                "mean": df['bmi'].mean(),
                "std": df['bmi'].std(),
                "min": df['bmi'].min(),
                "max": df['bmi'].max()
            }
        }

        # Dodaj statystyki glucose i blood_pressure jeśli dostępne
        if not df['glucose'].isna().all():
            stats["glucose_stats"] = {
                "mean": df['glucose'].mean(),
                "std": df['glucose'].std(),
                "min": df['glucose'].min(),
                "max": df['glucose'].max()
            }

        if not df['blood_pressure'].isna().all():
            stats["blood_pressure_stats"] = {
                "mean": df['blood_pressure'].mean(),
                "std": df['blood_pressure'].std(),
                "min": df['blood_pressure'].min(),
                "max": df['blood_pressure'].max()
            }

        return stats

    def clear_database(self):
        """Czyści wszystkie dane z bazy"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM patients")
        conn.commit()
        conn.close()
        print("Baza danych została wyczyszczona")


# Funkcje pomocnicze dla kompatybilności wstecznej
def create_sample_data(db_path="data/diabetes.db", num_samples=50):
    """Tworzy przykładowe dane dla testów"""
    db = DiabetesDatabase(db_path)

    # Generuj losowe dane
    np.random.seed(42)  # dla powtarzalności

    for i in range(num_samples):
        age = np.random.normal(45, 15)  # średnia 45, odchylenie 15
        age = max(18, min(80, age))  # ograniczenia wieku

        bmi = np.random.normal(28, 6)  # średnia 28, odchylenie 6
        bmi = max(15, min(50, bmi))  # ograniczenia BMI

        # Prawdopodobieństwo cukrzycy zależy od wieku i BMI
        risk_score = (age - 30) * 0.02 + (bmi - 25) * 0.05
        probability = 1 / (1 + np.exp(-risk_score))  # funkcja logistyczna
        outcome = 1 if np.random.random() < probability else 0

        # Opcjonalne cechy
        glucose = np.random.normal(120 if outcome == 1 else 95, 20)
        glucose = max(70, min(200, glucose))

        blood_pressure = np.random.normal(85 if outcome == 1 else 75, 15)
        blood_pressure = max(60, min(120, blood_pressure))

        pregnancies = np.random.poisson(2) if np.random.random() < 0.5 else None  # tylko część danych

        db.add_record(
            age=age,
            bmi=bmi,
            outcome=outcome,
            glucose=glucose,
            blood_pressure=blood_pressure,
            pregnancies=pregnancies
        )

    print(f"Utworzono {num_samples} przykładowych rekordów")
    return db


if __name__ == "__main__":
    # Test bazy danych
    print("=== Test bazy danych ===")

    # Utwórz przykładowe dane
    db = create_sample_data(num_samples=20)

    # Pokaż statystyki
    stats = db.get_statistics()
    print(f"\nStatystyki: {stats}")

    # Pobierz dane treningowe
    df_basic = db.get_basic_training_data()
    print(f"\nPodstawowe dane treningowe: {df_basic.shape}")
    print(df_basic.head())

    df_enhanced = db.get_enhanced_training_data()
    print(f"\nRozszerzone dane treningowe: {df_enhanced.shape}")
    print(df_enhanced.head())