# data_loader.py
import pandas as pd
import sqlite3
import os
import argparse
from pathlib import Path


class DiabetesDataLoader:
    def __init__(self, db_path="data/diabetes.db"):
        self.db_path = db_path
        # Utwórz katalog jeśli nie istnieje
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_database()

    def init_database(self):
        """Inicjalizuje bazę danych z pełną strukturą tabeli"""
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
        print(f"Baza danych zainicjalizowana: {self.db_path}")

    def load_csv_data(self, csv_file_path, clear_existing=False):
        """
        Ładuje dane z pliku CSV do bazy danych

        Args:
            csv_file_path: ścieżka do pliku CSV
            clear_existing: czy wyczyścić istniejące dane
        """
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"Plik {csv_file_path} nie istnieje")

        # Wczytaj dane CSV
        try:
            df = pd.read_csv(csv_file_path)
            print(f"Wczytano {len(df)} rekordów z pliku {csv_file_path}")
        except Exception as e:
            raise Exception(f"Błąd wczytywania pliku CSV: {e}")

        # Sprawdź czy mamy wymagane kolumny
        required_columns = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Brakujące kolumny w pliku CSV: {missing_columns}")

        # Normalizuj nazwy kolumn (usunięcie spacji, małe litery)
        column_mapping = {
            'Pregnancies': 'pregnancies',
            'Glucose': 'glucose',
            'BloodPressure': 'blood_pressure',
            'SkinThickness': 'skin_thickness',
            'Insulin': 'insulin',
            'BMI': 'bmi',
            'DiabetesPedigreeFunction': 'diabetes_pedigree_function',
            'Age': 'age',
            'Outcome': 'outcome'
        }

        df_clean = df[required_columns].rename(columns=column_mapping)

        # Sprawdź dane - usuń wiersze z brakującymi wartościami BMI i Age (kluczowe dla klasyfikatora)
        initial_count = len(df_clean)
        df_clean = df_clean.dropna(subset=['bmi', 'age', 'outcome'])
        final_count = len(df_clean)

        if initial_count != final_count:
            print(f"Usunięto {initial_count - final_count} rekordów z brakującymi danymi")

        # Sprawdź typy danych
        print("\nStatystyki danych:")
        print(df_clean.describe())
        print(f"\nRozkład klasy docelowej (outcome):")
        print(df_clean['outcome'].value_counts())

        # Zapisz do bazy danych
        conn = sqlite3.connect(self.db_path)

        if clear_existing:
            conn.execute("DELETE FROM patients")
            print("Wyczyszczono istniejące dane")

        # Wstaw dane
        df_clean.to_sql('patients', conn, if_exists='append', index=False)
        conn.commit()

        # Sprawdź liczbę rekordów w bazie
        cursor = conn.execute("SELECT COUNT(*) FROM patients")
        total_records = cursor.fetchone()[0]
        conn.close()

        print(f"Zapisano {final_count} rekordów do bazy danych")
        print(f"Łączna liczba rekordów w bazie: {total_records}")

        return final_count

    def show_database_stats(self):
        """Pokazuje statystyki bazy danych"""
        conn = sqlite3.connect(self.db_path)

        # Podstawowe statystyki
        cursor = conn.execute("SELECT COUNT(*) FROM patients")
        total_count = cursor.fetchone()[0]

        if total_count == 0:
            print("Baza danych jest pusta")
            conn.close()
            return

        print(f"\n=== STATYSTYKI BAZY DANYCH ===")
        print(f"Łączna liczba rekordów: {total_count}")

        # Rozkład klasy docelowej
        cursor = conn.execute("SELECT outcome, COUNT(*) FROM patients GROUP BY outcome")
        outcome_dist = cursor.fetchall()
        print(f"\nRozkład klasy docelowej:")
        for outcome, count in outcome_dist:
            percentage = (count / total_count) * 100
            label = "Cukrzyca" if outcome == 1 else "Brak cukrzycy"
            print(f"  {label}: {count} ({percentage:.1f}%)")

        # Podstawowe statystyki liczbowe
        df = pd.read_sql_query("SELECT * FROM patients", conn)

        print(f"\nStatystyki opisowe (kluczowe zmienne):")
        key_vars = ['age', 'bmi', 'glucose', 'blood_pressure']
        for var in key_vars:
            if var in df.columns:
                print(f"  {var.upper()}:")
                print(f"    Średnia: {df[var].mean():.2f}")
                print(f"    Odchylenie std: {df[var].std():.2f}")
                print(f"    Min: {df[var].min():.2f}, Max: {df[var].max():.2f}")

        conn.close()

    def export_sample_data(self, output_file="sample_data.csv", sample_size=100):
        """Eksportuje przykładowe dane do pliku CSV"""
        conn = sqlite3.connect(self.db_path)

        query = f"SELECT * FROM patients ORDER BY RANDOM() LIMIT {sample_size}"
        df = pd.read_sql_query(query, conn)
        conn.close()

        df.to_csv(output_file, index=False)
        print(f"Wyeksportowano {len(df)} przykładowych rekordów do {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Ładowanie danych o cukrzycy do bazy danych')

    parser.add_argument('--load-csv', type=str, metavar='CSV_FILE',
                        help='Ścieżka do pliku CSV z danymi')
    parser.add_argument('--clear-db', action='store_true',
                        help='Wyczyść istniejące dane przed ładowaniem')
    parser.add_argument('--stats', action='store_true',
                        help='Pokaż statystyki bazy danych')
    parser.add_argument('--export-sample', type=str, metavar='OUTPUT_FILE',
                        help='Eksportuj przykładowe dane do pliku CSV')
    parser.add_argument('--db-path', type=str, default='data/diabetes.db',
                        help='Ścieżka do pliku bazy danych (domyślnie: data/diabetes.db)')

    args = parser.parse_args()

    loader = DiabetesDataLoader(db_path=args.db_path)

    if args.load_csv:
        try:
            loader.load_csv_data(args.load_csv, clear_existing=args.clear_db)
            print("\n✅ Dane zostały pomyślnie załadowane!")
        except Exception as e:
            print(f"❌ Błąd podczas ładowania danych: {e}")
            return 1

    if args.stats:
        loader.show_database_stats()

    if args.export_sample:
        try:
            loader.export_sample_data(args.export_sample)
        except Exception as e:
            print(f"❌ Błąd podczas eksportu: {e}")

    if not any([args.load_csv, args.stats, args.export_sample]):
        parser.print_help()

    return 0


if __name__ == "__main__":
    exit(main())