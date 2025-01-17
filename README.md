
# Projekt ReadMe

Dieses Repository enthält Skripte und Notebooks, die verschiedene Aufgaben im Bereich des algorithmischen Handels und der Analyse von Handelsstrategien abdecken. Nachfolgend finden Sie eine Übersicht der enthaltenen Dateien und deren Funktionen.

---

## Dateienübersicht

### 1. **BTC_Volatility.ipynb**
- **Beschreibung**: 
  - Dieses Jupyter Notebook analysiert die Volatilität von Bitcoin (BTC) basierend auf historischen Marktdaten.
  - Es berechnet statistische Metriken wie Preisveränderungen und Volatilität und erstellt Visualisierungen zur Darstellung von Trends und Mustern.

---

### 2. **ML_Algo_Training_Script_V6.py**
- **Beschreibung**: 
  - Python-Skript zur Entwicklung und zum Training von Reinforcement-Learning-Modellen für Handelsstrategien.
  - **Hauptfunktionen**:
    - Datenabfrage (z. B. von Bitcoin-Daten aus einer Datenbank).
    - Feature-Engineering und Datenvorbereitung.
    - Training von Modellen und Speicherung von Leistungsmetriken.
    - Logging und Visualisierung der Verluste während des Trainingsprozesses.
  - Unterstützt CUDA für die Hardwarebeschleunigung.

---

### 3. **ML_Algo_V5_GC_vrm_2.ipynb**
- **Beschreibung**: 
  - Jupyter Notebook zur Feinabstimmung und Evaluierung von maschinellen Lernmodellen für Handelsstrategien.
  - Enthält Datenvorbereitung, Modellimplementierung und die Berechnung von Strategiemetriken wie der Sharpe Ratio und der Profitabilität.
  - Wurde für das Training auf den gefilterten Datensatz verwendet

---

### 4. **Strategy_Generator_V10.py**
- **Beschreibung**: 
  - Python-Skript zur Generierung automatisierter Handelsstrategien durch Optimierung von Parameterkombinationen.
  - **Hauptfunktionen**:
    - Validierung von Strategieparametern (z. B. RSI-Perioden, ADX-Schwellen).
    - Anwendung von Filtern zur Auswahl profitabler Strategien.
    - Speicherung von Strategien in einer Datenbank.
    - Erstellung von Visualisierungen zur Analyse von Indikatoren und Strategien.

---

### 5. **strategy_plot.py**
- **Beschreibung**: 
  - Python-Skript zur Visualisierung von Handelsstrategien und deren Profitabilität.
  - **Hauptfunktionen**:
    - Erstellung von Histogrammen und Balkendiagrammen, um die Verteilung der Profitabilität zu analysieren.
    - Analyse und Darstellung der besten Strategien basierend auf Leistungsmetriken.
    - Berechnung der durchschnittlichen Indikatorwerte der Top-Strategien.

---

### 6. **multiprocessing.py**
- **Beschreibung**: 
  - Skript zur parallelen Ausführung mehrerer Instanzen des Strategie-Generators (`Strategy_Generator_V10.py`).
  - **Hauptfunktionen**:
    - Startet mehrere Instanzen des Strategie-Generators in separaten Terminals.
    - Speichert individuelle Logs für jede Instanz.
    - Nützlich zur gleichzeitigen Verarbeitung großer Datenmengen oder zur Optimierung von Strategien.

---

## Nutzungshinweise
- Stellen Sie sicher, dass alle erforderlichen Bibliotheken und Abhängigkeiten installiert sind, bevor Sie die Skripte oder Notebooks ausführen.
- Konfigurieren Sie die Datenbankverbindungen in den Skripten (`db_connection_params`), um den Zugriff auf die notwendigen Daten sicherzustellen.
- Verwenden Sie das Skript `multiprocessing.py`, um Berechnungen parallel auszuführen und die Effizienz zu erhöhen.

## Voraussetzungen
- **Programmiersprache**: Python 3.8 oder höher.
- **Bibliotheken**: `pandas`, `numpy`, `psycopg2`, `matplotlib`, `torch`, `gym`, `tqdm`, `argparse`, u.a.
- **Hardware**: CUDA-kompatible GPU (optional, für hardwarebeschleunigtes Training).

## Autor
Dieses Projekt wurde erstellt, um algorithmische Handelsstrategien zu entwickeln und zu analysieren.

---
