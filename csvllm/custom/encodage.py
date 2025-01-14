import chardet
import pandas as pd

csv_path = "../../formations/formations.csv"

# Détecte l'encodage du fichier
with open(csv_path, 'rb') as file:
    raw = file.read()
    result = chardet.detect(raw)
    print(f"L'encodage détecté est : {result['encoding']}")

# Utilisez ensuite l'encodage détecté
df = pd.read_csv(csv_path, encoding=result['encoding'])