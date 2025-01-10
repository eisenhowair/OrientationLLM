import json
import csv
import sys

def convert_json_to_csv(input_json_file, output_csv_file):
    try:
        # Lecture du fichier JSON avec détection explicite de l'encodage
        with open(input_json_file, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        
        # Récupération de toutes les clés uniques
        all_keys = set()
        for obj in data.values():
            all_keys.update(obj.keys())
        headers = list(all_keys)
        
        # Écriture dans le fichier CSV avec utf-8-sig pour ajouter le BOM
        with open(output_csv_file, 'w', encoding='utf-8-sig', newline='') as csv_file:
            writer = csv.DictWriter(csv_file,
                                  fieldnames=headers,
                                  delimiter=',',
                                  quoting=csv.QUOTE_NONE,
                                  escapechar=None)
            
            # Écriture des en-têtes
            writer.writeheader()
            
            # Écriture des données
            for obj in data.values():
                cleaned_obj = {}
                for k, v in obj.items():
                    if v is None:
                        cleaned_obj[k] = ''
                    else:
                        # Remplace les \n ET les virgules par des pipes
                        value = str(v).replace('\n', '|').replace(',', '|')
                        # Gestion des pipes multiples qui pourraient être créés
                        while '||' in value:
                            value = value.replace('||', '|')
                        # Supprime les pipes en début et fin si présents
                        value = value.strip('|')
                        cleaned_obj[k] = value
                
                writer.writerow(cleaned_obj)
            
        print(f"Conversion réussie! Le fichier CSV a été créé : {output_csv_file}")
        
    except FileNotFoundError:
        print(f"Erreur: Le fichier JSON '{input_json_file}' n'a pas été trouvé.")
    except json.JSONDecodeError:
        print(f"Erreur: Le fichier '{input_json_file}' n'est pas un JSON valide.")
    except Exception as e:
        print(f"Une erreur est survenue: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input.json output.csv")
        sys.exit(1)
    
    input_json_file = sys.argv[1]
    output_csv_file = sys.argv[2]
    
    convert_json_to_csv(input_json_file, output_csv_file)