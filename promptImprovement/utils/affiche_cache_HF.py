import os
from huggingface_hub import scan_cache_dir

# Obtenir les informations du cache
cache_info = scan_cache_dir()
print(cache_info)
# Afficher les modèles téléchargés
print("Modèles téléchargés :")
for repo in cache_info.repos:
    print(f"\nModèle: {repo.repo_id}")
    print(f"Taille: {repo.size_on_disk / 1024 / 1024:.2f} MB")
    print(f"Dernière modification: {repo.last_modified}")
    print("Fichiers:")
    for ref in repo.refs:
        print(f"  - {ref}")
        # for file in ref.files:
        # print(f"    * {file.file_name} ({file.size_on_disk / 1024 / 1024:.2f} MB)")
