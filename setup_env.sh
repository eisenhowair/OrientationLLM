#!/bin/bash
sudo chown -R $(whoami):$(whoami) ~/OrientationLLM/ProjetLLM # pour avoir les droits sur l'environnement
export TMPDIR='/var/tmp'
pip3 install --no-cache-dir -r requirements.txt
chainlit create-secret

echo "Si problème d'espace sur le périphérique, relancer ce fichier"
# si problème de place: rm -rf ~/.cache/pip
# si problème d'installation dans le venv, supprimer le venv et le recréer manuellement