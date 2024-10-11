#!/bin/bash

# Mise à jour et installation de pip
sudo apt update && sudo apt install python3-pip -y

# Résolution des éventuels problèmes de dpkg
sudo dpkg --configure -a

# Installation de python3.8-venv
sudo apt install python3.8-venv -y

# Téléchargement et installation de Ollama
curl -fsSL https://ollama.com/install.sh | sh
systemctl enable ollama
ollama pull llama3:instruct

# Création de l'environnement virtuel
python3 -m venv ProjetLLM

# Instructions pour l'utilisateur sur comment activer l'environnement
echo "Utilise la commande 'source ProjetLLM/bin/activate' avant de lancer 'setup_env' pour activer l'environnement virtuel."
