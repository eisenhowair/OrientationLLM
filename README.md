
<h3 align="center">Projet "Développement d'une intelligence artificielle pour l'orientation étudiante"</h3>
<p align="center">L'objectif de ce projet est de créer un chatbot dirigé par un modèle de langage dans le but d'aider des étudiants dans leur orientation.</p>
<br/>


<details>
  <summary><strong>Sommaire</strong></summary>
  <ol>
    <li>
      <a href="#à-propos-du-projet">À propos du projet</a>
      <ul>
        <li><a href="#technologies">Technologies</a></li>
      </ul>
    </li>
    <li>
      <a href="#pour-commencer">Pour commencer</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#utilisation">Utilisation</a></li>
    <li><a href="#contacts">Contacts</a></li>
  </ol>
</details>


## À propos du projet

Dans ce projet se trouvent plusieurs travaux utilisant les modèles de langage. On y retrouve un travail sur la comparaison de modèles, deux versions de chatbot pour l'orientation étudiante et d'autres petits travaux comme des essais de RAG.


### Technologies
[![Static Badge](https://img.shields.io/badge/Langchain-blue?style=for-the-badge&link=https%3A%2F%2Fwww.langchain.com%2F)](https://www.langchain.com/)

[![Static Badge](https://img.shields.io/badge/LLAMAINDEX-lightblue?style=for-the-badge)
](https://www.llamaindex.ai/)

[![Static Badge](https://img.shields.io/badge/Chainlit-pink?style=for-the-badge&link=https%3A%2F%2Fwww.langchain.com%2F)
](https://chainlit.io/)

[![Static Badge](https://img.shields.io/badge/FAISS-white?style=for-the-badge)
](https://ai.meta.com/tools/faiss/)

## Pour commencer

### Installation

Suivez ces étapes pour mettre en place le projet.
Depuis la racine du projet :

1. Installez Ollama et récupérer le modèle
	- `curl -fsSL https://ollama.com/install.sh | sh`  (source : [Ollama](https://ollama.com/download))
	- `ollama pull qwen2.5:3b-instruct`

2. Certaines applications ont leur propre **requirements.txt** qui contient les les modules nécessaires.
Installez les modules python avec :
	- `pip install -r requirements.txt`


Enfin, chaque dossier disposant d'un fichier **.env.example** nécessite un fichier **.env**, dont le contenu sera indiqué dans le **.env.example**.

## Utilisation

Les implémentations se trouve dans le dossier Realisation. Chaque dossier représente une ou plusieurs applications. Vous trouverez un README dans chaque dossier pour l'utilisation et les exigences supplémentaires.

| Dossier | Description |
|-----------|-----------|
| [/EnsembleModel](https://github.com/eisenhowair/OrientationLLM/tree/main/Realisations/EnsembleModel) | Comparaison de modèles |
| [/multiAgent](https://github.com/eisenhowair/OrientationLLM/tree/main/Realisations/multiAgent) | Chatbot |
| [/OrientationFormations ](https://github.com/eisenhowair/OrientationLLM/tree/main/Realisations/OrientationFormations ) | Chatbot |
| [/scraping](https://github.com/eisenhowair/OrientationLLM/tree/main/Realisations/scraping) | Scraping des formations UHA |
| [/experimentations  ](https://github.com/eisenhowair/OrientationLLM/tree/main/experimentations) | Tests  |

## Contacts

Elias Mouaheb - elias.mouaheb@uha.fr

Théo Nicod - theo.nicod@uha.fr
