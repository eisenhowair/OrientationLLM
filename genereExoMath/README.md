<!-- GETTING STARTED -->
<a name="readme-top"></a>
# Modèle à des fins de Génération et Aide à la Résolution d'exercices mathématiques

## Introduction

Le but des programmes de ce dossier est de mettre en place un générateur d'exercices de mathématiques, qui permet ensuite d'aider à la résolution 
de l'exercice, en donnant des indices, ou en corrigeant l'exercice après un certain nombre de tentatives.

Trois versions du programme existent, mais exo_math_2prompts.py sert de version la plus aboutie, avec _prep_message_chainlit_prompts.py_ pour répartir le code.
Les 2 autres fichiers de code représentent des tentatives, chacun avec ses défauts. 

### Installation

1. Avant de pouvoir lancer le programme, il est nécessaire d'installer les dépendances et logiciels requis, trouvables dans le fichier requirements.txt du dossier parent

```sh
  pip install -r ../requirements.txt
  ```

2. Il faudra ensuite télécharger le modèle correspondant

```sh
  ollama pull llama3:instruct
  ```

3. Enfin, se créer un fichier .env avec les variables LITERAL_API_KEY et CHAINLIT_AUTH_SECRET, avec la première nécessitant de se créer un compte sur LiteralAI (puis aller dans Settings -> General -> Default Key), et la seconde qui est trouvable en tapant 
 ```sh
  chainlit create-secret
  ```


## Fonctionnement

### Initialisation du programme

Au lancement du programme, 2 variables sont initialisées pour la mémoire : une globale pour quitter et revenir à la discussion, une courte pour la conversation en cours. Ces variables sont enrichies à chaque message pour générer des exercices, sans surcharger le modèle.

### Génération d'exercice

La fonction setup_exercice_model() commence par récupérer la discussion actuelle, puis les loisirs de l'utilisateur. Elle va utiliser ses données, ainsi qu'un prompt spécifiquement écrit pour que le modèle soit à même de créer des exercices suivant certaines règles, et va les passer à un objet Runnable à renvoyer.

Le Runnable renvoyé est utilisé pour récupérer une réponse du modèle, qui sera donc un exercice de mathématique. L'exercice est enregistré dans la session, pour que le correcteur y ait accès facilement. Enfin le flag "compris" lui aussi dans la session est passé à False, pour appeler le correcteur.

### Résolution d'exercice


La fonction setup_aide_model() prépare les prompts spécifiques pour aider l'utilisateur et permet au modèle de communiquer efficacement. Après plusieurs tentatives (infructueuses) où le modèle a donné des indices pour aider,il donne la réponse. La fonction verifie_comprehension demande si l'utilisateur a compris et passe au prochain exercice. Ensuite, la mémoire courte est réinitialisée.

### Autres

Le programme dispose aussi de fonctions permettant de quitter la discussion pour la reprendre plus tard, ou de changer l'âge de l'utilisateur aux yeux du modèle. Il est à noter que bien que l'IA ait conscience de l'âge, elle n'arrive pas à générés des exercices qui y sont pertinents.


## Intérêt par rapport aux précédentes versions

Le fichier exo_math_3prompts.py utilise un prompt pour aider l'utilisateur avec les indices, et un autre pour donner la réponse. Cela créait des malentendus dans certains cas, en plus de ne pas être très fluide.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
