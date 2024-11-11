# Analyse comparative des modèles de langage

## Première discussion

### Messages utilisateur :
> "Bonjour, je viens de finir ma terminal spé maths/physique, et je ne sais pas quoi faire de mon avenir"

> ...

> "j'ai peur que si je tente ma chance dans l'écologie je ne gagnerais pas assez d'argent, c'est pour ça. Mes parents veulent que je gagne beaucoup d'argent" 

> ...

> "je sais pas, je n'ai aucune idée de quoi faire maintenant"

#### Llama3.1:8b-instruct-q4_1
Remarque: <br>
Légers problèmes d'accord avec le genre, formulation légèrement maladroite par moment. Mentionne parfois des consignes du prompt entre parenthèses, comme le ton à employer
- Note aspect humain : 4/5
- Note pertinence : 4/5
- Note langue : 3/5

#### Qwen-2.5:3b-instruct
Remarque:<br>
Propose directement beaucoup (pas forcément un problème). Formulation légèrement maladroite, en de rares occasions. Respecte bien l'ambiance dictée par le prompt (chaleureux, gentil).
Aime bien faire des listes de suggestions. A des réponses assez ouvertes.
- Note aspect humain : 5/5
- Note pertinence : 4/5
- Note langue : 4.5/5

#### Llama-3.2-1B-Instruct
Remarque:<br>
Formulation maladroite, message long (pas forcément un problème), cale un remerciement à l'école de philosophie à la fin de son message (??), a du mal avec le vouvoiement et tutoiement.
- Note aspect humain : 3.5/5
- Note pertinence : 3.5/5
- Note langue : 2.5/5

#### TinyLlama-1.1B-Chat-v1.0
Remarque:<br>
S'implique trop dans son rôle (demande à prendre rendez-vous en physique avec l'étudiant, en donnant une adresse de lycée à Carcassonne). Répond aussi parfois en un gros paragraphe sans air, sans majuscule.
- Note aspect humain : 2/5
- Note pertinence : 1/5
- Note langue : 2.5/5

#### SmolLM2-1.7B-Instruct
Remarque:<br>
Utilisation de "Program:" ou "Human:" avant chaque ligne, discussion tout seul et part dans tous les sens ("Human: Oui, il y a un truand ici."), syntaxe et orthographe horribles. On dirait que ce qu'il dit a du sens, mais les phrases sont tellement fausses que même le sens se perd.
- Note aspect humain : 0/5
- Note pertinence : 1/5
- Note langue : 1/5


### Résultats de la première discussion /15
- llama3.1:8b-instruct-q4_1 : 11 points
- qwen-2.5:3b-instruct : 13.5 points
- Llama-3.2-1B-Instruct : 9.5 points
- TinyLlama-1.1B-Chat-v1.0 : 5.5 points
- SmolLM2-1.7B-Instruct : 2 points


## Deuxième discussion

### Messages utilisateur :

> "bonjour, je ne sais pas quoi faire de mon avenir, j'ai l'impression d'avoir gâché 2 ans de ma vie en faisant une licence que j'ai abandonné"

> ...

> "je sais pas, je n'ai aucune idée de quoi faire maintenant" 

> ...

#### Llama3.1:8b-instruct-q4_1
Remarque: <br>
Rassurant dans ses propos, encourageant (formidable), quelques erreurs de syntaxe, assez pour qu'on puisse fermer les yeux dessus
- Note aspect humain : 4.5/5
- Note pertinence : 4/5
- Note langue : 4/5

#### Qwen-2.5:3b-instruct
Remarque:<br>
Très rassurant, se met bien dans le rôle donné d'un conseiller d'orientation chaleureux. Arrive bien à rebondir pour diriger la discussion de par ses questions, sans que ces dernières ne paraissent forcées.
- Note aspect humain : 5/5
- Note pertinence : 5/5
- Note langue : 4.5/5

#### Llama-3.2-1B-Instruct
Remarque:<br>
Commence chaque message par une fomrule, ce qui est bizarre passé le premier message, légères fautes de syntaxe. Pose des questions originales et intéressantes, mais la formulation laise à désirer. Adopte une attitude positive. Suit trop les consignes du prompt à la lettre, ça se voit dans son plan de réponse (formulation type "Je vais maintenant ...").
- Note aspect humain : 3.5/5
- Note pertinence : 4/5
- Note langue : 3.5/5

#### TinyLlama-1.1B-Chat-v1.0
Remarque:<br>
Répond des phrases totalement hasardeuses par moment. Un bon plan de réponse semble se dégager des réponses, mais le nombre de mots qui ne sont pas à leur place dans les phrases rend toute compréhension ardue.
- Note aspect humain : 2/5
- Note pertinence : 2/5
- Note langue : 2/5

#### SmolLM2-1.7B-Instruct
Remarque:<br>
Une phrase sur deux n'a aucun sens, dommage parce que la deuxième phrase laisse penser que la requête utilisateur a été comprise. Mentionne encore parfois "AI:" ou "Human:" au début des phrases. Parle comme quelqu'un ayant un niveau extrêmement mauvais en francais, mais avec un énorme vocabulaire. Parfois dit une phrase, parfois 50.
- Note aspect humain : 1/5
- Note pertinence : 1.5/5
- Note langue : 1.5/5


### Résultats de la seconde discussion /15
- llama3.1:8b-instruct-q4_1 : 12.5 points
- qwen-2.5:3b-instruct : 14.5 points
- Llama-3.2-1B-Instruct : 11 points
- TinyLlama-1.1B-Chat-v1.0 : 6 points
- SmolLM2-1.7B-Instruct : 4 points


## Troisième discussion

### Messages utilisateur :

> "Bonjour, je m'appelle Thomas et je sors du lycée hôtelier, sans savoir quoi faire de ma vie. Je n'ai plus de motivation, à part l'écologie"

> ...

> "Mais est-ce que mon expérience, dans un domaine tout à fait différent, peut être utile ?"

> ...

#### Modèle: TinyLlama-1.1B-Chat-v1.0
Remarque :<br>
Réponses très confuses et difficilement compréhensibles. Le modèle se perd dans des considérations inutiles et des phrases décousues, sans jamais véritablement répondre à la question de Thomas. Manque de structure et de cohérence dans le discours, peu de lien avec la problématique posée.
- Note aspect humain : 1.5/5
- Note pertinence : 1/5
- Note langue : 1/5

#### Modèle: SmolLM2-1.7B-Instruct
Remarque :<br>
Réponses extrêmement courtes et dépourvues de contenu pertinent. Aucune réelle prise en compte de la question de Thomas, manque d'effort pour établir un dialogue utile ou offrir des conseils concrets. Peu d'engagement et absence de suivi de la problématique.
- Note aspect humain : 1/5
- Note pertinence : 0.5/5
- Note langue : 1/5

#### Modèle: llama3.1:8b-instruct-q4_1
Remarque :<br>
Propose une réponse qui prend en compte l'émotion de Thomas et valorise son intérêt pour l'écologie. Bonne tentative de créer un lien avec son expérience passée en hôtellerie. La réponse est structurée et encourageante, bien que parfois légèrement formelle.
- Note aspect humain : 4/5
- Note pertinence : 4/5
- Note langue : 4/5

#### Modèle: Llama-3.2-1B-Instruct
Remarque :<br>
Commence par une introduction engageante mais se perd ensuite dans des phrases mal articulées et des concepts flous. Manque de cohérence dans la transition entre les idées. Effort visible pour être encourageant, mais les réponses manquent de structure claire.
- Note aspect humain : 2.5/5
- Note pertinence : 2.5/5
- Note langue : 2.5/5

#### Modèle: qwen-2.5:3b-instruct
Remarque :<br>
Excellente compréhension des préoccupations de Thomas. Le modèle valorise son expérience en hôtellerie et propose des pistes concrètes pour la lier à son intérêt pour l'écologie. Le ton est engageant et encourageant, avec des suggestions pertinentes et adaptées. Le langage est fluide et bien structuré.
- Note aspect humain : 5/5
- Note pertinence : 5/5
- Note langue : 4.5/5

### Résultats de la troisième discussion /15
- TinyLlama-1.1B-Chat-v1.0 : 3.5 points
- SmolLM2-1.7B-Instruct : 2.5 points
- llama3.1:8b-instruct-q4_1 : 12 points
- Llama-3.2-1B-Instruct : 7.5 points
- qwen-2.5:3b-instruct : 14.5 points

### Scores totaux /45
1. **qwen-2.5:3b-instruct** : 42.5/45
2. **llama3.1:8b-instruct-q4_1** : 35.5/45
3. **Llama-3.2-1B-Instruct** : 28/45
4. **TinyLlama-1.1B-Chat-v1.0** : 15/45
5. **SmolLM2-1.7B-Instruct** : 8.5/45