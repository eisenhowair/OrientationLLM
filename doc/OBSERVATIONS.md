Avec un prompt où les Shots étaient dans la variable string du prompt elle-même,
le modèle avait tendance à avoir des formulations étranges, comme mentionner les roles dans le chat en citant même son prompt

Ce problème a été réglé en mettant les Shots dans les fonctions langchain prévus pour, notamment ```FewShotChatMessagePromptTemplate()```. Problème restant, le modèle, en de rares occasions, confondait les exemples des Shots avec l'utilisateur à qui il parlait, lui prêtant ainsi des formations qui ne sont pas les siennes. 
Une séparation plus propre des shots et des instructions, avec une ligne telle que "Les exemples suivants sont des conseils passés :" entre les 2 semble résoudre le problème.

Le modèle aime commencer chaque message par un bonjour (sans doute dû à l'entrainement), même après une consigne dans le prompt lui disant de ne pas le faire. La mention du ton chaleureux pour mettre l'étudiant à l'aise smeble être une réussite, le modèle est très doux.

Par contre le modèle bombarde de questions à chaque message, ça rend la conversation moins naturelle.

La version prompt_no_domain_no_formation_v3 gère mieux le cas où l'utilisateur n'est pas sûr de ce qu'il souhaite faire.
Donc il y a moins un bombardement de question qui mettrait mal à l'aise. En plus de ça, enlever les directives par
rapport à la salutation fait que le modèle gère lui-même, et répète moins souvent bonjour (2 messages sur 10, par rapport à 8/10 en lui disant de ne pas dire bonjour au-delà du premier message
)

prompt_no_domain_no_formation_v3 a quand du mal à proposer des solutions, et reste dans l'enchainement question-> question,
alors que les questions n'ont pas d'intérêt (beaucoup trop précises)


Après mise en place de l'ensemble de modèles, les comparaisons sont plus faciles. Ainsi, les modèles nemotron sont MAUVAIS, répondant hors sujet, ou répétant des indications du prompt ("AI:"), parfois même répondant en anglais. Le modèle llama3:instruct fait souvent des fautes d'orthographes.

A première observation, les modèles llama semblent mieux réagir au 0 shot qu'au Few-shot, mais peut-être est-ce dû à une implémentation pas assez bonne des shots.

Pour l'instant les modèles llama3.1 et 3.2 semblent se démarquer (les comparaisons sont réalisées en 0 shot pour l'instant)

Les modèles mistral-small et mistral-nemo sont tous 2 trop lourd, ne présentant pas de version Ollama sous 5, voire 8 Gb
