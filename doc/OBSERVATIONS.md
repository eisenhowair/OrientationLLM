Avec un prompt où les Shots étaient dans la variable string du prompt elle-même,
le modèle avait tendance à avoir des formulations étranges, comme mentionner les roles dans le chat en citant même son prompt

Ce problème a été réglé en mettant les Shots dans les fonctions langchain prévus pour, notamment ```FewShotChatMessagePromptTemplate()```. Problème restant, le modèle, en de rares occasions, confondait les exemples des Shots avec l'utilisateur à qui il parlait, lui prêtant ainsi des formations qui ne sont pas les siennes. 
Une séparation plus propre des shots et des instructions, avec une ligne telle que "Les exemples suivants sont des conseils passés :" entre les 2 semble résoudre le problème.

Le modèle aime commencer chaque message par un bonjour (sans doute dû à l'entrainement), même après une consigne dans le prompt lui disant de ne pas le faire. La mention du ton chaleureux pour mettre l'étudiant à l'aise smeble être une réussite, le modèle est très doux.

Par contre le modèle bombarde de questions à chaque message, ça rend la conversation moins naturelle.
