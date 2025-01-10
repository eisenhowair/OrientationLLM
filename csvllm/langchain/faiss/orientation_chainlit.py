import chainlit as cl
from orientation import OrientationChatbot, OrientationCriteria

chatbot = None
user_responses = {}
initial_questions_done = False

@cl.on_chat_start
async def start():
    global chatbot, initial_questions_done
    
    chatbot = OrientationChatbot(vector_store_dir="vector_store")
    
    try:
        await cl.Message(content="Chargement des données...").send()
        chatbot.load_data('../../../formations/formations.csv')
        
        await cl.Message(content="Je suis là pour vous aider à trouver la formation idéale ! Je vais vous poser quelques questions.").send()
        
        #première question sur le diplôme
        diplome = await cl.AskUserMessage(
            content="Quel niveau d'études recherchez-vous ? (licence/master/doctorat)",
        ).send()

        user_responses["diplome"] = diplome['output'].lower()
        
        #deuxième question sur le domaine
        domaine = await cl.AskUserMessage(
            content="Quel domaine d'études vous intéresse ? (informatique/sciences/droit/économie/lettres/langues)",
        ).send()
        
        user_responses["domaine"] = domaine['output'].lower()
        
        #troisième question sur le type de formation
        formation = await cl.AskUserMessage(
            content="Quel type de formation préférez-vous ? (initial/alternance/formation continue)",
        ).send()
        
        user_responses["type_formation"] = formation['output'].lower()

        await cl.Message(content="Bien ! Laissez-moi quelques secondes pour trouver les formations qui vous correspondent...").send()
        
        #création des critères avec les réponses
        criteria = OrientationCriteria(
            diplome=user_responses["diplome"],
            domaine_interet=user_responses["domaine"],
            type_formation=user_responses["type_formation"]
        )
        
        result = chatbot.process_orientation(criteria)
        
        #envoi des résultats
        await cl.Message(
            content=f"Voici les formations qui correspondent à vos critères :\n\n{result}"
        ).send()

        initial_questions_done = True
        
        await cl.AskUserMessage(
            content="Avez-vous des questions ?"
        ).send()
        
        
    except Exception as e:
        await cl.Message(content=f"Une erreur s'est produite : {str(e)}").send()

@cl.on_message
async def main(message: cl.Message):
    if initial_questions_done:
        try:
            criteria = OrientationCriteria(
                diplome=user_responses["diplome"],
                domaine_interet=user_responses["domaine"],
                type_formation=user_responses["type_formation"],
                user_query=message.content
            )
            
            result = chatbot.process_orientation(criteria)
            await cl.Message(content=result).send()
            
        except Exception as e:
            await cl.Message(content=f"Une erreur s'est produite : {str(e)}").send()

@cl.on_chat_end
def end():
    global chatbot, user_responses
    chatbot = None
    user_responses = {}