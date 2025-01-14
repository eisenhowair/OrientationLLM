import chainlit as cl
from orientation import OrientationChatbot, OrientationCriteria

chatbot = None
user_responses = {}
initial_questions_done = False

@cl.on_chat_start
async def start():
    global chatbot, initial_questions_done, user_responses

    chatbot = OrientationChatbot(vector_store_dir="vector_store")
    chatbot.load_data('../../../formations/formations.csv')

    try:
        await cl.Message(content="Bonjour, je suis là pour vous aider à trouver la formation qui vous correspond ! Commençons par quelques questions ...").send()
        
        # Première question
        diplome_msg = await cl.AskUserMessage(
            content="Quel niveau d'études recherchez-vous ? (licence/master/doctorat...)", 
        ).send()

        if diplome_msg:
            user_responses["diplome"] = diplome_msg['output'].lower()
        else:
            return
        
        # Deuxième question 
        domaine_msg = await cl.AskUserMessage(
            content="Quel domaine d'études vous intéresse ? (informatique/sciences/droit/économie/lettres/langues)",
        ).send()

        if domaine_msg:
            user_responses["domaine"] = domaine_msg['output'].lower()
        else:
            return

        # Créer les critères
        criteria = OrientationCriteria(
            diplome=user_responses["diplome"],
            domaine_interet=user_responses["domaine"],
        )

        # Traiter la demande avec un message d'attente
        await cl.Message(content="Je recherche les formations qui correspondent à vos critères...").send()
        
        # Faire le traitement dans un bloc try séparé
        try:
            result = await cl.make_async(chatbot.process_orientation)(criteria)
            await cl.Message(content=f"{result}").send()
            initial_questions_done = True
            await cl.Message(content="Avez-vous des questions ?").send()
            
        except Exception as e:
            await cl.Message(content=f"Erreur lors de la recherche : {str(e)}").send()

    except Exception as e:
        print(f"Error: {str(e)}")
        await cl.Message(content="Une erreur s'est produite, veuillez réessayer.").send()

@cl.on_message
async def main(message: cl.Message):
    if initial_questions_done and chatbot:
        try:
            #await cl.Message(content="Je traite votre demande...").send()
            
            criteria = OrientationCriteria(
                diplome=user_responses.get("diplome", ""),
                domaine_interet=user_responses.get("domaine", ""),
                user_query=message.content
            )
            
            result = await cl.make_async(chatbot.process_orientation)(criteria)
            await cl.Message(content=result).send()
            
        except Exception as e:
            print(f"Error: {str(e)}")
            await cl.Message(content="Une erreur s'est produite, veuillez réessayer.").send()