from llama_index.query_pipeline import (
    QueryPipeline as QP,
    Link,
    InputComponent,
)
from llama_index.query_engine.pandas import PandasInstructionParser
from llama_index.prompts import PromptTemplate

from llama_index.llms import Ollama

import pandas as pd

df = pd.read_csv("../../../formations/formations.csv")


instruction_str = (
    "1. Convert the query to executable Python code using Pandas.\n"
    "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
    "3. The code should represent a solution to the query.\n"
    "4. PRINT ONLY THE EXPRESSION.\n"
    "5. Do not quote the expression.\n"
)

pandas_prompt_str = (
    "You are working with a pandas dataframe in Python.\n"
    "The name of the dataframe is `df`.\n"
    "This is the result of `print(df.head())`:\n"
    "{df_str}\n\n"
    "Follow these instructions:\n"
    "{instruction_str}\n"
    "Query: {query_str}\n\n"
    "Expression:"
)
response_synthesis_prompt_str = (
    "Given an input question, synthesize a response from the query results.\n"
    "Query: {query_str}\n\n"
    "Pandas Instructions (optional):\n{pandas_instructions}\n\n"
    "Pandas Output: {pandas_output}\n\n"
    "Response: "
)
""" ------------------ """
""" ------------------ """
# instruction_str = (
#     "1. Convertis la requête en code Python exécutable en utilisant Pandas.\n"
#     "2. La dernière ligne du code doit être une expression Python qui peut être appelée avec la fonction `eval()`.\n"
#     "3. Le code doit représenter une solution à la requête.\n"
#     "4. PRINT UNIQUEMENT L'EXPRESSION.\n"
#     "5. Ne mets pas l'expression entre guillemets.\n"
# )

# pandas_prompt_str = (
#     "Tu travailles avec un dataframe pandas en Python.\n"
#     "Le nom du dataframe est `df`.\n"
#     "Voici le résultat de `print(df.head())` :\n"
#     "{df_str}\n\n"
#     "Suis ces instructions :\n"
#     "{instruction_str}\n"
#     "Requête : {query_str}\n\n"
#     "Expression :"
# )

# response_synthesis_prompt_str = (
#     "Étant donné une question en entrée, synthétise une réponse à partir des résultats de la requête.\n"
#     "Requête : {query_str}\n\n"
#     "Instructions Pandas (optionnelles) :\n{pandas_instructions}\n\n"
#     "Résultat Pandas : {pandas_output}\n\n"
#     "Réponse : "
# )

""" ------------------ """
""" ------------------ """

# instruction_str = (
#     "1. Convertis la requête en un code python exécutable en utilisant Pandas.\n"
#     "2. Pour les recherches de formations, utiliser df[df[`column`].str.contains(`search_term`, case=False, na=False)]\n"
#     "3. Pour filtrer par niveau, utiliser df['Diplôme'] == 'niveau_recherché'\n"
#     "4. Le résultat doit inclure au minimum les colonnes `Nom de la formation`, `Diplôme`, et `Secteur(s) d'activité`\n"
#     "5. AFFICHE SEULEMENT L'EXPRESSION.\n"
# )

# pandas_prompt_str = (
#     "Tu analyses un DataFrame de formations universitaires avec les colonnes suivantes:\n"
#     "- Discipline(s): domaine principal de la formation\n"
#     "- Diplôme: type de diplôme (B.U.T., Master, etc.)\n"
#     "- Nom de la formation: intitulé complet de la formation\n"
#     "- Régime(s) d'études: modalités d'études possibles\n"
#     "- Secteur(s) d'activité: secteurs professionnels visés\n"
#     "Le nom du DataFrame est `df`.\n"
#     "Données exemple:\n"
#     "{df_str}\n\n"
#     "Instructions:\n"
#     "{instruction_str}\n"
#     "Question: {query_str}\n\n"
#     "Expression Pandas à exécuter:"
# )

# response_synthesis_prompt_str = (
#     "Tu es un conseiller d'orientation universitaire expert.\n"
#     "Question de l'étudiant: {query_str}\n\n"
#     "Requête Pandas exécutée: {pandas_instructions}\n\n"
#     "Résultats trouvés: {pandas_output}\n\n"
#     "Instructions:\n"
#     "1. Commence par indiquer le nombre de formations trouvées\n"
#     "2. Présente chaque formation de manière claire avec:\n"
#     "   - Le nom de la formation\n"
#     "   - Le type de diplôme\n"
#     "   - Les modalités d'études possibles\n"
#     "3. Ajoute des conseils pertinents sur les débouchés\n"
#     "Réponse:"
# )

# response_synthesis_prompt_str = (
#     "Given an input question, synthesize a response from the query results.\n"
#     "Query: {query_str}\n\n"
#     "Pandas Instructions (optional):\n{pandas_instructions}\n\n"
#     "Pandas Output: {pandas_output}\n\n"
#     "Response: "
# )




pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
    instruction_str=instruction_str, df_str=df.head(5)
)
pandas_output_parser = PandasInstructionParser(df)
response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)

llm = Ollama(model="qwen2.5:7b",
             temperature=0, 
             request_timeout=120.0,
             )

qp = QP(
    modules={
        "input": InputComponent(),
        "pandas_prompt": pandas_prompt,
        "llm1": llm,
        "pandas_output_parser": pandas_output_parser,
        "response_synthesis_prompt": response_synthesis_prompt,
        "llm2": llm,
    },  
    verbose=True,
)
qp.add_chain(["input", "pandas_prompt", "llm1", "pandas_output_parser"])
qp.add_links(
    [
        Link("input", "response_synthesis_prompt", dest_key="query_str"),
        Link(
            "llm1", "response_synthesis_prompt", dest_key="pandas_instructions"
        ),
        Link(
            "pandas_output_parser",
            "response_synthesis_prompt",
            dest_key="pandas_output",
        ),
    ]
)
# add link from response synthesis prompt to llm2
qp.add_link("response_synthesis_prompt", "llm2")


response = qp.run(
    query_str="Je cherche une licence informatique",
)

print(response.message.content)

