from llama_index.llms import Ollama

llm = Ollama(
    model="llama3.2",
    base_url="http://localhost:11434",
    request_timeout=30.0
)

response = llm.complete("Bonjour, comment vas-tu ?")
print(response)