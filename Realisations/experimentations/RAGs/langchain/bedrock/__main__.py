from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_community.chat_models import BedrockChat
import boto3

session = boto3.Session(
    aws_access_key_id='',
    aws_secret_access_key='',
    region_name='eu-central-1'
)

bedrock_client = session.client(service_name='bedrock-runtime')

llm=BedrockChat(model_id="anthropic.claude-3-sonnet",
            client=bedrock_client,
            model_kwargs={"temperature":0.})

agent = create_csv_agent(
    llm,
    "heart_failure_clinical_records.csv",
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    allow_dangerous_code=True
)

agent.run("how many people have more than 50 years old?")
