from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
# chat template
chat_template = ChatPromptTemplate([
    ('system','You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

chat_history = []
# load chat history
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())


# create prompt
prompt = chat_template.invoke({'chat_history':chat_history, 'query':'Where is my refund'})

result = model.invoke(prompt)
print(result.content)