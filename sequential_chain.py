from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate 5 wealthy cities of {topic}',
    input_variables=['topic']
) 
prompt2 = PromptTemplate(
    template='Average income \n{text}',
    input_variables=['text']
) 
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
parser = StrOutputParser()
chain = prompt1 | model | parser | prompt2 | model | parser
result = chain.invoke({'topic':'Pakistan'})
print(result)
chain.get_graph().print_ascii()

