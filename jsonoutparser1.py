from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()


model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
Parser = JsonOutputParser()
template = PromptTemplate(
    template='Give me the name,date and month of countries that got independece in 1947 \n {format_instruction} ',
    input_variables=[],
    partial_variables={'format_instruction':Parser.get_format_instructions}
)
chain = template | model | Parser
result = chain.invoke({})
print(result)