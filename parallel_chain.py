from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

prompt1 = PromptTemplate(
    template='Summarize the following text in 1 line {text}',
    input_variables=['text']
) 
prompt2 = PromptTemplate(
    template='Create one question from that summarized text \n{text}',
    input_variables=['text']
)
prompt3 = PromptTemplate(
    template='Combine the summary and question in single document \n notes ->{notes} and quiz ->{quiz}',
    input_variables=['notes','quiz']
)  

model1 = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
model2 = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

parser = StrOutputParser()
parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser ,
    'quiz' : prompt2 | model2 | parser
})
merge_chain = prompt3 | model1 | parser
chain = parallel_chain | merge_chain
result = chain.invoke({'text':'Pakistan is only Muslim country in the world that has Nuclear Weapons . Pakistan is economically weak but militarily strong . Pakistanis are religious people and they are muslims'})
print(result)
chain.get_graph().print_ascii()

