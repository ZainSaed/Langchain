from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough

load_dotenv()
prompt1 = PromptTemplate(
    template='Generate a joke about {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='Generate a explaination about {joke}',
    input_variables=['joke']
) 
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
parser = StrOutputParser()
sequential_chain = RunnableSequence(prompt1,model,parser)
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explaination': RunnableSequence(prompt2,model,parser)
})
final_chain = RunnableSequence(sequential_chain,parallel_chain)
result = final_chain.invoke({'topic':'AI'})
print(result)
