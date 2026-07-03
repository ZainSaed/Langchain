from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel

load_dotenv()
prompt1 = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='Generate a linkeding post about {topic}',
    input_variables=['topic']
) 
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
parser = StrOutputParser()
parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1,model,parser),
    'post': RunnableSequence(prompt2,model,parser)
})
result = parallel_chain.invoke({'topic':'AI'})
print(result['tweet'])
print(result['post'])
