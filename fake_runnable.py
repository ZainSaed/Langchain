import random
from abc import ABC, abstractmethod


class Runnable(ABC):

    @abstractmethod
    def invoke(self, input_data):
        pass


class NakliLLM(Runnable):

    def __init__(self):
        print("LLM Created")

    def invoke(self, prompt):
        response_list = [
            "Islamabad is the capital of Pakistan.",
            "PSL stands for Pakistan Super League.",
            "AI stands for Artificial Intelligence."
        ]

        return {
            "response": random.choice(response_list)
        }


class NakliPromptTemplate(Runnable):

    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def invoke(self, input_dict):
        return self.template.format(**input_dict)


class NakliStrOutputParser(Runnable):

    def invoke(self, input_data):
        return {
            "response": input_data["response"]
        }


class RunnableConnector(Runnable):

    def __init__(self, runnable_list):
        self.runnable_list = runnable_list

    def invoke(self, input_data):
        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)

        return input_data


template1 = NakliPromptTemplate(
    template="Write a poem about {topic}",
    input_variables=["topic"]
)

template2 = NakliPromptTemplate(
    template="Explain the following statement:\n{response}",
    input_variables=["response"]
)

llm = NakliLLM()
parser = NakliStrOutputParser()

chain1 = RunnableConnector([
    template1,
    llm,
    parser
])

chain2 = RunnableConnector([
    template2,
    llm,
    parser
])

final_chain = RunnableConnector([
    chain1,
    chain2
])

result = final_chain.invoke({
    "topic": "Pakistan"
})

print(result)