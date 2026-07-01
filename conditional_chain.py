from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    PydanticOutputParser,
)
from langchain_core.runnables import (
    RunnableParallel,
    RunnableBranch,
    RunnableLambda,
)
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()


class Feedback(BaseModel):
    sentiment: Literal["Positive", "Negative"] = Field(
        description="Sentiment of the customer feedback"
    )


model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser = StrOutputParser()
pydantic_parser = PydanticOutputParser(pydantic_object=Feedback)

classifier_prompt = PromptTemplate(
    template="""
Determine whether the following customer feedback is Positive or Negative.

Feedback:
{feedback}

{format_instructions}
""",
    input_variables=["feedback"],
    partial_variables={
        "format_instructions": pydantic_parser.get_format_instructions()
    },
)

classifier_chain = classifier_prompt | model | pydantic_parser

parallel_chain = RunnableParallel(
    {
        "feedback": RunnableLambda(lambda x: x["feedback"]),
        "sentiment": classifier_chain,
    }
)

positive_prompt = PromptTemplate(
    template="""
The following customer left a positive review.

Customer Feedback:
{feedback}

Write a warm and professional thank-you response.
""",
    input_variables=["feedback"],
)

negative_prompt = PromptTemplate(
    template="""
The following customer left a negative review.

Customer Feedback:
{feedback}

Write a polite apology and assure the customer that the issue will be addressed.
""",
    input_variables=["feedback"],
)

branch_chain = RunnableBranch(
    (
        lambda x: x["sentiment"].sentiment == "Positive",
        positive_prompt | model | parser,
    ),
    (
        lambda x: x["sentiment"].sentiment == "Negative",
        negative_prompt | model | parser,
    ),
    RunnableLambda(lambda _: "Couldn't determine the sentiment."),
)

chain = parallel_chain | branch_chain

result = chain.invoke(
    {
        "feedback": "This is a terrible phone."
    }
)

print(result)

chain.get_graph().print_ascii()