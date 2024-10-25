from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_upstage import ChatUpstage as Chat
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed


from typing import Any, Dict, List, Optional

MODEL_NAME = "solar-pro"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def build_kg(
    claimed_facts: List[Dict[str, Any]],
    context: str,
    llm: Optional[Chat] = Chat(model=MODEL_NAME),
) -> Dict[str, Any]:
    """
    Build a knowledge graph from claimed facts and context information.

    Args:
        claimed_facts (List[Dict[str, Any]]): The list of extracted claimed facts.
        context (str): The context information retrieved from the search.
        llm (Optional[Chat]): The language model to use for processing, if needed.

    Returns:
        Dict[str, Any]: The constructed knowledge graph with source information.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert in building knowledge graphs.
                Your task is to analyze the given context and construct a knowledge graph, using the claimed facts only as inspiration for the schema without assuming their truth.
                Include source information for each fact.""",
            ),
            (
                "human",
                """Given the following context and claimed facts, construct a knowledge graph.
                Assume all information in the context is true, but use the claimed facts only as hints for the types of relations to look for.

Context:
{context}

Claimed Facts (use only as schema hints):
{claimed_facts}

Construct the knowledge graph as a JSON object where keys are entities and values are dictionaries of relations. Each relation should have a "value" and a "source" (a relevant quote from the context).

Example format:
{{
  "Entity1": {{
    "relation1": {{
      "value": "Value1",
      "source": "Relevant quote from context"
    }},
    "relation2": {{
      "value": "Value2",
      "source": "Another relevant quote"
    }}
  }},
  "Entity2": {{
    ...
  }}
}}

Ensure that:
1. All information comes from the context, not the claimed facts.
2. Each fact has a source quote from the context.
3. The schema is inspired by, but not limited to, the relations in the claimed facts.

Construct the knowledge graph:""",
            ),
        ]
    )

    output_parser = JsonOutputParser()
    chain = prompt | llm | output_parser

    facts_str = "\n".join([fact["statement"] for fact in claimed_facts])

    kg = chain.invoke({"context": context, "claimed_facts": facts_str})

    return kg
