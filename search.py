from fc import MODEL_NAME


from langchain_core.prompts import ChatPromptTemplate
from langchain_upstage import ChatUpstage as Chat


from typing import Any, Dict, List, Optional

SYSTEM_PROMPT = "You are an expert at generating concise and relevant search keywords. Your task is to analyze the given text and extracted facts, then produce a list of 3-5 search keywords or short phrases that would be most effective for finding additional context and verification information."
USER_PROMPT = """Given the following text and extracted facts, generate a list of 3-5 search keywords or short phrases:

Text: {text}

Extracted Facts:
{facts}

Provide only the keywords or short phrases, separated by commas."""


def search_context(
    text: str,
    claimed_facts: List[Dict[str, Any]],
    search_tool: Any,
    llm: Optional[Chat] = Chat(model=MODEL_NAME),
) -> str:
    """
    Search for relevant information using claimed facts.

    Args:
        text (str): The original input text.
        claimed_facts (List[Dict[str, Any]]): The list of extracted claimed facts.
        search_tool (Any): The search tool to use for finding information (e.g., DuckDuckGoSearchResults).
        llm (Optional[Chat]): The language model to use for processing, if needed.

    Returns:
        str: The relevant context information found from the search.
    """

    # Step 1: Generate search keywords
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_PROMPT,
            ),
            (
                "human",
                USER_PROMPT,
            ),
        ]
    )

    facts_str = "\n".join(
        [
            fact["statement"]
            for fact in claimed_facts
        ]
    )
    keywords_response = llm.invoke(prompt.format(text=text, facts=facts_str))

    # Parse the keywords from the response
    keywords = [kw.strip() for kw in keywords_response.content.split(",") if kw.strip()]

    # Step 2: Perform search using the generated keywords
    search_query = " ".join(keywords)
    search_results = search_tool.run(search_query)

    # Step 3: Return the search results
    return search_results
