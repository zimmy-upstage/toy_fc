import claim
from fc import MODEL_NAME


from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_upstage import ChatUpstage as Chat


from typing import Any, Dict, List, Optional


# 3. **Significant**: Is the statement important or impactful? Avoid minor slip-ups or insignificant errors.
# 4. **Viral**: Is the statement likely to be passed on and repeated by others?

SYSTEM_PROMPT = """Extract statements to fact-check based on specific criteria and assign them a tag indicating their suitability for fact-checking.

Consider the following criteria for each statement:
1. **Verifiable**: Is the statement rooted in a fact that can be verified? Avoid statements that are opinions, hyperboles, or exaggerated rhetorical statements.
2. **Misleading**: Does the statement seem misleading or sound incorrect in any way?
3. **Curiosity**: Would a typical person hear or read the statement and wonder or question if it's true?

# Steps

1. **Extract Statements**: Carefully identify and extract verbatim statements eligible for fact-checking.
2. **Assess and Tag**: Evaluate each extracted statement against the above criteria and assign tags.

# Output Format

- Each extracted statement should be followed by its assigned tag in JSON format.
- Give tags from ["Verifiable", "Misleading", "Curiosity"]
- Json structure: array of objects with keys "statement" and "tags" like below
```json
[
    {{
        "statement": #YOUR_STATEMENT#,
        "tags": #YOUR_TAGS#
    }},
    {{
        "statement": #YOUR_STATEMENT#,
        "tags": #YOUR_TAGS#
    }}
]
```

# Examples

**Input:** 
"The unemployment rate is the lowest it's been in 50 years. In contrast, the stock market is bad."

**Output:** 
[{{"statement": "The unemployment rate is the lowest it's been in 50 years.", "tags": ["Verifiable"]}}, {{"statement": "In contrast, the stock market is bad.", "tags": ["Misleading", "Curiosity"]}}]

# Notes

- Focus on statements that fulfill multiple criteria for a more robust fact-checking process.
- Ensure that opinion-based statements are not extracted for fact-checking purposes."""

USER_PROMPT = """Extract as many claims as possible from the following text:

# Passage
{input_text}"""


def extracted_claimed_facts(
    text: str, llm: Optional[Chat] = Chat(model=MODEL_NAME)
) -> List[Dict[str, Any]]:
    """
    Extract claimed facts from the given text, including entities and their relationships.

    Args:
        text (str): The input text to extract facts from.
        llm (Optional[Chat]): The language model to use for extraction, if needed.

    Returns:
        List[Dict[str, Any]]: A list of extracted facts, where each fact is represented as a dictionary.
    """

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_PROMPT,
            ),
            (
                "human",
                USER_PROMPT.format(input_text=text),
            ),
        ]
    )

    # Create the output parser
    output_parser = JsonOutputParser()

    # Create the chain
    chain = prompt | llm | output_parser

    # Run the chain
    result = chain.invoke({"input_text": text})

    return result
