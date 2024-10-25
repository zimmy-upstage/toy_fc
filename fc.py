import json
from typing import Any, Dict, List, Optional, Union

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_upstage import ChatUpstage as Chat
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from claim import extracted_claimed_facts
from kg import build_kg
from search import search_context

MAX_SEAERCH_RESULTS = 5

MODEL_NAME = "llama-3.1-70b-versatile"
MODEL_NAME = "solar-pro"
ddg_search = DuckDuckGoSearchResults()


from typing import Any, Dict, List, Optional, Union

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate


def verify_facts(
    claimed_facts: List[Dict[str, Any]],
    context: str,
    kg: Dict[str, Any],
    confidence_threshold: float,
    llm: Optional[Chat] = Chat(model=MODEL_NAME),
) -> Dict[str, Dict[str, Any]]:
    """
    Verify the claimed facts against the knowledge graph and context.

    Args:
        claimed_facts (List[Dict[str, Any]]): The list of extracted claimed facts.
        context (str): The context information retrieved from the search.
        kg (Dict[str, Any]): The constructed knowledge graph.
        confidence_threshold (float): The confidence threshold for fact verification.
        llm (Optional[Chat]): The language model to use for verification, if needed.

    Returns:
        Dict[str, Dict[str, Any]]: Verified facts with status, confidence, and explanation.
        The structure is: {fact_id: {"claimed": str, "status": str, "confidence": float, "explanation": str}}
    """

    kg_str = json.dumps(kg, indent=2)
    verified_facts = {}

    # TODO: change to truth-o-meter rating
    # valid_statuses = {"true", "false", "probably true", "probably false", "not sure"}

    for i, fact in enumerate(claimed_facts):
        verification_result = verify_one_fact(context, kg_str, fact, llm)

        # status = verification_result.get("status", "not sure").lower()
        # confidence = verification_result.get("confidence", 0.0)
        # explanation = verification_result.get("explanation", "")

        # Validate status
        # if status not in valid_statuses:
        #     status = "not sure"

        # Validate confidence score
        # if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
        #     confidence = 0.0

        # # Apply confidence threshold
        # if confidence < confidence_threshold:
        #     status = "not sure"

        verified_facts[str(i)] = {
            "claimed": fact["statement"],
            **verification_result,
        }

    return verified_facts


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def verify_one_fact(context, kg_str, fact, llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Determine the Truth-O-Meter rating for a given claim based on its alignment with a provided knowledge graph, reflecting the statement's accuracy level.

Utilize the following rating system to classify the claim:

- **TRUE**: The statement is accurate with no significant information omitted.
- **MOSTLY TRUE**: The statement is accurate, but requires clarification or additional context.
- **HALF TRUE**: The statement is partially accurate, omitting vital details or context.
- **MOSTLY FALSE**: The statement contains some truth, but overlooks critical facts that change the overall impression.
- **FALSE**: The statement is inaccurate.
- **PANTS ON FIRE**: The statement is not only inaccurate but also makes a ridiculous claim.

# Steps
1. **Analyze the Claim**: Review the statement to understand its assertions.
2. **Reasoning**: Evaluate how the statement's details align with or diverge from the facts. Consider any missing context or overlooked information.
3. **Assign a Rating**: Based on the comparison and reasoning, choose the most appropriate Truth-O-Meter rating.

# Examples

**Example 1:**

- **Claim**: "X is the largest producer of Y."
- **Reasoning**: While X produces a significant amount, Z is verified as the largest, contradicting the claim.
- **Rating**: FALSE

**Example 2:**

- **Claim**: "A supports B according to government statistics."
- **Reasoning**: The claim is accurate but omits critical conditions attached to A’s support.
- **Rating**: MOSTLY TRUE

# Notes

- Pay attention to the potential for missing context or partial truths.
- Use reasoning to substantiate the chosen rating before making a conclusion.
- Consider any relevant factual elements that exist outside the explicit nodes present in the knowledge graph.""",
            ),
            (
                "human",
                """Verify the following claimed fact using the provided knowledge graph and context.
Claimed Fact: {claim}

Knowledge Graph:
{kg}

Additionally, assign a confidence score between 0.0 and 1.0 that reflects the certainty of the categorization.
Provide the result in a JSON object with the following structure:
{{
  "Rating": <Rating>,
  "confidence": <Confidence>,
  "explanation": "<BRIEF_EXPLANATION>"
}}

Ensure that:
1. The categorization is based on the information in the knowledge graph and context.
2. The confidence score accurately reflects the certainty of the categorization.
3. The explanation briefly justifies the verification decision and confidence score.""",
            ),
        ]
    )

    output_parser = JsonOutputParser()
    chain = prompt | llm | output_parser

    verification_result = chain.invoke(
        {
            "claim": fact["statement"],
            # TODO: add tags
            "kg": kg_str,
            # "context": context, #TODO : add context
        }
    )

    return verification_result


def fc(
    text: str,
    context: Optional[str] = None,
    kg: Optional[Dict] = None,
    verify_sources: bool = True,
    confidence_threshold: float = 0.7,
    llm=Chat(model=MODEL_NAME),
) -> Dict[str, Dict[str, Union[str, float, bool]]]:
    """
    Function to perform fact checking on a given text using a knowledge graph.

    Args:
        text (str): The text to be checked.
        context (Optional[str]): Additional context to be used for fact checking.
        kg (Optional[Dict]): The knowledge graph to be used for fact checking.
        verify_sources (bool): Whether to verify the sources of the information.
        confidence_threshold (float): The confidence threshold for the fact checking.
        llm (Optional[Chat]): The language model to use for processing, if needed.

    Returns:
        Dict[str, Dict[str, Union[str, float, bool]]]: The fact checked information.
        The structure is: {fact_id: {"claimed": str, "verified": bool, "confidence": float, "explanation": str}}
    """

    print("\n--- Starting Fact Checking Process ---")
    print(f"Input text: {text}")

    print("\nStep 1: Extracting claimed facts")
    claimed_facts = extracted_claimed_facts(text, llm)
    print(f"Extracted {len(claimed_facts)} claimed facts:")
    # for i, fact in enumerate(claimed_facts):
    #     print(f"  {i+1}. {fact['entity']} {fact['relation']} {fact['value']}")

    if context is None:
        print("\nStep 2: Searching for relevant context")
        context = search_context(text, claimed_facts, ddg_search, llm)
        print(f"Retrieved context (first 100 characters): {context[:100]}...")
    else:
        print("\nStep 2: Using provided context")

    if kg is None:
        print("\nStep 3: Building knowledge graph")
        kg = build_kg(claimed_facts, context, llm)
        print(f"Built knowledge graph with {len(kg)} entities")
    else:
        print("\nStep 3: Using provided knowledge graph")

    print("\nStep 4: Verifying facts")
    verified_facts = verify_facts(claimed_facts, context, kg, confidence_threshold, llm)
    print(f"Verified {len(verified_facts)} facts:")
    for fact_id, result in verified_facts.items():
        print(f"  Fact {fact_id}:")
        print(f"    Claimed: {result['claimed']}")
        print(f"    Status: {result['status']}")
        print(f"    Confidence: {result['confidence']}")
        print(
            f"    Explanation: {result['explanation'][:100]}..."
        )  # Truncate long explanations

    print("\n--- Fact Checking Process Completed ---")

    # Final step
    print("\nStep 5: Adding fact-check annotations to the original text")
    fact_checked_text = add_fact_check_to_text(text, verified_facts, llm)
    print("Fact-checked text generated")

    return verified_facts, fact_checked_text


def add_fact_check_to_text(text, verified_facts, llm=Chat(model=MODEL_NAME)):
    # First, let's create a mapping of claimed facts to their verifications
    fact_map = {fact["claimed"]: fact for fact in verified_facts.values()}

    # Now, let's ask the LLM to annotate the original text
    system_message = HumanMessage(
        content="""
    You are an AI assistant tasked with adding fact-check annotations to a given text.
    For each fact in the text that has been verified, add an inline annotation 
    right after the fact, using the following format:
    [Fact: <STATUS> (Confidence: <CONFIDENCE>) - <BRIEF_EXPLANATION>]
    Where <STATUS> is True, False, or Unsure, <CONFIDENCE> is the confidence score,
    and <BRIEF_EXPLANATION> is a very short explanation.
    """
    )

    human_message = HumanMessage(
        content=f"""
    Original text:
    {text}

    Verified facts:
    {fact_map}

    Please add fact-check annotations to the original text based on the verified facts.
    """
    )

    response = llm([system_message, human_message])

    return response.content
