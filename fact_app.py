import gradio as gr
from pathlib import Path

from pydantic import BaseModel


class FactCheckResult(BaseModel):
    claim: str
    fact_rating: str
    reference: str
    suggestion: str


def load_sample(path="sample"):
    path = Path(path)
    articles = []
    articles_list = list(path.glob("*.txt"))
    for article_path in articles_list:
        with open(article_path, "r", encoding="utf-8") as file:
            articles.append(file.read())
    labels = [article_path.stem for article_path in articles_list]
    return articles, labels


def highlight_article(article: str, output_claims: list[FactCheckResult]):
    # Sort claims by their position in the article to handle overlapping claims
    claims_with_pos = []
    for claim in output_claims:
        start_idx = article.find(claim.claim)
        if start_idx != -1:  # Only include claims found in article
            claims_with_pos.append(
                (start_idx, start_idx + len(claim.claim), claim, claim.fact_rating)
            )
    claims_with_pos.sort()  # Sort by start position

    # Build highlighted segments
    highlighted = []
    current_pos = 0

    for start_idx, end_idx, claim, fact_rating in claims_with_pos:
        # Add non-highlighted text before claim
        if current_pos < start_idx:
            highlighted.append((article[current_pos:start_idx], None))
        # Add highlighted claim
        highlighted.append((article[start_idx:end_idx], fact_rating))
        current_pos = end_idx

    # Add remaining text after last claim
    if current_pos < len(article):
        highlighted.append((article[current_pos:], None))

    return highlighted


with gr.Blocks() as demo:
    samples, sample_labels = load_sample(path="sample")
    sample_fact_check_results = [
        FactCheckResult(
            claim="Upstage is South Korea's most successful AI company",
            fact_rating="TRUE",
            reference="TBU",
            suggestion="TBU",
        ),
        FactCheckResult(
            claim=" they were struggling under the weight of proofreading over 60 articles per day.",
            fact_rating="FALSE",
            reference="TBU",
            suggestion="TBU",
        ),
    ]

    with gr.Tabs("Fact Check") as tabs:
        with gr.TabItem("Article", id=0):
            with gr.Column():
                source_article = gr.Textbox(interactive=True)
                with gr.Row():
                    example = gr.Examples(
                        samples,
                        inputs=source_article,
                        example_labels=sample_labels,
                    )
                fact_check_button = gr.Button("Fact Check", variant="primary")

        with gr.TabItem("Fact Check", id=1):
            highlighted = highlight_article(samples[0], sample_fact_check_results)
            with gr.Row():
                article_placeholder = gr.HighlightedText(
                    highlighted,
                    color_map={
                        "claim": "green",
                        "label": "red",
                        "TRUE": "green",
                        "FALSE": "red",
                    },
                    interactive=False,
                )
                with gr.Column(scale=0.5):
                    for i in range(len(sample_fact_check_results)):
                        with gr.Accordion(f"Claim{i+1}"):
                            claim_placeholder = gr.Textbox(
                                sample_fact_check_results[i].claim,
                                label="Claim",
                                lines=3,
                            )
                            label_placeholder = gr.Textbox(
                                sample_fact_check_results[i].fact_rating,
                                label="Fact Rating",
                            )
                            reference_placeholder = gr.Textbox(
                                sample_fact_check_results[i].reference,
                                label="Reference",
                            )
                            suggestion_placeholder = gr.Textbox(
                                sample_fact_check_results[i].suggestion,
                                label="Suggestion",
                            )

    def change_tab(id):
        return gr.Tabs(selected=id)

    fact_check_button.click(change_tab, gr.Number(1, visible=False), tabs)

if __name__ == "__main__":
    demo.launch(server_port=30626, share=True)
