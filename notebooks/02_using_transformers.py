import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from transformers import pipeline
    classifier = pipeline("sentiment-analysis")
    classifier(
        [
            "Baby, oh yeah!",
            "Dating is screwed."
        ]
    )
    return


@app.cell
def _():
    from transformers import AutoTokenizer
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return checkpoint, tokenizer


@app.cell
def _(tokenizer):
    raw_inputs = [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
    inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
    print(inputs)
    return inputs, raw_inputs


@app.cell
def _(checkpoint):
    from transformers import AutoModel
    model = AutoModel.from_pretrained(checkpoint)
    return (model,)


@app.cell
def _(inputs, model):
    outputs = model(**inputs)
    print(outputs.last_hidden_state.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model head for sequence classification
    """)
    return


@app.cell
def _(checkpoint, inputs):
    from transformers import AutoModelForSequenceClassification
    classifier_model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    classifier_outputs = classifier_model(**inputs)
    print(classifier_outputs)
    print(classifier_outputs.logits)
    return (classifier_outputs,)


@app.cell
def _(classifier_outputs):
    import torch
    classifier_predictions = torch.nn.functional.softmax(classifier_outputs.logits, dim=-1)
    classifier_predictions
    return (classifier_predictions,)


@app.cell
def _(classifier_predictions, raw_inputs):
    import pandas as pd
    classifier_predictions_df = pd.DataFrame(
        classifier_predictions.detach().cpu().numpy(),
        index=raw_inputs,
        columns=["negative", "positive"],
    )
    classifier_predictions_df
    return


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
