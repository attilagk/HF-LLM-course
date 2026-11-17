import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from transformers import pipeline
    classifier = pipeline("sentiment-analysis")
    classifier(
        [
            "Baby, oh yeah$!",
            "Dating is screwed."
        ]
    )
    return


@app.cell
def _():
    from transformers import AutoTokenizer
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return (tokenizer,)


@app.cell
def _(tokenizer):
    raw_inputs = [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
    inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
    print(inputs)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
