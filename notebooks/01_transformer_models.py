import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from transformers import pipeline

    return (pipeline,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sentiment analysis
    """)
    return


@app.cell
def _(pipeline):
    classifier = pipeline("sentiment-analysis")
    classifier("I've been waiting for a Hugging Face course my whole life.")
    return (classifier,)


@app.cell
def _(classifier):
    classifier(["I've been waiting for a Hugging Face course my whole life.", "I hate it so much!"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zero shot classifier
    """)
    return


@app.cell
def _(pipeline):
    zeroshot_classifier = pipeline("zero-shot-classification")
    return (zeroshot_classifier,)


@app.cell
def _(zeroshot_classifier):
    zeroshot_classifier(
        "This is a course about the Transformers library",
        candidate_labels=["education", "politics", "business"],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Text generation
    """)
    return


@app.cell
def _(pipeline):
    generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M")
    return (generator,)


@app.cell
def _(generator):
    generator("Hier kommt der Panzermensch.")
    return


@app.cell
def _(generator):
    generator(["Szia. Hogy nézel ki?", "Hány nap egy hét?"], max_length=25, num_return_sequences=3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mask filling
    """)
    return


@app.cell
def _(pipeline):
    unmasker = pipeline("fill-mask")
    unmasker("This course will teach you all about <mask> models.", top_k=3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Named entity recognition
    """)
    return


@app.cell
def _(pipeline):
    ner = pipeline("ner")
    # ner("My name is Attila and I am a parent in Rego Park")
    ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Question answerer
    """)
    return


@app.cell
def _(pipeline):
    question_answerer = pipeline("question-answering")
    question_answerer("What is nine times three?")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summarization
    """)
    return


@app.cell
def _(pipeline):
    summarizer = pipeline("summarization")
    summarizer("""Weekend hack: HackerTube. I was annoyed with YouTube's recommendations so I had Claude filter down 43M comments over 20 years of HN to 6,454 videos that keep being recommended. *Significantly* better viewing experience, ~2 hours with Opus 5.5:
    https://hackertube.org


    - Excellent recs without any logins (state is kept in local browser session)
    - Thumbnail is a representative frame from middle of the video instead of the super duper A/B test hyper-optimized user uploads
    - Underlying data from the comment analysis as one zip file, point your Claude at it and ask for your own version
    - HackerTube Longs ✊

    A bit more on the process:
    I use Claude Code (desktop app for Mac), Opus 5.5 on Medium effort as my default. I only talk to Claude and have it do everything, the only "manual work" I helped with in tandem was in the browser (reminiscent of MenuGen): 1) I had to create some kind of API key from Google Cloud UI to query YouTube for video metadata, 2) I booted up a VPS box for serving and 3) I bought the domain name. Once the box was up I gave Claude the IP and told it update, secure, harden, deploy and optimize the code, and it did the whole thing.

    Many awesome projects that would have normally taken weeks/months even can now be few-shotted for breakfast. What else are we not thinking of.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Translation
    """)
    return


@app.cell
def _(pipeline):
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
