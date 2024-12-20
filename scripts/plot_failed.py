import marimo

__generated_with = "0.8.15"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import altair as alt
    import pandas as pd
    from pathlib import Path
    return Path, alt, mo, pd


@app.cell
def __(Path, alt, mo, pd):
    filepath = Path("compare_simple_adjoin.csv")
    df = pd.read_csv(filepath, sep="\t")

    chart = alt.Chart(df)
    y1 = chart.encode(alt.X("x:Q"), alt.Y("y1:Q"))
    y2 = chart.encode(alt.X("x:Q"), alt.Y("y1:Q"), alt.Y2("y2:Q"))
    mo.ui.altair_chart(
        y1.mark_line() + y2.mark_line()
    )
    return chart, df, filepath, y1, y2


if __name__ == "__main__":
    app.run()
