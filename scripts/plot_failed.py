import marimo

__generated_with = "0.8.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def __():
    import marimo as mo
    import altair as alt
    import pandas as pd
    import os
    from pathlib import Path
    return Path, alt, mo, os, pd


@app.cell(hide_code=True)
def __(mo):
    button_reload = mo.ui.button(label="Reload")
    button_reload
    return button_reload,


@app.cell
def __(button_reload, mo):
    button_reload

    file_browser = mo.ui.file_browser(multiple=False, restrict_navigation=True, filetypes=[".csv"])
    file_browser
    return file_browser,


@app.cell
def __(Path, file_browser, mo):
    if file_browser.value:
        filepath = file_browser.value[0].path
    else:
        filepath = mo.cli_args().get("file")
    filepath = Path(filepath)
    return filepath,


@app.cell
def __(alt, filepath, mo, os, pd):
    df = pd.read_csv(filepath, sep="\t")

    chart = alt.Chart(df, title=f"{filepath.relative_to(os.getcwd())}")
    y1 = chart.encode(alt.X("x:Q"), alt.Y("y1:Q"))
    y2 = chart.encode(alt.X("x:Q"), alt.Y("y2:Q"))
    mo.ui.altair_chart(
        y1.mark_line(color="black") + y2.mark_line(color="red")
    )
    return chart, df, y1, y2


if __name__ == "__main__":
    app.run()
