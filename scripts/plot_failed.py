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
    file_browser = mo.ui.file_browser(multiple=False, restrict_navigation=True, filetypes=[".csv"])
    file_browser
    return file_browser,


@app.cell(hide_code=True)
def __(Path, file_browser, mo):
    if file_browser.value:
        filepath = file_browser.value[0].path
    else:
        filepath = mo.cli_args().get("file")
    filepath = Path(filepath)
    return filepath,


@app.cell(hide_code=True)
def __(mo):
    button_reload = mo.ui.button(label="Reload")
    button_reload
    return button_reload,


@app.cell
def __(button_reload, filepath, mo, pd):
    button_reload

    df = pd.read_csv(filepath, sep="\t")

    column_multiselect = mo.ui.multiselect(sorted(c for c in df.columns if c not in ["x", "i"]))
    column_multiselect
    return column_multiselect, df


@app.cell
def __(alt, button_reload, column_multiselect, df, filepath, mo, os):
    button_reload

    # For X axis choose either "x" or "i"
    x_col_name = next(iter(set(df.columns) & set(["x", "i"])))

    df_long = df.melt(id_vars=[x_col_name], value_vars=column_multiselect.value)
    chart = alt.Chart(df_long, title=f"{filepath.absolute().relative_to(os.getcwd())}")
    mo.ui.altair_chart(
        chart.mark_line().encode(
            alt.X(f"{x_col_name}:Q"),
            alt.Y("value:Q"),
            alt.Color("variable:N"),
        )
    )
    return chart, df_long, x_col_name


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
