test: clean_csv
    cargo test

build:
    cargo build

clean_csv:
    rm -fv *.csv || true

clean: clean_csv
    rm -rf target || true

marimo:
    marimo edit scripts/plot_failed.py

