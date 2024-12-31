test: clean_csv
	cargo test
clean_csv:
	rm -fv *.csv || true
clean: clean_csv
	rm -rf target || true
marimo:
	marimo edit scripts/plot_failed.py
