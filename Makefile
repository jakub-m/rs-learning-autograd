test: clean_csv
	cargo test
clean_csv:
	rm -fv *.csv || true
