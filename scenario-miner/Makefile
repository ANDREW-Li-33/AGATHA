PY := .venv/bin/python
MINER := .venv/bin/miner
DEMO_DB := data/demo/miner.duckdb

.PHONY: setup test demo clean-demo

setup:
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) -e '.[dev,gemini,openai,label]'

test:
	$(PY) -m pytest -q

# End-to-end on synthetic data: no download, no API key.
demo: clean-demo
	$(MINER) synth --out data/demo/extracted --segments 30
	$(MINER) ingest --extracted data/demo/extracted --db $(DEMO_DB)
	$(MINER) split --db $(DEMO_DB)
	$(MINER) label --db $(DEMO_DB) --backend heuristic --run-id demo_heuristic
	$(MINER) render --db $(DEMO_DB) --random 6 --out-dir data/demo/overlays
	$(MINER) eval --db $(DEMO_DB) --out reports/demo_eval.md --n-boot 500
	$(MINER) query sql/mining_queries.sql --db $(DEMO_DB) --max-rows 5

clean-demo:
	rm -rf data/demo reports/demo_eval.md
