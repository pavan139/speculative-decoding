PYTHON ?= python
.PHONY: test demo evaluate
test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s checks -v
demo:
	PYTHONPATH=src $(PYTHON) -m speculative_decoding demo --output artifacts/demo.json
evaluate:
	PYTHONPATH=src $(PYTHON) -m speculative_decoding evaluate --output artifacts/evaluation.json
