# Convenience targets. Everything here is a thin wrapper over the CLI.
.PHONY: help install install-quality install-dev scrape index serve inspect ask eval validate-golden test test-all docker-up docker-down clean

PY ?= python

help:
	@echo "install          core deps (lite profile, no torch)"
	@echo "install-quality  + torch for the quality profile"
	@echo "install-dev      + ragas/deepeval/pytest"
	@echo "scrape           fetch documentation into data/raw"
	@echo "index            chunk, embed and upsert (RECREATE=1 to rebuild)"
	@echo "                 EMBEDDING=mxbai to index with another model"
	@echo "serve            run the API and frontend on :8000"
	@echo "inspect Q='...'  compare dense / sparse / hybrid / reranked"
	@echo "eval             score every configuration"
	@echo "test             free tests only (no API key needed)"
	@echo "test-all         + the LLM-judged gate (costs money)"

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install --no-deps -e .

install-quality: install
	$(PY) -m pip install -r requirements-quality.txt

install-dev: install
	$(PY) -m pip install -r requirements-dev.txt

scrape:
	$(PY) -m askthedocs.cli scrape

index:
	$(PY) -m askthedocs.cli index $(if $(RECREATE),--recreate,) 		$(if $(EMBEDDING),--embedding $(EMBEDDING),) 		$(if $(RERANKER),--reranker $(RERANKER),) 		$(if $(ONLY),--only $(ONLY),)

serve:
	$(PY) -m askthedocs.cli serve --reload

inspect:
	@test -n "$(Q)" || (echo "usage: make inspect Q='your question'" && exit 1)
	$(PY) -m askthedocs.cli inspect "$(Q)" $(if $(EMBEDDING),--embedding $(EMBEDDING),)

ask:
	@test -n "$(Q)" || (echo "usage: make ask Q='your question'" && exit 1)
	$(PY) -m askthedocs.cli ask "$(Q)" $(if $(NAIVE),--naive,)

eval:
	$(PY) -m askthedocs.evals.run

validate-golden:
	$(PY) -m askthedocs.evals.build_golden --validate

test:
	$(PY) -m pytest -m "not eval" -q

test-all:
	$(PY) -m pytest -q

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	rm -rf data/qdrant evals_out .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
