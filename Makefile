# Utilise le Python du venv s'il est actif, sinon tente python3, sinon python
PY := $(if $(VIRTUAL_ENV),$(VIRTUAL_ENV)/bin/python,$(shell command -v python3 || command -v python))

.PHONY: setup train

setup:
	$(PY) -m pip install -r requirements.txt

train:
	$(PY) -m src.models.train

