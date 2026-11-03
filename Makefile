PYTHON ?= python3
VERSION ?=

.PHONY: release
release:
	$(PYTHON) scripts/release.py "$(VERSION)"
