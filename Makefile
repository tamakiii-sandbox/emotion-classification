.PHONY: help install

help:
	@cat $(firstword $(MAKEFILE_LIST))

install:
	poetry install --no-root
