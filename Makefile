# Team NIDHI, IEEE BigData Cup 2026. Run from the repository root.
ARMS := mnli dmnli cmine prot d4 qnli mnli7 lnli ltail cbnsynapi friendfac
PY   ?= python

.PHONY: help test verify-bundle reproduce clean

help:
	@echo "make test           run the test suite (CPU, no data needed)"
	@echo "make verify-bundle  check the gated, text-bearing files against artifacts/MANIFEST.md"
	@echo "make reproduce      rebuild the submitted predictions and check them byte for byte"

test:
	$(PY) -m pytest -q tests

verify-bundle:
	$(PY) tools/verify_bundle.py

# Two steps, in this order: the factor column, then all three columns with a
# full comparison against the submitted file. Needs the gated bundle.
reproduce: verify-bundle
	$(PY) src/factor_threshold_stabilise.py --arms $(ARMS) --use tfit \
	    --match-oof-scale --expect-change 0.30 \
	    --champion submissions/NIDHI_scaled.csv \
	    --build-test artifacts/factors_scaled_test.csv
	$(PY) src/build_submission.py --verify

clean:
	rm -f artifacts/factors_scaled_test.csv
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
