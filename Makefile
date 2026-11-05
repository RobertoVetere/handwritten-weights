# Reproduce every table of the paper. All targets run on CPU.
PY ?= python3
N ?= 2000   # problems per GPT-3 arithmetic subset (paper: 2000)

.PHONY: setup graft graft-exhaustive graft-side-effects facts arith adder all

setup:                 ## download the model + build the story corpus and statistics (~10 min)
	$(PY) scripts/download_model.py
	cd src && $(PY) corpus.py && $(PY) stats.py && $(PY) const_dir.py

graft:                 ## Table 1: GPT-3 arithmetic, base vs grafted (N per subset; ~35 min at N=2000)
	cd src && $(PY) eval.py $(N)

graft-exhaustive:      ## all a,b in 0..99, add and sub, through the full model (~15 min)
	cd src && $(PY) exhaustive_graft.py

graft-side-effects:    ## circuit logits are exactly 0 on stories; identical generations
	cd src && $(PY) preserve.py

facts:                 ## Table 2: facts written into the model's own weights, variants a, b, c (~25 min)
	cd src && $(PY) facts_variants.py a && $(PY) facts_variants.py b && $(PY) facts_variants.py c

arith:                 ## Table 3: single-digit addition written into the model's own weights (~10 min)
	cd src && $(PY) arith_eval.py

adder:                 ## warm-up: a 2-layer adder written from scratch, exhaustive check up to 3 digits
	cd adder_from_scratch && $(PY) handwrite.py && $(PY) edit_sub.py && $(PY) exhaustive.py

all: setup graft graft-exhaustive graft-side-effects facts arith
