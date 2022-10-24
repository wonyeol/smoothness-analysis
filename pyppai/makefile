#-------------------------------
# Entry points
all: execs
execs: 	supp-main   supp-batch \
	diff-main   diff-batch
# Main target to run all Lipschitz compositional analysis tests
compl:  run-diff-examples-comp-lipsch
# Main target to run all differentiability compositional analysis tests
compd:  run-diff-examples-comp-diff
# Main target to run all reparameterisation tests
reparl: run-diff-repar-lipsch
repard: run-diff-repar-diff
# compare two (old)
compb:  run-diff-examples-comp run-diff-examples-comp-old
#-------------------------------
# Aliases
SUPP-MAIN    =./_build/default/main_support/entry_support.exe
SUPP-BATCH   =./_build/default/batch_support/batch_support.exe
DIFF-MAIN    =./_build/default/main_diff/entry_diff.exe
DIFF-BATCH   =./_build/default/batch_diff/batch_diff.exe
#-------------------------------
# Support analysis
supp-main:  $(SUPP-MAIN)  # Main analyzer (single file)
supp-batch: $(SUPP-BATCH) # Batch analyzer (many files)
$(SUPP-MAIN):
	dune build $(SUPP-MAIN)  && cp $(SUPP-MAIN)  supp-main.exe
$(SUPP-BATCH):
	dune build $(SUPP-BATCH) && cp $(SUPP-BATCH) supp-batch.exe
#-------------------------------
# Continuity/diff-mainrentiability for Pyro model,guide pairs
diff-main:  $(DIFF-MAIN)  # Main analyzer (single file)
diff-batch: $(DIFF-BATCH) # Batch analyzer (many files)
$(DIFF-MAIN):
	dune build $(DIFF-MAIN)  && cp $(DIFF-MAIN)  diff-main.exe
$(DIFF-BATCH):
	dune build $(DIFF-BATCH) && cp $(DIFF-BATCH) diff-batch.exe
#-------------------------------
# Running batch
run-supp-suite:    $(SUPP-BATCH)
	$(SUPP-BATCH) suite -q -to 10
run-supp-examples: $(SUPP-BATCH)
	$(SUPP-BATCH) examples -q -to 10
run-diff-examples: $(DIFF-BATCH)
	$(DIFF-BATCH) -analysis
run-diff-examples-comp: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -comp -no-fwd -analysis > comp.all
run-diff-examples-comp-lipsch: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -comp -no-fwd -analysis -g-lipsch > compl.all
run-diff-examples-comp-diff: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -comp -no-fwd -analysis -g-diff > compd.all
run-diff-examples-comp-old: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -comp -old-comp -no-fwd -analysis > out.comp-old
run-diff-examples-comp-fwd: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -comp -fwd -analysis > out.comp-fwd
run-diff-examples-fwd: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -no-comp -fwd -analysis > out.fwd
run-diff-examples-fwd-lipsch: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -no-comp -fwd -analysis -g-lipsch > outl.fwd
run-diff-examples-fwd-diff: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -no-comp -fwd -analysis -g-diff > outd.fwd
run-diff-examples-none: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -no-comp -no-fwd -analysis > out.none
run-diff-repar-old: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -srepar-o > out.repar-old
run-diff-repar-lipsch: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -srepar-n -g-lipsch > reparl.all
run-diff-repar-diff: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -srepar-n -g-diff > repard.all
#-------------------------------
# Running all specific tests
compd-split:    compd.air	compd.br	compd.csis	compd.cvae   \
		compd.dmm	compd.dpmm	compd.lda	compd.mhmm   \
		compd.prodlda	compd.scanvi	compd.sgdef	compd.spnor  \
		compd.ssvae  	compd.vae
compl-split:    compl.air	compl.br	compl.csis	compl.cvae   \
		compl.dmm	compl.dpmm	compl.lda	compl.mhmm   \
		compl.prodlda	compl.scanvi	compl.sgdef	compl.spnor  \
		compl.ssvae	compl.vae
fwdd-split:     fwdd.air	fwdd.br		fwdd.csis	fwdd.cvae    \
		fwdd.dmm	fwdd.dpmm	fwdd.lda	fwdd.mhmm    \
		fwdd.prodlda	fwdd.scanvi	fwdd.sgdef	fwdd.spnor   \
		fwdd.ssvae	fwdd.vae
repard-split:   repard.air	repard.br	repard.csis	repard.cvae  \
		repard.dmm	repard.dpmm	repard.lda	repard.mhmm  \
		repard.prodlda	repard.scanvi	repard.sgdef	repard.spnor \
		repard.ssvae	repard.vae
reparl-split:   reparl.air	reparl.br	reparl.csis	reparl.cvae  \
		reparl.dmm	reparl.dpmm	reparl.lda	reparl.mhmm  \
		reparl.prodlda	reparl.scanvi	reparl.sgdef	reparl.spnor \
		reparl.ssvae	reparl.vae
#-------------------------------
# Running specific tests
# air
compl.air: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -comp -no-fwd -analysis -g-lipsch -sel 0 > $@
compd.air: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -comp -no-fwd -analysis -g-diff -sel 0 > $@
fwdd.air: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -no-comp -fwd -analysis -g-diff -sel 0 > $@
reparl.air: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -srepar-n -g-lipsch -sel 0 > $@
repard.air: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -srepar-n -g-diff -sel 0 > $@
# br
compl.br: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -comp -no-fwd -analysis -g-lipsch -sel 2 > $@
compd.br: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -comp -no-fwd -analysis -g-diff -sel 2 > $@
fwdd.br: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -no-comp -fwd -analysis -g-diff -sel 2 > $@
reparl.br: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -srepar-n -g-lipsch -sel 2 > $@
repard.br: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -srepar-n -g-diff -sel 2 > $@
# csis
compl.csis: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -comp -no-fwd -analysis -g-lipsch -sel 4 > $@
compd.csis: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -comp -no-fwd -analysis -g-diff -sel 4 > $@
fwdd.csis: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -no-comp -fwd -analysis -g-diff -sel 4 > $@
reparl.csis: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -srepar-n -g-lipsch -sel 4 > $@
repard.csis: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -srepar-n -g-diff -sel 4 > $@
# dmm
compl.dmm: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -comp -no-fwd -analysis -g-lipsch -sel 6 > $@
compd.dmm: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -comp -no-fwd -analysis -g-diff -sel 6 > $@
fwdd.dmm: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -no-comp -fwd -analysis -g-diff -sel 6 > $@
reparl.dmm: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -srepar-n -g-lipsch -sel 6 > $@
repard.dmm: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -srepar-n -g-diff -sel 6 > $@
# lda
compl.lda: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -comp -no-fwd -analysis -g-lipsch -sel 8 > $@
compd.lda: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -comp -no-fwd -analysis -g-diff -sel 8 > $@
fwdd.lda: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -no-comp -fwd -analysis -g-diff -sel 8 > $@
reparl.lda: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -srepar-n -g-lipsch -sel 8 > $@
repard.lda: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -srepar-n -g-diff -sel 8 > $@
# sgdef
compl.sgdef: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -comp -no-fwd -analysis -g-lipsch -sel 10 > $@
compd.sgdef: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -comp -no-fwd -analysis -g-diff -sel 10 > $@
fwdd.sgdef: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -no-comp -fwd -analysis -g-diff -sel 10 > $@
reparl.sgdef: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -srepar-n -g-lipsch -sel 10 > $@
repard.sgdef: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -srepar-n -g-diff -sel 10 > $@
# ssvae
compl.ssvae: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -comp -no-fwd -analysis -g-lipsch -sel 12 > $@
compd.ssvae: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -comp -no-fwd -analysis -g-diff -sel 12 > $@
fwdd.ssvae: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -no-comp -fwd -analysis -g-diff -sel 12 > $@
reparl.ssvae: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -srepar-n -g-lipsch -sel 12 > $@
repard.ssvae: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -srepar-n -g-diff -sel 12 > $@
# vae
compl.vae: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -comp -no-fwd -analysis -g-lipsch -sel 14 > $@
compd.vae: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -comp -no-fwd -analysis -g-diff -sel 14 > $@
fwdd.vae: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -no-comp -fwd -analysis -g-diff -sel 14 > $@
reparl.vae: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -srepar-n -g-lipsch -sel 14 > $@
repard.vae: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -srepar-n -g-diff -sel 14 > $@
# spnor
compl.spnor: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -comp -no-fwd -analysis -g-lipsch -sel 16 > $@
compd.spnor: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -comp -no-fwd -analysis -g-diff -sel 16 > $@
fwdd.spnor: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -no-comp -fwd -analysis -g-diff -sel 16 > $@
reparl.spnor: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -srepar-n -g-lipsch -sel 16 > $@
repard.spnor: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -srepar-n -g-diff -sel 16 > $@
# dpmm
compl.dpmm: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -comp -no-fwd -analysis -g-lipsch -sel 18 > $@
compd.dpmm: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -comp -no-fwd -analysis -g-diff -sel 18 > $@
fwdd.dpmm: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -no-comp -fwd -analysis -g-diff -sel 18 > $@
reparl.dpmm: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -srepar-n -g-lipsch -sel 18 > $@
repard.dpmm: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -srepar-n -g-diff -sel 18 > $@
# cvae
compl.cvae: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -comp -no-fwd -analysis -g-lipsch -sel 20 > $@
compd.cvae: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -comp -no-fwd -analysis -g-diff -sel 20 > $@
fwdd.cvae: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -no-comp -fwd -analysis -g-diff -sel 20 > $@
reparl.cvae: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -srepar-n -g-lipsch -sel 20 > $@
repard.cvae: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -srepar-n -g-diff -sel 20 > $@
# scanvi
compl.scanvi: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -comp -no-fwd -analysis -g-lipsch -sel 22 > $@
compd.scanvi: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -comp -no-fwd -analysis -g-diff -sel 22 > $@
fwdd.scanvi: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -no-comp -fwd -analysis -g-diff -sel 22 > $@
reparl.scanvi: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -srepar-n -g-lipsch -sel 22 > $@
repard.scanvi: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -srepar-n -g-diff -sel 22 > $@
# prodlda
compl.prodlda: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -comp -no-fwd -analysis -g-lipsch -sel 24 > $@
compd.prodlda: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -comp -no-fwd -analysis -g-diff -sel 24 > $@
fwdd.prodlda: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -no-comp -fwd -analysis -g-diff -sel 24 > $@
reparl.prodlda: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -srepar-n -g-lipsch -sel 24 > $@
repard.prodlda: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -srepar-n -g-diff -sel 24 > $@
# mhmm
compl.mhmm: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -comp -no-fwd -analysis -g-lipsch -sel 26 > $@
compd.mhmm: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -comp -no-fwd -analysis -g-diff -sel 26 > $@
fwdd.mhmm: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -no-comp -fwd -analysis -g-diff -sel 26 > $@
reparl.mhmm: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -srepar-n -g-lipsch -sel 26 > $@
repard.mhmm: $(DIFF-BATCH)
	time $(DIFF-BATCH) -v -srepar-n -g-diff -sel 26 > $@
#-------------------------------
# PHONY misc targets
.PHONY: $(SUPP-MAIN)  $(DIFF-MAIN)  \
	$(SUPP-BATCH) $(DIFF-BATCH) \
	all execs \
	supp-main supp-batch diff-main diff-batch \
	run-supp-suite run-supp-examples run-diff-examples \
	wc edit clean
wc:
	wc */*.ml*
edit:
	emacs --background-color=Black --foreground-color=White makefile &
clean: 
	rm -rf */*~ _build *.exe */.merlin
#-------------------------------
