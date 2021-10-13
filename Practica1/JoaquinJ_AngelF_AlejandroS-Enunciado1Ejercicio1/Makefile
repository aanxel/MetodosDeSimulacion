all: program

program:
	gcc bat1.c -o bat1 -ltestu01 -lprobdist -lmylib -lm

run:
ifndef SEED
	./bat1
else
	./bat1 $(SEED) > results/seed-$(SEED).txt
endif

.EXPORT_ALL_VARIABLES:
LD_LIBRARY_PATH = /usr/local/lib
LIBRARY_PATH = /usr/local/lib
C_INCLUDE_PATH = /usr/local/include