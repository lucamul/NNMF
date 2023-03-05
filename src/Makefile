CBLIB = CBLAS/lib/cblas_LINUX.a
BLLIB = BLAS-3.10.0/blas_LINUX.a
CC = gcc
CFLAGS = -O3 -ffast-math -march=native
FC = gfortran
LOADER = $(FC)
FFLAGS = -O3

all: nnmf_blas nnmf_basic nnmf_inlined test_suite_init_w verification

benchmark: nnmf_blas nnmf_basic nnmf_loop_reordering nnmf_ilp nnmf_blocking_caches nnmf_blocking_registers nnmf_blocking_ilp nnmf_blocking_vectorize_128 nnmf_blocking_vectorize_256

nnmf_blas:
	mkdir -p build
	$(CC) -c $(CFLAGS) -ICBLAS/src -ICBLAS/include nnmf/blas.c -o build/blas.o
	$(CC) -c $(CFLAGS) init_w.c -lm -o build/init_w.o
	$(LOADER) $(FFLAGS) -o build/blas.bin build/blas.o build/init_w.o $(CBLIB) $(BLLIB)

nnmf_basic:
	mkdir -p build
	$(CC) main.c $(CFLAGS) -o build/basic.bin init_w.h init_w.c utils.h utils.c nnmf_common.h nnmf/basic.c nnmf.h nnmf.c -lm

nnmf_inlined:
	mkdir -p build
	$(CC) main.c $(CFLAGS) -o build/inlined.bin init_w.h init_w.c utils.h utils.c nnmf_common.h nnmf/inline.c nnmf.h nnmf.c -lm

nnmf_without_transposes:
	mkdir -p build
	$(CC) main.c $(CFLAGS) -o build/without_transposes.bin init_w.h init_w.c utils.h utils.c nnmf_common.h nnmf/without_transposes.c nnmf.h nnmf.c -lm

nnmf_loop_reordering:
	mkdir -p build
	$(CC) main.c $(CFLAGS) -o build/loop_reordering.bin init_w.h init_w.c utils.h utils.c nnmf_common.h nnmf/opt_1.c nnmf.h nnmf.c -lm

nnmf_ilp:
	mkdir -p build
	$(CC) main.c $(CFLAGS) -o build/ilp.bin init_w.h init_w.c utils.h utils.c nnmf_common.h nnmf/opt_2.c nnmf.h nnmf.c -lm

nnmf_blocking_caches:
	mkdir -p build
	$(CC) main.c $(CFLAGS) -o build/blocking_caches.bin init_w.h init_w.c utils.h utils.c nnmf_common.h nnmf/opt_3.c nnmf.h nnmf.c -lm

nnmf_blocking_registers:
	mkdir -p build
	$(CC) main.c $(CFLAGS) -o build/blocking_registers.bin init_w.h init_w.c utils.h utils.c nnmf_common.h nnmf/opt_4.c nnmf.h nnmf.c -lm

nnmf_blocking_ilp:
	mkdir -p build
	$(CC) main.c $(CFLAGS) -o build/blocking_ilp.bin init_w.h init_w.c utils.h utils.c nnmf_common.h nnmf/opt_5.c nnmf.h nnmf.c -lm

nnmf_blocking_vectorize_128:
	mkdir -p build
	$(CC) main.c $(CFLAGS) -o build/blocking_vectorize_128.bin init_w.h init_w.c utils.h utils.c nnmf_common.h nnmf/opt_6.c nnmf.h nnmf.c -lm 

nnmf_blocking_vectorize_256:
	mkdir -p build
	$(CC) main.c $(CFLAGS) -o build/blocking_vectorize_256.bin init_w.h init_w.c utils.h utils.c nnmf_common.h nnmf/opt_7.c nnmf.h nnmf.c -lm

test_suite_init_w:
	mkdir -p test
	$(CC) tests_init_w.c -o test/tests init_w.h init_w.c -lm

verification:
	mkdir -p test
	$(CC) verify.c $(CFLAGS) -o test/verify init_w.h init_w.c utils.h utils.c nnmf_common.h nnmf.h nnmf.c -lm 
clean:
	rm -f build/* test/*
