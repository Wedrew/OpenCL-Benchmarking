CC=gcc-8

SRC_BENCHMARK = benchmark.c
OUT_BENCHMARK= benchmark

all: clean benchmark

benchmark:
	$(CC) $(SRC_BENCHMARK) -o $(OUT_BENCHMARK) -framework OpenCL

clean:
	rm -f $(OUT_BENCHMARK)