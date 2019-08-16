CC=gcc

SRC_BENCHMARK = benchmark.c
OUT_BENCHMARK= benchmark

all: clean benchmark

benchmark:
	$(CC) $(SRC_BENCHMARK) -o $(OUT_BENCHMARK)

clean:
	rm -f $(OUT_BENCHMARK)
