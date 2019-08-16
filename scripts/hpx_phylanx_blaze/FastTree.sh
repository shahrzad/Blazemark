clang -Wall -O3 -finline-functions -funroll-loops -fopenmp -march=native -lm ../FastTree.c -o FastTree

clang -Wall -g -O3 -finline-functions -funroll-loops -DOPENMP -fopenmp -march=native -lm ../FastTree.c -o FastTree_debug 
