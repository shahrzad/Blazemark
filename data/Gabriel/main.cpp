#include <iostream>
#include <blaze/Math.h>
#include <blaze/math/typetraits/getMflop.h>

int main() {
    
    blaze::DynamicVector<int, blaze::columnVector> a{1, 1, 1},b{2, 2, 2}, d{3, 3, 3};
    blaze::DynamicVector<int, blaze::rowVector> at{1, 1, 1};

    blaze::DynamicMatrix<int, blaze::rowMajor> M{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    blaze::DynamicMatrix<int, blaze::columnMajor> Mt(M);

    /// Element Wise Operations 

    // Vectors
    
    //std::cout << blaze::getTotalMflop(5 * a) << std::endl;                // 3        
    //std::cout << blaze::getTotalMflop(5 * (a + b)) << std::endl;          // 6
    //std::cout << blaze::getTotalMflop(5 * at) << std::endl;               // 3
    //std::cout << blaze::getTotalMflop(a + b + d) << std::endl;            // 6
    //std::cout << blaze::getTotalMflop(at + at + at) << std::endl;         // 6
    //std::cout << blaze::getTotalMflop(5 * a + 6 * b) << std::endl;        // 9
    //std::cout << blaze::getTotalMflop(5 * at + 6 * (at + at)) << std::endl; // 12

    //Matrices

    //std::cout << blaze::getTotalMflop(5 * M) << std::endl;                  // 9
    //std::cout << blaze::getTotalMflop(5 * Mt) << std::endl;                 // 9
    //std::cout << blaze::getTotalMflop(5 * (M + M)) << std::endl;            // 18
    //std::cout << blaze::getTotalMflop(M + M) << std::endl;                  // 9
    //std::cout << blaze::getTotalMflop(Mt + Mt + Mt) << std::endl;           // 18
    //std::cout << blaze::getTotalMflop(5 * M + 6 * M) << std::endl;          // 27


    // Non Element-Wise Operations

    //Vectors

    //std::cout << blaze::getTotalMflop(M * a) << std::endl;             // 15 ok
    //std::cout << blaze::getTotalMflop(Mt * a) << std::endl;            // 15 ok
    //std::cout << blaze::getTotalMflop(a + M * a) << std::endl;         // 18 ok
    //std::cout << blaze::getTotalMflop(M * a + M * a) << std::endl;     // 33 ok
    //std::cout << blaze::getTotalMflop(M * M * a) << std::endl;         // 30 ok
    //std::cout << blaze::getTotalMflop(M * (a + b)) << std::endl;       // 18 ok
    //std::cout << blaze::getTotalMflop((M + M) * a) << std::endl;       // 24 ok
    //std::cout << blaze::getTotalMflop((M + M) * (a + b)) << std::endl; // 27 ok

    // Matrices
    std::cout << blaze::getTotalMflop(M * M * M) << std::endl;         //  90 ok
    std::cout << blaze::getTotalMflop(Mt * Mt * Mt) << std::endl;      //  90 ok
    std::cout << blaze::getTotalMflop(M * (M + M)) << std::endl;       //  54 ok
    std::cout << blaze::getTotalMflop((M + M) * M) << std::endl;       //  54 ok
    std::cout << blaze::getTotalMflop((M + M) * (M + M)) << std::endl; //  63 
    std::cout << blaze::getTotalMflop(M * M + M * M) << std::endl;     //  9 not ok should be 99 !!!
    std::cout << blaze::getTotalMflop(5 * M + M * M) << std::endl;     //  63 

    
    return 0;
}
