#ifndef GETMFLOP
#define GETMFLOP

#include <blaze/Math.h>
#include <blaze/util/typetraits/IsReference.h>
#include <blaze/util/EnableIf.h>
#include <blaze/math/typetraits/IsExpression.h>
#include <typeinfo>

namespace blaze {

////////////////////////////

// LEAFS OF TEMPLATE TREE

////////////////////////////

// Dynamic Vector Mflop computation

template< typename Type   // Data type of the vector
        , bool TF>        // Transpose flag
std::size_t getMflop(const blaze::DynamicVector<Type, TF>& vec) {
    return 0;
}


// Dynamic Matrix Mflop computation

template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
std::size_t getMflop(const blaze::DynamicMatrix<Type, SO>& mat) {
    return 0;
}


////////////////////////////

// MFLOP FOR MANY EXPRESSIONS

////////////////////////////


/////// Element wise operations /////////

// Overloading for AddExpr  (vectors and matrices)
template< typename T >  //  base type of the expression
std::size_t getMflop(const blaze::Matrix<T>& array) {
    return getMflop(~array);
}

template< typename T >  //  base type of the expression
std::size_t getMflop(const blaze::AddExpr<T>& array) {
    return getMflop((~array).leftOperand()) + getMflop((~array).rightOperand()) + 1;
}


// Overloading for DVecScalarMult

template< typename VT  // Type of the left-hand side dense vector
        , typename ST  // Type of the right-hand side scalar value
        , bool TF >    // Transpose flag
std::size_t getMflop(const blaze::DVecScalarMultExpr<VT, ST, TF >& vec) {
    return getMflop((~vec).leftOperand()) + 1;
}


// Overleading for DMatScalarMult
template< typename MT   // Type of the left-hand side dense Matrix
        , typename ST   // Type of the right-hand side scalar value
        , bool SO >     // Storage order
std::size_t getMflop(const blaze::DMatScalarMultExpr<MT, ST, SO >& mat) {
    return getMflop((~mat).leftOperand()) + 1;
}





//////// Non Element-Wise Operations ///////////

// Overloading for MatVecMult
template< typename VT>  // Type of the right-hand side dense vector
auto getMflop(const blaze::MatVecMultExpr<VT>& vec) -> blaze::EnableIf_t<
															blaze::IsReference_v<typename VT::VectorType::LT> &&
															blaze::IsReference_v<typename VT::VectorType::RT>, std::size_t>
{
    return (~vec).size() * (2 + getMflop((~vec).leftOperand()) + getMflop((~vec).rightOperand())) - 1;
}

template< typename VT>  // Type of the right-hand side dense vector
auto getMflop(const blaze::MatVecMultExpr<VT>& vec) -> blaze::EnableIf_t<
															!blaze::IsReference_v<typename VT::VectorType::LT> &&
															blaze::IsReference_v<typename VT::VectorType::RT>, std::size_t>
{
    return getMflop((~vec).leftOperand()) + (~vec).size() * (2 + getMflop((~vec).rightOperand())) - 1 ;
}

template< typename VT>  // Type of the right-hand side dense vector
auto getMflop(const blaze::MatVecMultExpr<VT>& vec) -> blaze::EnableIf_t<
															blaze::IsReference_v<typename VT::VectorType::LT> &&
															!blaze::IsReference_v<typename VT::VectorType::RT>, std::size_t>
{
    return (~vec).size() * (2 + getMflop((~vec).leftOperand())) - 1 + getMflop((~vec).rightOperand());
}

template< typename VT> // Type of the right-hand side dense vector
auto getMflop(const blaze::MatVecMultExpr<VT>& vec) -> blaze::EnableIf_t<
															!blaze::IsReference_v<typename VT::VectorType::LT> &&
															!blaze::IsReference_v<typename VT::VectorType::RT>, std::size_t>
{
    return 2 * (~vec).size() + getMflop((~vec).leftOperand()) + getMflop((~vec).rightOperand()) - 1;
}




// Overloading for MatMatMult
template< typename MT>
auto getMflop(const blaze::MatMatMultExpr<MT>& mat) -> blaze::EnableIf_t<
															blaze::IsReference_v<typename MT::MatrixType::LT> &&
															blaze::IsReference_v<typename MT::MatrixType::RT>, std::size_t>{
    return (~mat).leftOperand().columns() * (2 + getMflop((~mat).leftOperand()) + getMflop((~mat).rightOperand())) - 1;
}

template< typename MT>
auto getMflop(const blaze::MatMatMultExpr<MT>& mat) -> blaze::EnableIf_t<
															!blaze::IsReference_v<typename MT::MatrixType::LT> &&
															blaze::IsReference_v<typename MT::MatrixType::RT>, std::size_t>{
    return getMflop((~mat).leftOperand()) + (~mat).leftOperand().columns() * (2 + getMflop((~mat).rightOperand())) - 1;
}

template< typename MT>
auto getMflop(const blaze::MatMatMultExpr<MT>& mat) -> blaze::EnableIf_t<
															blaze::IsReference_v<typename MT::MatrixType::LT> &&
															!blaze::IsReference_v<typename MT::MatrixType::RT>, std::size_t>{
    return (~mat).leftOperand().columns() * (2 + getMflop((~mat).leftOperand())) - 1 + getMflop((~mat).rightOperand());
}

template< typename MT>
auto getMflop(const blaze::MatMatMultExpr<MT>& mat) -> blaze::EnableIf_t<
															!blaze::IsReference_v<typename MT::MatrixType::LT> &&
															!blaze::IsReference_v<typename MT::MatrixType::RT>, std::size_t>{
    return 2 * (~mat).leftOperand().columns() + getMflop((~mat).leftOperand()) + getMflop((~mat).rightOperand()) - 1;
}





////////////////////////////

// TOTAL NUMBER OF MFLOP

////////////////////////////


// Default DVector Total Mflop computation

template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
std::size_t getTotalMflop(const blaze::DenseVector<VT, TF>& vec) {
    auto copyVec(~vec);
    return (~vec).size() * getMflop(~vec);
}

// Default Matrix Total Mflop computation

template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order
std::size_t getTotalMflop(const blaze::DenseMatrix<MT, SO>& mat) {
    return (~mat).rows() * (~mat).columns() * getMflop(~mat);
}
}
#endif