#include <type_traits>
#include <iostream>

// https://en.cppreference.com/w/cpp/header/type_traits
// - SFINAE (Substitution Failure Is Not An Error)

template< typename T, typename = void >
struct has_prop
{
   static constexpr bool Value = false;
};

template< typename T >
struct has_prop <T, std::void_t<decltype(T::prop)> >
{
   static constexpr bool Value = true;
};

template<typename T>
constexpr bool has_prop_v = has_prop<T>::Value;

////

struct A
{
   //static constexpr bool prop = true;
};

int main( int, char const *[] )
{
   if constexpr( has_prop_v<A> )
   {
      std::cout << "hello";
   }
}
