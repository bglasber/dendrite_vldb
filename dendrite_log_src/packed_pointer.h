#ifndef __PP_H__
#define __PP_H__
#include <stdint.h>
#include <stdio.h>

typedef uint64_t packed_pointer;
static const uint64_t k_flag_bits_max = -1ull >> 16;
static const uint32_t k_shift = 48;

packed_pointer set_packed_pointer_int( uint64_t d, uint16_t flag_bits );
packed_pointer set_packed_pointer_ptr( void* ptr, uint16_t flag_bits );
packed_pointer set_flag_bits( const packed_pointer pp, uint16_t flag_bits );
packed_pointer set_as_int( const packed_pointer pp, uint64_t data );
packed_pointer set_as_pointer( const packed_pointer pp, void* ptr );
void* get_pointer( const packed_pointer pp );
uint64_t get_int( const packed_pointer pp );
uint16_t get_flag_bits( const packed_pointer pp );

inline packed_pointer set_packed_pointer_int( uint64_t d, uint16_t flag_bits ) {
    packed_pointer pp = 0;

    // make sure it fits

    uint64_t flag_bits_shift =
        ( uint64_t )( ( (uint64_t) flag_bits ) << k_shift );
    pp = d | flag_bits_shift;
    return pp;
}

inline packed_pointer set_packed_pointer_ptr( void* ptr, uint16_t flag_bits ) {
    uint64_t d = (uint64_t) ptr;
    return set_packed_pointer_int( d, flag_bits );
}

inline packed_pointer set_flag_bits( const packed_pointer pp,
                                     uint16_t             flag_bits ) {
    return set_packed_pointer_int( get_int( pp ), flag_bits );
}
inline packed_pointer set_as_int( const packed_pointer pp, uint64_t data ) {
    return set_packed_pointer_int( data, get_flag_bits( pp ) );
}

inline packed_pointer set_as_pointer( const packed_pointer pp, void* ptr ) {
    return set_packed_pointer_ptr( ptr, get_flag_bits( pp ) );
}

inline void* get_pointer( const packed_pointer pp ) {
    void* ptr = (void*) get_int( pp );
    return ptr;
}

inline uint64_t get_int( const packed_pointer pp ) {
    uint64_t i = pp & k_flag_bits_max;
    return i;
}
inline uint16_t get_flag_bits( const packed_pointer pp ) {
    uint16_t e = pp >> k_shift;
    return e;
}

#endif
