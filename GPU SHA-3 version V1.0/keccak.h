#pragma once
#define Round 24
#define Block_byte_size 136
#define ENDIAN_CHANGE(val) (((val <<32) & 0xffffffff00000000)|(val >>32))
#define ROTL64(Data, offset) ((Data << offset)|(Data >> (64-offset)))


typedef unsigned int word;
typedef unsigned char byte;
typedef unsigned long long ulong;