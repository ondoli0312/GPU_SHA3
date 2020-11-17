#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "keccak.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


static void HandleError(cudaError_t err,
	const char* file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

/*
__device__ unsigned char offset[25] = {
	0, 36, 3, 41, 18,
	1, 44, 10, 45, 2,
	62, 6, 43, 15, 61,
	28, 55, 25, 21, 56,
	27, 20, 39, 8, 14
};
*/

//__global__ void test(ulong* state, ulong* BC, unsigned char* offset, ulong* out)
__global__ void test( ulong* out)
{
	ulong state[25] = {0x0,};
	ulong BC[25] = { 0x0, };
	unsigned char offset[25] = { 0x0, };


	__shared__ ulong temp[25];


	int tid = threadIdx.x;
	ulong temp1 = BC[(tid + 4) % 5] ^ (ROTL64((BC[tid + 1] % 5), 1));

	temp[tid] = ROTL64(state[tid] ^ temp1, offset[tid * 5]);
	temp[tid + 5] = ROTL64(state[tid + 5] ^ temp1, offset[tid * 5 + 1]);
	temp[tid + 10] = ROTL64(state[tid + 10] ^ temp1, offset[tid * 5 + 2]);
	temp[tid + 15] = ROTL64(state[tid + 15] ^ temp1, offset[tid * 5 + 3]);
	temp[tid + 20] = ROTL64(state[tid + 20] ^ temp1, offset[tid * 5 + 4]);
	__syncthreads();
	for (int i = 0; i < 25; i++)
		out[i] = temp[i];
}


int main()
{

	ulong* output = (ulong*)malloc(sizeof(ulong) * 25);
	ulong* dev_output;
	ulong A[25];
	ulong BC[5];
	for (int i = 0; i < 25; i++)
		A[i] = i;
	
	for (int i = 0; i < 5; i++)
		BC[i] = INT_MAX + i;

	unsigned char offset[25] = {
	0, 36, 3, 41, 18,
	1, 44, 10, 45, 2,
	62, 6, 43, 15, 61,
	28, 55, 25, 21, 56,
	27, 20, 39, 8, 14
	};

	HANDLE_ERROR(cudaMalloc((void**)&dev_output, sizeof(ulong)*25));
	//test << <1, 5 >> > (A, BC, offset,dev_output);
	test << <1, 5 >> > (dev_output);
	HANDLE_ERROR(cudaMemcpy(output, dev_output, sizeof(ulong)*25, cudaMemcpyDeviceToHost));


	//HANDLE_ERROR(cudaMemcpy(output, dev_output, 25, cudaMemcpyDeviceToHost));
	
	for (int i = 0; i < 25; i++)
		printf("%llx ", output[i]);
	
	/*
	
	int c[10];
	int b[10];
	int a[10];
	int* dev_c = NULL;
	int* dev_b = NULL;
	int* dev_a = NULL;
	
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(int) * 10));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, sizeof(int) * 10));
	HANDLE_ERROR(cudaMalloc((void**)&dev_a, sizeof(int) * 10));

	for (int i = 0; i < 10; i++) {
		b[i] = i + 1;
		a[i] = i + 2;
	}

	HANDLE_ERROR(cudaMemcpy(dev_a, a, sizeof(int) * 10, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, sizeof(int) * 10, cudaMemcpyHostToDevice));
	add<<<10,10>>>(dev_a, dev_b, dev_c);

	HANDLE_ERROR(cudaMemcpy(c, dev_c, sizeof(int) * 10, cudaMemcpyDeviceToHost));
	for (int i = 0; i < 10; i++)
		printf("%d ", c[i]);
	*/
}

#if 0 
ulong RC[Round] = {
   0x0000000100000000, 0x0000808200000000, 0x0000808a80000000, 0x8000800080000000,
   0x0000808b00000000, 0x8000000100000000, 0x8000808180000000, 0x0000800980000000,
   0x0000008a00000000, 0x0000008800000000, 0x8000800900000000, 0x8000000a00000000,
   0x8000808b00000000, 0x0000008b80000000, 0x0000808980000000, 0x0000800380000000,	
   0x0000800280000000, 0x0000008080000000, 0x0000800a00000000, 0x8000000a80000000,
   0x8000808180000000, 0x0000808080000000, 0x8000000100000000, 0x8000800880000000
};

ulong State[25] = {0, };
ulong BC[5] = { 0, };
ulong buffer[25] = { 0, };

const byte offset[Round + 1] = {
	0, 36, 3, 41, 18,
	1, 44, 10, 45, 2,
	62, 6, 43, 15, 61,
	28, 55, 25, 21, 56,
	27, 20, 39, 8, 14
};

typedef struct SHA {
	byte buffer[136];
	ulong msglen;
}SHA3_INFO;

struct Theta_PARAMS {
	ulong state[5];
	int flag;
	int theta;
};


void Pho_Theta_function_No_use_Thread(ulong theta, ulong flag, ulong state0, ulong state1, ulong state2, ulong state3, ulong state4)
{
	ulong temp1 = RC[(theta + 4) % 5] ^ (ROTL64(BC[(theta + 1) % 5], 1));
	buffer[theta] = ROTL64(state0 ^ temp1, offset[flag]);
	buffer[theta + 5] = ROTL64(state1 ^ temp1, offset[flag + 1]);
	buffer[theta + 10] = ROTL64(state2 ^ temp1, offset[flag + 2]);
	buffer[theta + 15] = ROTL64(state3 ^ temp1, offset[flag + 3]);
	buffer[theta + 20] = ROTL64(state4 ^ temp1, offset[flag + 4]);

}


void Chi_lota_function_No_use_Thread(ulong flag, ulong state0, ulong state1, ulong state2, ulong state3, ulong state4) {

	State[flag] = ((state0) ^ ((~(state1)) & state2));
	State[flag + 1] = ((state1) ^ ((~(state2)) & state3));
	State[flag + 2] = ((state2) ^ ((~(state3)) & state4));
	State[flag + 3] = ((state3) ^ ((~(state4)) & state0));
	State[flag + 4] = ((state4) ^ ((~(state0)) & state1));
}

void keecak_Function_NO_use_Thread(SHA3_INFO* info)
{
	int j = 0;
	for (int i = 0; i < 17; i++) {

		State[i] = State[i] ^ ENDIAN_CHANGE(*(ulong*)(info->buffer + (i << 3)));
	}

	for (j = 0; j < Round; j++)
	{

		BC[0] = State[0] ^ State[5] ^ State[10] ^ State[15] ^ State[20];
		BC[1] = State[1] ^ State[6] ^ State[11] ^ State[16] ^ State[21];
		BC[2] = State[2] ^ State[7] ^ State[12] ^ State[17] ^ State[22];
		BC[3] = State[3] ^ State[8] ^ State[13] ^ State[18] ^ State[23];
		BC[4] = State[4] ^ State[9] ^ State[14] ^ State[19] ^ State[24];

		Pho_Theta_function_No_use_Thread(0, 0, State[0], State[5], State[10], State[15], State[20]);
		Pho_Theta_function_No_use_Thread(1, 5, State[1], State[6], State[11], State[16], State[21]);
		Pho_Theta_function_No_use_Thread(2, 10, State[2], State[7], State[12], State[17], State[22]);
		Pho_Theta_function_No_use_Thread(3, 15, State[3], State[8], State[13], State[18], State[23]);
		Pho_Theta_function_No_use_Thread(4, 20, State[4], State[9], State[14], State[19], State[24]);

		Chi_lota_function_No_use_Thread(0, buffer[0], buffer[6], buffer[12], buffer[18], buffer[24]);
		Chi_lota_function_No_use_Thread(5, buffer[3], buffer[9], buffer[10], buffer[16], buffer[22]);
		Chi_lota_function_No_use_Thread(10, buffer[1], buffer[7], buffer[13], buffer[19], buffer[20]);
		Chi_lota_function_No_use_Thread(15, buffer[4], buffer[5], buffer[11], buffer[17], buffer[23]);
		Chi_lota_function_No_use_Thread(20, buffer[2], buffer[8], buffer[14], buffer[15], buffer[21]);
		State[0] ^= RC[j];
	}
}

void Keccak_Init(SHA3_INFO* info) {
	info->msglen = 0;
	memset(info->buffer, 0, 136);
	memset((byte*)State, 0, 200);
}

__device__ byte pi[25] =
{
	0, 6, 12, 18, 24,
	3, 9, 10, 16, 22,
	1, 7, 13, 19, 20,
	4, 5, 11, 17, 23,
	2, 8, 14, 15, 21
};

__device__ byte offset[Round + 1] = {
0, 36, 3, 41, 18,
1, 44, 10, 45, 2,
62, 6, 43, 15, 61,
28, 55, 25, 21, 56,
27, 20, 39, 8, 14
};

__global__ void GPU_Keccak(ulong* state, ulong* BC)
{
	__shared__ ulong temp[25];


	int tid = threadIdx.x;
	ulong temp1 = BC[(tid + 4) % 5] ^ (ROTL64((BC[tid + 1] % 5), 1));

	temp[tid]		= ROTL64(state[tid]			^ temp1, offset[tid * 5]);
	temp[tid + 5]	= ROTL64(state[tid + 5]		^ temp1, offset[tid * 5 + 1]);
	temp[tid + 10]	= ROTL64(state[tid + 10]	^ temp1, offset[tid * 5 + 2]);
	temp[tid + 15]	= ROTL64(state[tid + 15]	^ temp1, offset[tid * 5 + 3]);
	temp[tid + 20]	= ROTL64(state[tid + 20]	^ temp1, offset[tid * 5 + 4]);

	//__syncthreads();

	state[5 * tid] = ((temp[pi[tid]]) ^ ((~temp[pi[tid + 1]] & temp[pi[tid + 2]])));
	state[5 * tid + 1] = ((temp[pi[tid + 1]]) ^ ((~temp[pi[tid + 2]] & temp[pi[tid + 3]])));
	state[5 * tid + 2] = ((temp[pi[tid + 2]]) ^ ((~temp[pi[tid + 3]] & temp[pi[tid + 4]])));
	state[5 * tid + 3] = ((temp[pi[tid + 3]]) ^ ((~temp[pi[tid + 4]] & temp[pi[tid + 0]])));
	state[5 * tid + 4] = ((temp[pi[tid + 4]]) ^ ((~temp[pi[tid]] & temp[pi[tid + 1]])));

	//__syncthreads();
}

void GPU_Absorbing(SHA3_INFO* info, byte* pt, word msglen){
	info->msglen += msglen;
	ulong* gpu_state;
	ulong* gpu_BC;
	int* a = (int*)malloc(sizeof(int));
	int* dev_a;
	HANDLE_ERROR(cudaMalloc((void**)&gpu_state, 25 * sizeof(ulong)));
	HANDLE_ERROR(cudaMalloc((void**)&gpu_BC, 5 * sizeof(ulong)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_a, sizeof(int)));

	while (msglen >= 136)
	{
		memcpy(info->buffer, pt, msglen);
		for (int i = 0; i < 17; i++) {
			State[i] = State[i] ^ ENDIAN_CHANGE(*(ulong*)(info->buffer + (i << 3)));
		}

		for (int i = 0; i < Round; i++) {

			BC[0] = State[0] ^ State[5] ^ State[10] ^ State[15] ^ State[20];
			BC[1] = State[1] ^ State[6] ^ State[11] ^ State[16] ^ State[21];
			BC[2] = State[2] ^ State[7] ^ State[12] ^ State[17] ^ State[22];
			BC[3] = State[3] ^ State[8] ^ State[13] ^ State[18] ^ State[23];
			BC[4] = State[4] ^ State[9] ^ State[14] ^ State[19] ^ State[24];
			
			for (int j = 0; j < 25; j++)
				printf("%llx ", State[j]);
			printf("\n");
			HANDLE_ERROR(cudaMemcpy(gpu_state, State, sizeof(ulong) * 25, cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(gpu_BC, BC, sizeof(ulong) * 5, cudaMemcpyHostToDevice));
			GPU_Keccak <<<1, 5 >>>(gpu_state, BC);

			printf("END\n");
			HANDLE_ERROR(cudaMemcpy(State, gpu_state, sizeof(ulong) * 25, cudaMemcpyDeviceToHost));
	
		}
		msglen -= 136;
		pt = pt + 136;
	}
	HANDLE_ERROR(cudaFree(gpu_state));
	HANDLE_ERROR(cudaFree(gpu_BC));
}

void GGPU_Keccak(byte* pt, word bytelen, byte* output)
{
	SHA3_INFO info;
	Keccak_Init(&info);
	GPU_Absorbing(&info, pt, bytelen);
	for (int i = 0; i < 4; i++)
		*(ulong*)(output + 8 * i) = ENDIAN_CHANGE(State[i]);
}

void KeccakR1088_Absorting_No_use(SHA3_INFO* info, byte* pt, word msglen)
{
	info->msglen += msglen;
	while (msglen >= 136) {
		memcpy(info->buffer, pt, 136);
		keecak_Function_NO_use_Thread(info);
		msglen -= 136;
		pt = pt + 136;
	}

	//PADDING
	memcpy(info->buffer, pt, msglen);
	word temp = info->msglen % 136;
	memset(info->buffer + temp, 0, 136 - temp);
	info->buffer[temp++] = 0x06;
	info->buffer[135] = 0x80;
	keecak_Function_NO_use_Thread(info);

}


void KeccakR1088_Sqeezing(SHA3_INFO* info, byte* output) {

	for (int i = 0; i < 4; i++)
		*(ulong*)(output + 8 * i) = ENDIAN_CHANGE(State[i]);
}



void KeccakR1088_No_use_Thread(byte* pt, word bytelen, byte* output)
{
	SHA3_INFO info;
	Keccak_Init(&info);
	KeccakR1088_Absorting_No_use(&info, pt, bytelen);
	KeccakR1088_Sqeezing(&info, output);
}

#endif