/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#define BOX_SIZE 1024

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "simple_knn.h"
#include <cfloat>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <vector>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#define __CUDACC__
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

struct CustomMin
{
	__device__ __forceinline__
		float3 operator()(const float3& a, const float3& b) const {
		return { min(a.x, b.x), min(a.y, b.y), min(a.z, b.z) };
	}
};

struct CustomMax
{
	__device__ __forceinline__
		float3 operator()(const float3& a, const float3& b) const {
		return { max(a.x, b.x), max(a.y, b.y), max(a.z, b.z) };
	}
};

__host__ __device__ uint32_t prepMorton(uint32_t x)
{
	x = (x | (x << 16)) & 0x030000FF;
	x = (x | (x << 8)) & 0x0300F00F;
	x = (x | (x << 4)) & 0x030C30C3;
	x = (x | (x << 2)) & 0x09249249;
	return x;
}

__host__ __device__ uint32_t coord2Morton(float3 coord, float3 minn, float3 maxx)
{
	uint32_t x = prepMorton(((coord.x - minn.x) / (maxx.x - minn.x)) * ((1 << 10) - 1));
	uint32_t y = prepMorton(((coord.y - minn.y) / (maxx.y - minn.y)) * ((1 << 10) - 1));
	uint32_t z = prepMorton(((coord.z - minn.z) / (maxx.z - minn.z)) * ((1 << 10) - 1));

	return x | (y << 1) | (z << 2);
}

__global__ void coord2Morton(int P, const float3* points, float3 minn, float3 maxx, uint32_t* codes)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	codes[idx] = coord2Morton(points[idx], minn, maxx);
}

struct MinMax
{
	float3 minn;
	float3 maxx;
};

__global__ void boxMinMax(uint32_t P, float3* points, uint32_t* indices, MinMax* boxes)
{
	// 전체 grid에서 현재 trhead의 고유 index를 추출
	auto idx = cg::this_grid().thread_rank();

	MinMax me;
	if (idx < P)
	{
		me.minn = points[indices[idx]];
		me.maxx = points[indices[idx]];
	}
	else
	// 대응되는 point가 없는 thread
	{
		me.minn = { FLT_MAX, FLT_MAX, FLT_MAX };
		me.maxx = { -FLT_MAX,-FLT_MAX,-FLT_MAX };
	}

	// block 내 모든 thread가 공유하는 MinMax 배열 ( grid X )
	__shared__ MinMax redResult[BOX_SIZE];

	// 실질적으로 최대, 최소를 구하는 과정
	for (int off = BOX_SIZE / 2; off >= 1; off /= 2)
	{
		if (threadIdx.x < 2 * off)	// threadIdx.x는 현재 trhead의 block 내부에서의 idx를 의미한다. ( 0 ~ 1023 )
			redResult[threadIdx.x] = me;	// 초기 BOX_SIZE * 2는 1024이므로 반드시 이 조건문에 걸리고, 이는 redResult의 초기값을 설정하는 것을 의미한다.  
		__syncthreads();	// thread들이 동기화되기를 기다린다.

		if (threadIdx.x < off)
		{
			MinMax other = redResult[threadIdx.x + off];
			me.minn.x = min(me.minn.x, other.minn.x);
			me.minn.y = min(me.minn.y, other.minn.y);
			me.minn.z = min(me.minn.z, other.minn.z);
			me.maxx.x = max(me.maxx.x, other.maxx.x);
			me.maxx.y = max(me.maxx.y, other.maxx.y);
			me.maxx.z = max(me.maxx.z, other.maxx.z);
		}
		__syncthreads();	
	}

	// 최종적으로는 trheadIdx.x == 0에 전체 block에 대한 min, max가 저장되게 되어있다.
	if (threadIdx.x == 0)
		boxes[blockIdx.x] = me;
}

__device__ __host__ float distBoxPoint(const MinMax& box, const float3& p)
{
	// point p가 box의 범위 밖에 존재한다면, 그 차이만큼에 대한 거리를 계산하여 반환.
	float3 diff = { 0, 0, 0 };
	if (p.x < box.minn.x || p.x > box.maxx.x)
		diff.x = min(abs(p.x - box.minn.x), abs(p.x - box.maxx.x));
	if (p.y < box.minn.y || p.y > box.maxx.y)
		diff.y = min(abs(p.y - box.minn.y), abs(p.y - box.maxx.y));
	if (p.z < box.minn.z || p.z > box.maxx.z)
		diff.z = min(abs(p.z - box.minn.z), abs(p.z - box.maxx.z));
	return diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
}

// k = 3
template<int K>
__device__ void updateKBest(const float3& ref, const float3& point, float* knn)
{
	// ref point와 src point 사이의 (L2)^2을 distance로 이용
	float3 d = { point.x - ref.x, point.y - ref.y, point.z - ref.z };
	float dist = d.x * d.x + d.y * d.y + d.z * d.z;
	for (int j = 0; j < K; j++)	// Min distance를 유지
	{
		if (knn[j] > dist)
		{
			float t = knn[j];
			knn[j] = dist;
			dist = t;
		}
	}
}

__global__ void boxMeanDist(uint32_t P, float3* points, uint32_t* indices, MinMax* boxes, float* dists)
{

	int idx = cg::this_grid().thread_rank();	// 전체 grid에서의 현재 trhead의 idx
	if (idx >= P)	// point에 대응되지 않는 thread는 무시
		return;

	float3 point = points[indices[idx]];
	float best[3] = { FLT_MAX, FLT_MAX, FLT_MAX };

	// 정렬된 indices를 대상으로, 현재 idx의 앞뒤 3개(총 6개)의 point를 대상
	// min distance를 구한다.
	for (int i = max(0, idx - 3); i <= min(P - 1, idx + 3); i++)
	{
		if (i == idx)
			continue;
		updateKBest<3>(point, points[indices[i]], best);
	}

	float reject = best[2];	// updateKBest<3>를 통해 결정한 min distance 3개를 reject에 저장.
	best[0] = FLT_MAX;
	best[1] = FLT_MAX;
	best[2] = FLT_MAX;

	// BOX_SIZE = 1024
	// boxes : Block 별로 min, max Point를 저장
	// b : 각 Block에 대한 index를 의미한다. ( 즉 이 코드는 BOX 단위로 최근접 distance를 검사하는 코드이다. )
	for (int b = 0; b < (P + BOX_SIZE - 1) / BOX_SIZE; b++)
	{
		MinMax box = boxes[b];
		float dist = distBoxPoint(box, point);
		if (dist > reject || dist > best[2])	// 기존에 구한 min_dist보다 크면 무시
			continue;

		// 인접 BOX와의 distance가 더 작게 연산된다면, min_dist를 새롭게 다시 연산
		// b * BOX_SIZE : 해당 BOX에 포함되어 있는 thread의 시작 인덱스
		for (int i = b * BOX_SIZE; i < min(P, (b + 1) * BOX_SIZE); i++)
		{
			if (i == idx)
				continue;
			updateKBest<3>(point, points[indices[i]], best);
		}
	}
	dists[indices[idx]] = (best[0] + best[1] + best[2]) / 3.0f;	// 최종적으로 구한 인접한 3개의 point에 대하여 평균 거리 저장.
}

/*
P : 초기 point의 갯수
points : 초기 point의 위치
meanDists : 1차원의 0으로 채워진, points 갯수 크기의 배열
*/
void SimpleKNN::knn(int P, float3* points, float* meanDists)
{
	float3* result;
	cudaMalloc(&result, sizeof(float3));	// CUDA의 GPU 메모리 할당 함수 / float3 = 4byte * 3 (12byte) 크기 할당 요청
	size_t temp_storage_bytes;

	float3 init = { 0, 0, 0 }, minn, maxx;

/*
struct CustomMin
{
	__device__ __forceinline__
		float3 operator()(const float3& a, const float3& b) const {
		return { min(a.x, b.x), min(a.y, b.y), min(a.z, b.z) };
	}
};
*/

	cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, points, result, P, CustomMin(), init);	// GPU에서 points를 대상으로 CustomMin()연산을 하여 최소값 추출
	thrust::device_vector<char> temp_storage(temp_storage_bytes);	// CUDA의 임시 GPU 메모리 할당 함수 ( temp_storage_bytes 만큼 )

	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMin(), init);
	cudaMemcpy(&minn, result, sizeof(float3), cudaMemcpyDeviceToHost);	// 결과(result)를 GPU to CPU 방향으로 복사 ( result -> minn )

/*
struct CustomMax
{
	__device__ __forceinline__
		float3 operator()(const float3& a, const float3& b) const {
		return { max(a.x, b.x), max(a.y, b.y), max(a.z, b.z) };
	}
};
*/
	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMax(), init);	// GPU에서 points를 대상으로 CustomMin()연산을 하여 최대값 추출
	cudaMemcpy(&maxx, result, sizeof(float3), cudaMemcpyDeviceToHost);	// 결과(result)를 GPU to CPU 방향으로 복사 ( result -> maxx )

	thrust::device_vector<uint32_t> morton(P);	// 각 포인트의 Morton code(=Z-order curve index)를 저장할 GPU 벡터 (길이 P)
	thrust::device_vector<uint32_t> morton_sorted(P);	// Morton code를 정렬한 결과를 저장할 GPU 벡터 (길이 P)
	/*
		coord2Morton : CUDA 커널 함수로, points들을 min, max를 이용해서 정규화하고, 이를 Morton code로 변환하여, morton 벡터에 저장
		<< <(P + 255) / 256, 256 >> > : p개의 point를 256개씩 나누어서 병렬처리
	*/
	coord2Morton << <(P + 255) / 256, 256 >> > (P, points, minn, maxx, morton.data().get());	

	thrust::device_vector<uint32_t> indices(P);
	thrust::sequence(indices.begin(), indices.end());	// 임시 벡터 indices를 0에서 P-1로 초기화 ( RadixSort에서 사용할 index )
	thrust::device_vector<uint32_t> indices_sorted(P);	

	// CUB 라이브러리의 RadixSort를 이용해서 (morton, indices)를 정렬하여 morton_sorted에 저장
	// 위와 동일하게 CUB에 필요한 임시 저장소의 크기를 먼저 결정하기 위해 2번 호출
	cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);	// GPU에서 points를 대상으로 CustomMin()연산을 하여 최소값 추출
	temp_storage.resize(temp_storage_bytes);

	cub::DeviceRadixSort::SortPairs(temp_storage.data().get(), temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);

	// #define BOX_SIZE 1024
	uint32_t num_boxes = (P + BOX_SIZE - 1) / BOX_SIZE;		// 병렬처리를 위한 block의 갯수 결정
	thrust::device_vector<MinMax> boxes(num_boxes);
	boxMinMax << <num_boxes, BOX_SIZE >> > (P, points, indices_sorted.data().get(), boxes.data().get());	// boxes에 각 block 별로 병렬처리된 point에서의 min, max 값이 저장된다.
	boxMeanDist << <num_boxes, BOX_SIZE >> > (P, points, indices_sorted.data().get(), boxes.data().get(), meanDists);	// 

	cudaFree(result);
}