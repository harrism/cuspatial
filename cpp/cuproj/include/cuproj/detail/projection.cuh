/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuproj/detail/pipeline.cuh>
#include <cuproj/operation/transverse_mercator.cuh>

#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/transform.h>

namespace cuproj {

#ifdef __CUDACC__
template <typename Coordinate, typename T>
projection<Coordinate, T>::projection(projection<Coordinate, T> const& other)
{
  params_                = other.params_;
  constructed_direction_ = other.constructed_direction_;
  copy_operations(other.operations_);
}
template <typename Coordinate, typename T>
projection<Coordinate, T>& projection<Coordinate, T>::operator=(
  projection<Coordinate, T> const& other)
{
  params_                = other.params_;
  constructed_direction_ = other.constructed_direction_;
  copy_operations(other.operations_);
  return *this;
}
#endif

template <typename Coordinate, typename T>
template <class InputCoordIter, class OutputCoordIter>
void projection<Coordinate, T>::transform(InputCoordIter first,
                                          InputCoordIter last,
                                          OutputCoordIter result,
                                          direction dir,
                                          rmm::cuda_stream_view stream) const
{
#ifdef __CUDACC__
  dir = (constructed_direction_ == direction::FORWARD) ? dir : reverse(dir);

  if (dir == direction::FORWARD) {
    auto pipe = detail::pipeline<Coordinate, direction::FORWARD>{
      params_, operations_.data().get(), operations_.size()};
    thrust::transform(rmm::exec_policy(stream), first, last, result, pipe);
  } else {
    auto pipe = detail::pipeline<Coordinate, direction::INVERSE>{
      params_, operations_.data().get(), operations_.size()};
    thrust::transform(rmm::exec_policy(stream), first, last, result, pipe);
  }
#endif
}

template <class Coordinate, typename T>
void projection<Coordinate, T>::setup(std::vector<operation_type> const& operations)
{
#ifdef __CUDACC__
  std::for_each(operations.begin(), operations.end(), [&](auto const& op) {
    switch (op) {
      case operation_type::TRANSVERSE_MERCATOR: {
        auto op = transverse_mercator<Coordinate>{params_};
        params_ = op.setup(params_);
        break;
      }
      // TODO: some ops don't have setup.  Should we make them all have setup?
      default: break;
    }
  });

  operations_.resize(operations.size());
  thrust::copy(operations.begin(), operations.end(), operations_.begin());
#endif
}

template <class Coordinate, typename T>
void projection<Coordinate, T>::copy_operations(thrust::device_vector<operation_type> const& ops)
{
#ifdef __CUDACC__
  printf("Copyinging %lu operations\n", ops.size());
  thrust::copy(ops.begin(), ops.end(), operations_.begin());
#endif
}

}  // namespace cuproj
