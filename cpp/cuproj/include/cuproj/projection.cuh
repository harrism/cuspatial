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

#include <cuproj/ellipsoid.hpp>
#include <cuproj/operation/axis_swap.cuh>
#include <cuproj/operation/clamp_angular_coordinates.cuh>
#include <cuproj/operation/degrees_to_radians.cuh>
#include <cuproj/operation/offset_scale_cartesian_coordinates.cuh>
#include <cuproj/operation/operation.cuh>
#include <cuproj/operation/transverse_mercator.cuh>
#include <cuproj/projection_parameters.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/device_vector.h>

#include <cuda/std/tuple>

#include <type_traits>

namespace cuproj {

namespace detail {

/**
 * @internal
 * @brief A pipeline of projection operations applied in order to a coordinate
 *
 * @tparam Coordinate the coordinate type
 * @tparam dir The direction of the pipeline, FORWARD or INVERSE
 * @tparam T the coordinate value type
 */
template <typename Coordinate,
          direction dir = direction::FORWARD,
          typename T    = typename Coordinate::value_type>
struct pipeline {
  using iterator_type = std::conditional_t<dir == direction::FORWARD,
                                           operation_type const*,
                                           std::reverse_iterator<operation_type const*>>;

  pipeline(projection_parameters<T> const& params,
           operation_type const* ops,
           std::size_t num_stages)
    : params_(params), d_ops(ops), num_stages(num_stages)
  {
    if constexpr (dir == direction::FORWARD) {
      first_ = d_ops;
    } else {
      first_ = std::reverse_iterator(d_ops + num_stages);
    }
  }

  __device__ Coordinate operator()(Coordinate const& c) const
  {
    // TODO: improve this dispatch, and consider whether we can use virtual functions
    Coordinate c_out{c};
    thrust::for_each_n(thrust::seq, first_, num_stages, [&](auto const& op) {
      switch (op) {
        case operation_type::AXIS_SWAP: {
          auto op = axis_swap<Coordinate>{};
          c_out   = op(c_out, dir);
          break;
        }
        case operation_type::DEGREES_TO_RADIANS: {
          auto op = degrees_to_radians<Coordinate>{};
          c_out   = op(c_out, dir);
          break;
        }
        // case operation_type::RADIANS_TO_DEGREES:
        case operation_type::CLAMP_ANGULAR_COORDINATES: {
          auto op = clamp_angular_coordinates<Coordinate>{params_};
          c_out   = op(c_out, dir);
          break;
        }
        case operation_type::OFFSET_SCALE_CARTESIAN_COORDINATES: {
          auto op = offset_scale_cartesian_coordinates<Coordinate>{params_};
          c_out   = op(c_out, dir);
          break;
        }
        case operation_type::TRANSVERSE_MERCATOR: {
          auto op = transverse_mercator<Coordinate>{params_};
          c_out   = op(c_out, dir);
          break;
        }
      }
    });
    return c_out;
  }

  projection_parameters<T> params_;
  operation_type const* d_ops;
  iterator_type first_;
  std::size_t num_stages;
};

}  // namespace detail

template <typename Coordinate, typename T = typename Coordinate::value_type>
class projection {
 public:
  __host__ projection(std::vector<operation_type> const& operations,
                      projection_parameters<T> const& params)
    : params_(params)
  {
    setup(operations);
  }

  template <class CoordIter>
  void transform(CoordIter first,
                 CoordIter last,
                 CoordIter result,
                 direction dir,
                 rmm::cuda_stream_view stream = rmm::cuda_stream_default) const
  {
    static_assert(std::is_same_v<typename CoordIter::value_type, Coordinate>,
                  "Coordinate type must match iterator value type");

    if (dir == direction::FORWARD) {
      auto pipe = detail::pipeline<Coordinate, direction::FORWARD>{
        params_, operations_.data().get(), operations_.size()};
      thrust::transform(rmm::exec_policy(stream), first, last, result, pipe);
    } else {
      auto pipe = detail::pipeline<Coordinate, direction::INVERSE>{
        params_, operations_.data().get(), operations_.size()};
      thrust::transform(rmm::exec_policy(stream), first, last, result, pipe);
    }
  }

 private:
  void setup(std::vector<operation_type> const& operations)
  {
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
  }

  thrust::device_vector<operation_type> operations_;
  projection_parameters<T> params_;
};

}  // namespace cuproj
