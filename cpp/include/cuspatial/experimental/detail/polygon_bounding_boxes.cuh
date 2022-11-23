/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <rmm/cuda_stream_view.hpp>

namespace cuspatial {

template <class OffsetIteratorA,
          class OffsetIteratorB,
          class PointIterator,
          class BoundingBoxIterator>
BoundingBoxIterator polygon_bounding_boxes(OffsetIteratorA polygon_offsets_first,
                                           OffsetIteratorA polygon_offsets_last,
                                           OffsetIteratorB polygon_ring_offsets_first,
                                           OffsetIteratorB polygon_ring_offsets_last,
                                           PointIterator polygon_points_first,
                                           PointIterator polygon_points_last,
                                           BoundingBoxIterator output_first,
                                           rmm::cuda_stream_view stream)
{
  return BoundingBoxIterator{};
}
}  // namespace cuspatial
