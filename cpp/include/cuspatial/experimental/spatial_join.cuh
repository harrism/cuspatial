/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <cuspatial/detail/iterator.hpp>
#include <cuspatial/experimental/detail/join/intersection.cuh>
#include <cuspatial/experimental/detail/join/traversal.cuh>
#include <cuspatial/experimental/point_quadtree.cuh>

#include <rmm/device_uvector.hpp>

#include <iterator>
#include <utility>

namespace cuspatial {

/**
 * @addtogroup spatial_join
 * @{
 */

/**
 * @brief Search a quadtree for polygon or linestring bounding box intersections.
 *
 * @note 2D coordinates are converted into a 1D Morton code by dividing each x/y by the `scale`:
 * (`(x - min_x) / scale` and `(y - min_y) / scale`).
 * @note `max_depth` should be less than 16, since Morton codes are represented as `uint32_t`. The
 * eventual number of levels may be less than `max_depth` if the number of points is small or
 * `max_size` is large.
 *
 * @param keys_first: start quadtree key iterator
 * @param keys_last: end of quadtree key iterator
 * @param levels_first: start quadtree levels iterator
 * @param is_internal_nodes_first: start quadtree is_internal_node iterator
 * @param lengths_first: start quadtree length iterator
 * @param offsets_first: start quadtree offset iterator
 * @param bounding_boxes_first: start bounding boxes iterator
 * @param bounding_boxes_last: end of bounding boxes iterator
 * @param x_min The lower-left x-coordinate of the area of interest bounding box.
 * @param y_min The lower-left y-coordinate of the area of interest bounding box.
 * @param scale Scale to apply to each x and y distance from x_min and y_min.
 * @param max_depth Maximum quadtree depth at which to stop testing for intersections.
 * @param stream The CUDA stream on which to perform computations
 * @param mr The optional resource to use for output device memory allocations.
 *
 * @return A pair of UINT32 bounding box and leaf quadrant offset device vectors:
 *   - bbox_offset - indices for each polygon/linestring bbox that intersects with the quadtree.
 *   - quad_offset - indices for each leaf quadrant intersecting with a polygon/linestring bbox.
 *
 * @throw cuspatial::logic_error If scale is less than or equal to 0
 * @throw cuspatial::logic_error If max_depth is less than 1 or greater than 15
 */
template <class KeyIterator,
          class LevelIterator,
          class IsInternalIterator,
          class BoundingBoxIterator,
          class T = typename cuspatial::iterator_vec_base_type<BoundingBoxIterator>>
std::pair<rmm::device_uvector<uint32_t>, rmm::device_uvector<uint32_t>>
join_quadtree_and_bounding_boxes(
  KeyIterator keys_first,
  KeyIterator keys_last,
  LevelIterator levels_first,
  IsInternalIterator is_internal_nodes_first,
  KeyIterator lengths_first,
  KeyIterator offsets_first,
  BoundingBoxIterator bounding_boxes_first,
  BoundingBoxIterator bounding_boxes_last,
  T x_min,
  T y_min,
  T scale,
  int8_t max_depth,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default);

/**
 * @brief Test whether the specified points are inside any of the specified polygons.
 *
 * Uses the (polygon, quadrant) pairs returned by `cuspatial::join_quadtree_and_bounding_boxes` to
 * ensure only the points in the same quadrant as each polygon are tested for intersection.
 *
 * This pre-filtering can dramatically reduce the number of points tested per polygon, enabling
 * faster intersection testing at the expense of extra memory allocated to store the quadtree and
 * sorted point_indices.
 *
 * @param poly_quad_pairs_first iterator to the beginning of sequence of polygon/quadrant pairs
 * @param poly_quad_pairs_last iterator to the end of sequence of polygon/quadrant pairs
 * @param keys_first start quadtree key iterator
 * @param keys_last end of quadtree key iterator
 * @param levels_first start quadtree levels iterator
 * @param is_internal_nodes_first start quadtree is_internal_node iterator
 * @param lengths_first start quadtree length iterator
 * @param offsets_first start quadtree offset iterator
 * @param point_indices_first iterator to beginning of sequence of point indices returned by
 *                            `cuspatial::quadtree_on_points`
 * @param point_indices_last iterator to end of sequence of point indices returned by
 *                            `cuspatial::quadtree_on_points`
 * @param point_x x-coordinates of points to test
 * @param point_y y-coordinates of points to test
 * @param points_first iterator to beginning of sequence of (x, y) points to test
 * @param polygon_offsets_first iterator to beginning of range of indices to the first ring in each
                                polygon
 * @param polygon_offsets_last iterator to end of range of indices to the first ring in each polygon
 * @param ring_offsets_first iterator to beginning of range of indices to the first point in each
                             ring
 * @param ring_offsets_last iterator to end of range of indices to the first point in each ring
 * @param polygon_points_first iterator to beginning of range of polygon points
 * @param polygon_points_last iterator to end of range of polygon points
 * @param mr The optional resource to use for output device memory allocations.
 * @param stream The CUDA stream on which to perform computations
 *
 * @throw cuspatial::logic_error If the number of rings is less than the number of polygons.
 * @throw cuspatial::logic_error If any ring has fewer than four vertices.
 *
 * @return A pair of rmm::device_uvectors where each row represents a point/polygon intersection:
 *     polygon_offset - uint32_t polygon indices
 *     point_offset   - uint32_t point indices
 *
 * @note The returned polygon and point indices are offsets into the `poly_quad_pairs` input range
 *       and `point_indices` range, respectively.
 *
 **/
template <class PolyQuadPairIterator,
          class KeyIterator,
          class LevelIterator,
          class IsInternalIterator,
          class PointIndexIterator,
          class PointIterator,
          class PolygonOffsetIterator,
          class RingOffsetIterator,
          class VertexIterator>
std::pair<rmm::device_uvector<uint32_t>, rmm::device_uvector<uint32_t>> quadtree_point_in_polygon(
  PolyQuadPairIterator poly_quad_pairs_first,
  PolyQuadPairIterator poly_quad_pairs_last,
  KeyIterator keys_first,
  KeyIterator keys_last,
  LevelIterator levels_first,
  IsInternalIterator is_internal_nodes_first,
  KeyIterator lengths_first,
  KeyIterator offsets_first,
  PointIndexIterator point_indices_first,
  PointIndexIterator point_indices_last,
  PointIterator points_first,
  PolygonOffsetIterator polygon_offsets_first,
  PolygonOffsetIterator polygon_offsets_last,
  RingOffsetIterator polygon_ring_offsets_first,
  RingOffsetIterator polygon_ring_offsets_last,
  VertexIterator polygon_vertices_first,
  VertexIterator polygon_vertices_last,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default);

}  // namespace cuspatial

#include <cuspatial/experimental/detail/quadtree_bbox_filtering.cuh>
