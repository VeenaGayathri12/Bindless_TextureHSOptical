/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "common.h"

///////////////////////////////////////////////////////////////////////////////
/// \brief downscale image
///
/// CUDA kernel, relies heavily on texture unit
/// \param[in]  width   image width
/// \param[in]  height  image height
/// \param[in]  stride  image stride
/// \param[out] out     result
///////////////////////////////////////////////////////////////////////////////
void DownscaleKernel(
    int width, int height, int stride, float *out,
    sycl::ext::oneapi::experimental::sampled_image_handle texFine,
    const sycl::nd_item<3> &item_ct1) {
  const int ix = item_ct1.get_local_id(2) +
                 item_ct1.get_group(2) * item_ct1.get_local_range(2);
  const int iy = item_ct1.get_local_id(1) +
                 item_ct1.get_group(1) * item_ct1.get_local_range(1);

  if (ix >= width || iy >= height) {
    return;
  }

  float dx = 1.0f / (float)width;
  float dy = 1.0f / (float)height;

  float x = ((float)ix + 0.5f) * dx;
  float y = ((float)iy + 0.5f) * dy;

  out[ix + iy * stride] = 0.25f * (tex2D<float>(texFine, x - dx * 0.25f, y) +
                                   tex2D<float>(texFine, x + dx * 0.25f, y) +
                                   tex2D<float>(texFine, x, y - dy * 0.25f) +
                                   tex2D<float>(texFine, x, y + dy * 0.25f));
}

///////////////////////////////////////////////////////////////////////////////
/// \brief downscale image
///
/// \param[in]  src     image to downscale
/// \param[in]  width   image width
/// \param[in]  height  image height
/// \param[in]  stride  image stride
/// \param[out] out     result
///////////////////////////////////////////////////////////////////////////////
static void Downscale(const float *src, int width, int height, int stride,
                      int newWidth, int newHeight, int newStride, float *out) {
  dpct::dim3 threads(32, 8);
  dpct::dim3 blocks(iDivUp(newWidth, threads.x), iDivUp(newHeight, threads.y));

  sycl::ext::oneapi::experimental::sampled_image_handle texFine;
  dpct::image_data texRes;
  memset(&texRes, 0, sizeof(dpct::image_data));

  texRes.set_data_type(dpct::image_data_type::pitch);
  texRes.set_data_ptr((void *)src);
  texRes.set_channel(cudaCreateChannelDesc<float>());
  texRes.set_x(width);
  texRes.set_y(height);
  texRes.set_pitch(stride * sizeof(float));

  dpct::sampling_info texDescr;
  memset(&texDescr, 0, sizeof(dpct::sampling_info));

  texDescr.set(sycl::coordinate_normalization_mode::normalized);
  texDescr.set(sycl::filtering_mode::linear);
  texDescr.set(sycl::addressing_mode::mirrored_repeat);
  texDescr.set(sycl::addressing_mode::mirrored_repeat);
  /*
  DPCT1062:1: SYCL Image doesn't support normalized read mode.
  */

  checkCudaErrors(DPCT_CHECK_ERROR(
      texFine = dpct::experimental::create_bindless_image(texRes, texDescr)));

  /*
  DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_in_order_queue().parallel_for(
      sycl::nd_range<3>(blocks * threads, threads),
      [=](sycl::nd_item<3> item_ct1) {
        DownscaleKernel(newWidth, newHeight, newStride, out, texFine, item_ct1);
      });
}
