/*
 * Copyright (C) 2016-2023 T-Head Semiconductor Co., Ltd. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "rvv/rvv.h"

/*************************************************************
 * packn = vlenb / sizeof(float)
 * n_blk: pack2n/packn/n_tail
 *
 * src: [n, k]
 * dst: [n/n_blk, k, n_blk]
 ************************************************************/
static void reorder_weight_npack2n_fp32(const float *src, float *dst, int n, int k)
{
    const int packn = csrr_vlenb() / sizeof(float);
    const int pack2n = packn * 2;

    int vl = vsetvl_e32m2(pack2n);
    int count = n / pack2n;
#pragma omp parallel
    for (int i = 0; i < count; ++i) {
        const float *s_ptr = src + i * k * pack2n;
        float *d_ptr = dst + i * k * vl;
        for (int j = 0; j < k; j++) {
            vfloat32m2_t _src = vlse32_v_f32m2(s_ptr, k * sizeof(float), vl);
            vse32_v_f32m2(d_ptr, _src, vl);
            s_ptr += 1;
            d_ptr += vl;
        }
    }

    int i = pack2n * count;
    dst += count * k * vl; 
    while (i < n) {
        int vl = vsetvl_e32m1(n - i);
        const float *s_ptr = src + i * k;
        for (int j = 0; j < k; j++) {
            vfloat32m1_t _src = vlse32_v_f32m1(s_ptr, k * sizeof(float), vl);
            vse32_v_f32m1(dst, _src, vl);
            s_ptr += 1;
            dst += vl;
        }
        i += vl;
    }
}

void shl_rvv_fc_gemm_reorder_weight_fp32(struct csinn_tensor *weights)
{
    float *weight_data = (float *)weights->data;
    int n = weights->dim[0];  // out_nodes
    int k = weights->dim[1];  // in_nodes
    float *pa_reorder = (float *)shl_mem_alloc(n * k * sizeof(float));
    reorder_weight_npack2n_fp32(weight_data, pa_reorder, n, k);
    memcpy(weight_data, pa_reorder, n * k * sizeof(float));
    shl_mem_free(pa_reorder);
}

int shl_rvv_fullyconnected_gemm_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                     struct csinn_tensor *weights, struct csinn_tensor *bias,
                                     struct csinn_fc_params *params)
{
    if (input->layout >= CSINN_LAYOUT_NC1C0 && input->layout <= CSINN_LAYOUT_NC1DHWC0) {
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_fp32(input);
    }

    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *weights_data = (float *)weights->data;
    float *bias_data = (float *)bias->data;
    const int output_dims_count = output->dim_count;
    const int weights_dims_count = weights->dim_count;

    int batches = 1;
    /* compute the outer size */
    for (int i = 0; i < output_dims_count - 1; i++) {
        batches *= output->dim[i];
    }
    int output_depth = weights->dim[weights_dims_count - 2];  // output_nodes
    int accum_depth = weights->dim[weights_dims_count - 1];   // input_nodes

    int m = batches;
    int n = output_depth;
    int k = accum_depth;

    float *input_reorder = (float *)shl_mem_alloc(m * k * sizeof(float));
    float *weights_reorder = (float *)shl_mem_alloc(n * k * sizeof(float));
    reorder_weight_npack2n_fp32(weights_data, weights_reorder, n, k);

#pragma omp parallel
{
   const int ithr = omp_get_thread_num();
   const int nthr = omp_get_num_threads();

   int m0 = 0, m1 = 0;
   shl_multithread_splitter(m, nthr, ithr, &m0, &m1);

   int local_m = m1 - m0;
   shl_rvv_reorder_a_block_12xk_fp32(input_data + m0 * k, input_reorder + m0 * k, local_m, k, local_m, k);
   shl_rvv_gemm_a0b1_12xpack2n_fp32(output_data + m0 * n, input_reorder + m0 * k, weights_reorder, bias_data, local_m, k, n);
}

    shl_mem_free(input_reorder);
    shl_mem_free(weights_reorder);
    return CSINN_TRUE;
}
