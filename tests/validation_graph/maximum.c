/*
 * Copyright (C) 2016-2021 C-SKY Limited. All rights reserved.
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

#include "test_utils.h"
#include "csi_nn.h"
#include "math_snr.h"
#include "csi_pnna.h"

int main(int argc, char** argv)
{
    init_testsuite("Testing function of maximum(graph).\n");

    struct csi_session *sess = csi_alloc_session();
    sess->base_api = CSINN_LIGHT;
    sess->base_dtype = CSINN_DTYPE_INT8;
    csi_session_init(sess);
    csi_set_input_number(2, sess);
    csi_set_output_number(1, sess);

    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    float min_value, max_value;
    int in0_size = 1, in1_size = 1, out_size = 1;

    int *buffer = read_input_data_f32(argv[1]);

    struct csi_tensor *input0  = csi_alloc_tensor(sess);
    input0->dim_count = buffer[0];
    for(int i = 0; i < input0->dim_count; i++) {
        input0->dim[i] = buffer[1 + i];
        in0_size *= input0->dim[i];
    }

    float *input0_data = (float *)(buffer + 1 + input0->dim_count);
    /* get input0 min max */
    find_min_max((float *)input0_data, &max_value, &min_value, in0_size);
    input0->qinfo->min = min_value;
    input0->qinfo->max = max_value;
    input0->name = "input0";


    struct csi_tensor *input1  = csi_alloc_tensor(sess);
    input1->dim_count = input0->dim_count;
    for(int i = 0; i < input1->dim_count; i++) {
        input1->dim[i] = input0->dim[i];
        in1_size *= input1->dim[i];
    }

    float *input1_data = (float *)(buffer + 1 + input0->dim_count + in0_size);
    /* get input1 min max */
    find_min_max((float *)input1_data, &max_value, &min_value, in1_size);
    input1->qinfo->min = min_value;
    input1->qinfo->max = max_value;
    input1->name = "input1";


    struct csi_tensor *output = csi_alloc_tensor(sess);
    output->dim_count = input0->dim_count;
    for(int i = 0; i < output->dim_count; i++) {
        output->dim[i] = csi_ref_max_internal_s32(input0->dim[i], input1->dim[i]);  // in fact, ouput->dim[i] are always equal to input0->dim[i]
        out_size *= output->dim[i];
    }

    reference->data = (float *)(buffer + 1 + input0->dim_count + in0_size + in1_size);
    /* get output min max */
    find_min_max((float *)reference->data, &max_value, &min_value, out_size);
    output->qinfo->min = min_value;
    output->qinfo->max = max_value;
    output->name = "output";


    struct diso_params params;
    params.base.api = CSINN_API;
    params.base.name = "params";
    params.base.layout = CSINN_NCHW;
    params.base.run_mode = CSINN_RM_NPU_GRAPH;


    /*
        support broadcast, but input tensor has unsupported batch broadcast
        has test some simple cases by broadcast
    */
    if (csi_maximum_init(input0, input1, output, &params) != CSINN_TRUE) {
        printf("maximum init fail.\n\t");
        return -1;
    }

    csi_pnna_input_setup(input0, sess);
    csi_pnna_input_setup(input1, sess);
    csi_set_input(0, input0, sess);
    csi_set_input(1, input1, sess);

    csi_maximum(input0, input1, output, &params);

    csi_set_output(0, output, sess);
    csi_session_setup(sess);

    struct csi_tensor *input0_tensor = csi_alloc_tensor(NULL);
    struct csi_tensor *input1_tensor = csi_alloc_tensor(NULL);
    input0_tensor->data = input0_data;
    input1_tensor->data = input1_data;
    csi_update_input(0, input0_tensor, sess);
    csi_update_input(1, input1_tensor, sess);
    csi_session_run(sess);

    struct csi_tensor *output_tensor = csi_alloc_tensor(NULL);
    output_tensor->is_const = 0;
    int output_num = csi_get_output_number(sess);
    printf("output_num = %d\n", output_num);
    csi_get_output(0, output_tensor, sess);

    /* FIX ME */
    float difference = argc > 2 ? atof(argv[2]) : 1e-4;
    result_verify_f32(reference->data, output_tensor->data, input0->data, difference, out_size, false);

    /* evaluate error by kl and cosine similarity */
    float *output_tensor_data = (float *)output_tensor->data;
    float kl = compute_kl(output_tensor_data, reference->data, out_size);
    printf("The kl diver is %f.\n", kl);
    float cs = compute_cs(output_tensor_data, reference->data, out_size);
    printf("The cos sim is %f.\n", cs);

    /* free alloced memory */
    free(buffer);
    free(input0_tensor->qinfo);
    free(input0_tensor);
    free(input1_tensor->qinfo);
    free(input1_tensor);
    free(output_tensor->qinfo);
    free(output_tensor);

    csi_session_deinit(sess);
    csi_free_session(sess);
    return done_testing();
}