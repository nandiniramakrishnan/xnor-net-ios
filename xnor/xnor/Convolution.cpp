//
//  Convolution.cpp
//  xnor
//
//  Created by Nandini Ramakrishnan on 29/04/17.
//  Copyright © 2017 Shalini Ramakrishnan. All rights reserved.
//

#include "Convolution.hpp"
#include "im2col.hpp"
#include <arm_neon.h>

Convolution::Convolution(int k,int stride, int c, int pad, int group, int num) {
    this->k = 5;
    this->stride = stride;
    this->pad = pad;
    this->group = group;
    this->num = num;
    this->channel = c;
}

arma::fmat Convolution::binConv(uint8_t *input, int h_in, int w_in) {
    
    /* Toy example weights */
    //arma::fmat w(k*k, 1);
    //w.fill(1/(k*k));
    //arma::fmat w = { 1, 1, 1, 1, 1, 8, 8, 1, 1, 8, 8, 1, 1, 1, 1, 1 };
    //arma::fmat w = { 9, 0, 9, 7, 0, 7, 9, 0, 9 };
    //arma::fmat w = { 1, 1, 1, 1, 1, 1, 2, 4, 2, 1, 1, 4, 8, 4, 1, 1, 2, 4, 2, 1, 1, 1, 1, 1, 1 };
    arma::fmat w(1, k*k);
    w.fill(1.0);
    w = w.t();
    
    /* Average of weights */
    arma::fmat weight_avgs = arma::mean(w);
    
    /* Obtain patches from im2col */
    int h_out = (h_in + 2*pad - k) / stride + 1;
    int w_out = (w_in + 2*pad - k) / stride + 1;
    int numPatches = h_out * w_out;
    int sizeCol = k * k * channel * numPatches;
    float *data_col = new float[sizeCol];
    im2col(input,
               1,  h_in,  w_in,
               k,  stride, pad, data_col);
    
    
    /* Vector size for NEON operations */
    int neon_vec_size = 16;
    int patch_size = k*k;
    int num_vecs_per_patch = ceil((float)patch_size / (float)neon_vec_size);
    
    /* Get averages of sub tensors */
    arma::fmat patches(data_col, h_out*w_out, k*k);
    
//    std::cout << "mean = " << arma::max(arma::max(patches)) << std::endl;
//    std::cout << "stddev = " << arma::stddev(arma::vectorise(patches)) << std::endl;
    
    patches = (patches / arma::max(arma::max(patches))); /// arma::stddev(arma::vectorise(patches));
    w = w / arma::max(arma::max(w));
    arma::fmat K = arma::mean(patches, 1);
    /* Get signs of input patches using Armadillo */
    
    arma::uchar_mat input_signs(numPatches, neon_vec_size*num_vecs_per_patch);
    input_signs(arma::span::all, arma::span(0,k*k - 1) ) = arma::conv_to<arma::uchar_mat>::from(arma::sign(patches));
    input_signs = input_signs.t();  // Transpose the matrix to get patches as cols
    //input_signs.insert_rows( patch_size - 1, neon_vec_size - patch_size );
    printf("input signs done\n");
    /* Get signs of weights using Armadillo */
    arma::uchar_mat weight_signs(neon_vec_size*num_vecs_per_patch, num);
    weight_signs(arma::span(0,k*k - 1), arma::span::all ) = arma::conv_to<arma::uchar_mat>::from(arma::sign(w));
  //  input_signs = arma::sign(input_signs);
    printf("weight signs done\n");
//    arma::uchar_mat weight_signs = arma::conv_to<arma::uchar_mat>::from(arma::sign(w));
//    weight_signs.insert_rows( patch_size - 1, neon_vec_size - patch_size );
    
    /* Temp vector for doing bitcount */
    uint8_t *bitcount_ptr = new uint8_t[neon_vec_size*neon_vec_size*num_vecs_per_patch];
    
    /* Declare return value */
    arma::fmat result(numPatches, num);
    printf("before the for loops\n");
    /* Perform the convolution */
    for (int i = 0; i < num; i++) {
        //printf("filter number = %d\n", i);
        uint8_t *weight_sign_ptr = weight_signs.colptr(i);
        arma::fmat scaling_factor = K * weight_avgs(i);
        for (int j = 0; j < numPatches; j++) {
            
            //printf("patch number = %d\n", j);
            uint8_t *input_sign_patch_ptr = input_signs.colptr(j);
            for (int m = 0; m < num_vecs_per_patch; m++) {
                uint8x16_t input_vec = vld1q_u8(&input_sign_patch_ptr[m*neon_vec_size]);
                uint8x16_t weight_vec = vld1q_u8(weight_sign_ptr);
                uint8x16_t and_res = vandq_u8(input_vec, weight_vec);   // VAND q0,q0,q0
//                for (int l = 0; l < neon_vec_size; l++) {
//                    printf("%u & %u = %u\n", input_vec[l], weight_vec[l], and_res[l]);
//                    printf("---\n");
//                }
                vst1q_u8(bitcount_ptr, and_res);
                // load into arma vec and sum
                
                arma::uchar_vec bitcount(bitcount_ptr, neon_vec_size);
//                for (int l = 0; l < neon_vec_size;  l++) {
//                    printf("%u", bitcount(l));
//                }
//                printf("\n%u\n", arma::sum(bitcount));
                result(j, i) += arma::sum(bitcount);
            }
        }
        //std::cout << "Scaling factor = " << scaling_factor << std::endl;
        result.col(i) = result.col(i) % scaling_factor;
    }
    

    return reshape(result, h_out, w_out);
}
