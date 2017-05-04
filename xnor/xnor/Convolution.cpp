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
    this->k = k;
    this->stride = stride;
    this->pad = pad;
    this->group = group;
    this->num = num;
    this->channel = c;
    //this->w = arma::fmat(k*k*c/group, num, arma::fill::randn);
    this->b = arma::fvec(num, arma::fill::randn);
}

arma::fmat Convolution::binConv(uint8_t *input, int h_in, int w_in) {
    
    /* Toy example weights */
    arma::fmat w(k*k, 1);
    w.fill(1/(k*k));
    
//    w << 1 << 2 << arma::endr
//    << 1 << 1 << arma::endr
//    << 2 << 0 << arma::endr
//    << 0 << 2 << arma::endr;
    
    /* Average of weights */
    arma::fmat weight_avgs = arma::mean(w);
    
    /* Obtain patches from im2col */
    int h_out = (h_in + 2*pad - k) / stride + 1;
    int w_out = (w_in + 2*pad - k) / stride + 1;
    int numPatches = h_out * w_out;
    int sizeCol = k * k * channel * numPatches;
    uint8_t *data_col = new uint8_t[sizeCol];
    im2col(input,
               1,  h_in,  w_in,
               k,  stride, pad, data_col);
    
    /* Vector size for NEON operations */
    int neon_vec_size = 16;
    int patch_size = k*k;
    int num_vecs_per_patch = patch_size / neon_vec_size;
    
    /* Get averages of sub tensors */
    arma::uchar_mat patches(data_col, h_out*w_out, k*k);
    
    arma::fmat K = arma::mean(arma::conv_to<arma::fmat>::from(patches), 1);
    /* Get signs of input patches using Armadillo */
    
    arma::uchar_mat input_signs(numPatches, neon_vec_size);
    input_signs(arma::span::all, arma::span(0,k*k - 1) ) = patches;
    input_signs = arma::sign(input_signs);
    input_signs = input_signs.t();  // Transpose the matrix to get patches as cols
    std::cout << input_signs.n_rows << " " << input_signs.n_cols << std::endl;
    std::cout << patch_size << std::endl;
    //input_signs.insert_rows( patch_size - 1, neon_vec_size - patch_size );
    
    /* Get signs of weights using Armadillo */
    arma::uchar_mat weight_signs(neon_vec_size, num);
    std::cout << weight_signs.size() << std::endl;
    std::cout << "size of w = " << w.size() << std::endl;
    std::cout << weight_signs.n_rows << " " << weight_signs.n_cols << std::endl;
    std::cout << w.n_rows << " " << w.n_cols << std::endl;
    weight_signs(arma::span(0,k*k - 1), arma::span::all ) = arma::conv_to<arma::uchar_mat>::from(arma::sign(w));
  //  input_signs = arma::sign(input_signs);

//    arma::uchar_mat weight_signs = arma::conv_to<arma::uchar_mat>::from(arma::sign(w));
//    weight_signs.insert_rows( patch_size - 1, neon_vec_size - patch_size );
    
    /* Temp vector for doing bitcount */
    uint8_t *bitcount_ptr = new uint8_t[neon_vec_size];
    
    /* Declare return value */
    arma::fmat result(numPatches, num);
    printf("before the for loops\n");
    /* Perform the convolution */
    for (int i = 0; i < num; i++) {
        //printf("filter number = %d\n", i);
        uint8_t *weight_sign_ptr = weight_signs.colptr(i);
        arma::fmat scaling_factor = K * weight_avgs(i);
        for (int j = 0; j < numPatches; j++) {
            uint8_t *input_sign_patch_ptr = input_signs.colptr(j);
            //printf("patch number = %d\n", j);
            
            for (int m = 0; m < num_vecs_per_patch + 1; m++) {
                uint8x16_t input_vec = vld1q_u8(input_sign_patch_ptr);
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
                result(j, i) = arma::sum(bitcount);
            }
        }
        //std::cout << "Scaling factor = " << scaling_factor << std::endl;
        result.col(i) = result.col(i) % scaling_factor;
    }
    return result;
}
