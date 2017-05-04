//
//  Convolution.hpp
//  xnor
//
//  Created by Nandini Ramakrishnan on 29/04/17.
//  Copyright © 2017 Shalini Ramakrishnan. All rights reserved.
//

#ifndef Convolution_hpp
#define Convolution_hpp

#include <stdio.h>
#import <Accelerate/Accelerate.h>
#include "armadillo"

class Convolution {
    int k;
    int stride;
    int channel;
    int pad;
    int group;
    int num;
    arma::fmat w;
    arma::fvec b;
public:
    Convolution(int k, int stride, int c, int pad, int group, int num);
    arma::fmat binConv(uint8_t *input, int h_in, int w_in);
};

#endif /* Convolution_hpp */
