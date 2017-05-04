//
//  im2col.hpp
//  xnor
//
//  Created by Nandini Ramakrishnan on 30/04/17.
//  Copyright © 2017 Shalini Ramakrishnan. All rights reserved.
//

#ifndef im2col_hpp
#define im2col_hpp

#include <stdio.h>
#import <Accelerate/Accelerate.h>


void im2col(uint8_t* data_im,
            int channels,  int height,  int width,
            int ksize,  int stride, int pad, float *data_col);

#endif /* im2col_hpp */
