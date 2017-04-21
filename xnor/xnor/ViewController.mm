//
//  ViewController.m
//  xnor
//
//  Created by Nandini Ramakrishnan on 08/04/17.
//  Copyright © 2017 Shalini Ramakrishnan. All rights reserved.
//

#import "ViewController.h"
//#import "Accelerate/Accelerate.h"

#ifdef __cplusplus
#include "armadillo"
#include <stdlib.h>
#include <stddef.h>
#include <arm_neon.h>
#endif

@interface ViewController ()

@end

@implementation ViewController

float im2col_get_pixel(uint8_t *im, int height, int width, int channels,
                       int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;
    
    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

void im2col(uint8_t* data_im,
                        int channels,  int height,  int width,
                        int ksize,  int stride, int pad, float* data_col)
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;
    
    
    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                                                       im_row, im_col, c_im, pad);
            }
        }
    }
}

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    using namespace std;
    
    arma::wall_clock timer;
    
    int D = 4;
    
    // initialize random data
    arma::fmat A; A.randn(D,1);
    
    uint8_t data_im[8] = { 0, 8, 14, 2, 7, 7, 0, 1 };
    
    int channels = 2;
    int height = 3;
    int width = 3;
    int ksize = 2;
    int stride = 1;
    int pad = 0;
    float *data_col = new float[32];
    im2col(data_im,
           channels,  height,  width,
           ksize,  stride, pad, data_col);
    
    
    for (int i = 0; i < 32; i++) {
        cout << data_col[i] << " " << endl;
    }
    arma::mat x = { 1, 1, 2, 0 };
    x = repmat(x, 1, 2);
    arma::vec v = vectorise(x);
    double *vecData = v.memptr();
    
    for (int i = 0; i < 8; i++) {
        cout << vecData[i] << endl;
    }
    
    
    uint8x8_t vec1 = { 1, 1 };
    uint8x8_t vec2 = { 0, 1 };
    uint8x8_t vec3 = veor_u8(vec1, vec2);
    cout << vec3[0] << " " << vec3[1] << endl;
    
}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


@end
