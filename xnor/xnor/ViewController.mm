//
//  ViewController.m
//  xnor
//
//  Created by Nandini Ramakrishnan on 08/04/17.
//  Copyright © 2017 Shalini Ramakrishnan. All rights reserved.
//

#import "ViewController.h"
#import <Accelerate/Accelerate.h>
#import <Foundation/Foundation.h>

#ifdef __cplusplus
#include "armadillo"
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
//#include <arm_neon.h>
#endif

@interface ViewController ()

@end

@implementation ViewController

unsigned char* read_mnist_labels(std::string full_path, int number_of_labels) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };
    
    typedef unsigned char uchar;
    
    std::ifstream file(full_path, std::ios::binary);
    
    if(file.is_open()) {
        int magic_number = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        
        if(magic_number != 2049) throw std::runtime_error("Invalid MNIST label file!");
        
        file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);
        
        uchar* _dataset = new uchar[number_of_labels];
        for(int i = 0; i < number_of_labels; i++) {
            file.read((char*)&_dataset[i], 1);
            std::cout << _dataset[i] << std::endl;

        }
        return _dataset;
    } else {
        throw std::runtime_error("Unable to open file");
    }
}

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
//    
//    arma::wall_clock timer;
//    
//    int D = 4;
//    
//    // initialize random data
//    arma::fmat A; A.randn(D,1);
//    
//    uint8_t data_im[8] = { 0, 8, 14, 2, 7, 7, 0, 1 };
//    
//    int channels = 2;
//    int height = 3;
//    int width = 3;
//    int ksize = 2;
//    int stride = 1;
//    int pad = 0;
//    float *data_col = new float[32];
//    im2col(data_im,
//           channels,  height,  width,
//           ksize,  stride, pad, data_col);
//    
//    
//    for (int i = 0; i < 32; i++) {
//        cout << data_col[i] << " " << endl;
//    }
//    arma::mat x = { 1, 1, 2, 0 };
//    x = repmat(x, 1, 2);
//    arma::vec v = vectorise(x);
//    double *vecData = v.memptr();
//    
//    for (int i = 0; i < 8; i++) {
//        cout << vecData[i] << endl;
//    }
//    
//    
//    uint8x8_t vec1 = { 1, 1 };
//    uint8x8_t vec2 = { 0, 1 };
//    uint8x8_t vec3 = veor_u8(vec1, vec2);
//    cout << vec3[0] << " " << vec3[1] << endl;
//    cout << "Get sign bits ----" << endl;
//    vFloat w = { 1 , -3, -2, 6 };
//    vFloat e = { 9, 2, -1, -7 };
//    vUInt32 res1 = vsignbitf(w);             // DO PLAIN XOR ON THIS
//    vUInt32 res2 = vsignbitf(e);             // DO PLAIN XOR ON THIS
//    res1 = (uint32x4_t) res1;
//    res2 = (uint32x4_t) res2;
//    res1 = veorq_u32(res1, res2);
//    cout << res1[0] << " " << res1[1] << " " << res1[2] << " " << res1[3] << endl;
    
    /* Image dimension details */
    int imageWidth = 28;
    int imageHeight = 28;
    int numPixelsPerImage = imageWidth * imageHeight;
    //int numTrainImages = 60000;
    int numTestImages = 10000;
    
    uint8_t *images = [self getImages];
    uint8_t *labels = [self getLabels];
    arma::uchar_mat X(images, numPixelsPerImage, numTestImages);      /* Each column is an image */
    printf("%u\n",X.max());
    printf("%u\n", labels[3000]);
}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

- (uint8_t *)getImages {
    int imageOffset = 16;
    NSBundle *main = [NSBundle mainBundle];
    NSURL *URLI = [main URLForResource:@"t10k-images-idx3-ubyte" withExtension:@"data"];
    NSData *imageData = [[NSData alloc] initWithContentsOfURL:URLI];
    NSString *imagePath = [main pathForResource:@"t10k-images-idx3-ubyte" ofType:@"data"];
    std::string imagePathString = std::string([imagePath UTF8String]);
    int numImageBytes = imageData.length;
    int fd_i = open(imagePathString.c_str(), O_RDONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);
    if (fd_i == -1) {
        std::cout << "Error: failed to open output file at " << imagePathString << ". errno = " << errno << std::endl;
    }
    uint8_t *images = (uint8_t *)mmap(NULL, numImageBytes, PROT_READ, MAP_FILE | MAP_SHARED, fd_i, 0);
    images = images + imageOffset;
    return images;
}

- (uint8_t *)getLabels {
    int labelOffset = 8;
    NSBundle *main = [NSBundle mainBundle];
    NSURL *URLL = [main URLForResource:@"train-labels-idx1-ubyte" withExtension:@"data"];
    NSData *labelData = [[NSData alloc] initWithContentsOfURL:URLL];
    NSString *labelPath = [main pathForResource:@"train-labels-idx1-ubyte" ofType:@"data"];
    std::string labelPathString = std::string([labelPath UTF8String]);
    int numLabels = labelData.length;
    int fd_l = open(labelPathString.c_str(), O_RDONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);
    if (fd_l == -1) {
        std::cout << "Error: failed to open output file at " << labelPathString << ". errno = " << errno << std::endl;
    }
    uint8_t *labels = (uint8_t *)mmap(NULL, numLabels, PROT_READ, MAP_FILE | MAP_SHARED, fd_l, 0);
    labels = labels + labelOffset;
    return labels;
}

@end
