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
#include <fcntl.h>
#include "Convolution.hpp"
#endif

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    using namespace std;

    /* Image dimension details */
    int imageWidth = 28;
    int imageHeight = 28;
    int numPixelsPerImage = imageWidth * imageHeight;
    
    /* Dataset size details */
    int numTrainImages = 60000;
    int numTestImages = 10000;
    
    /* Get training data */
    uint8_t *images = [self getImages:@"train-images-idx3-ubyte"];
    uint8_t *labels = [self getLabels:@"train-labels-idx1-ubyte"];
    
    /* Convert image vector into matrix where each column is an image */
    arma::uchar_mat train_mat(images, numPixelsPerImage, numTrainImages);
    
    /* Perform binary convolution on first image (for testing) */
    uint8_t *image1 = train_mat.colptr(0);
    /* Toy example! */
    uint8_t toy[16] = { 0, 0, 4, 0, 0, 2, 5, 0, 0, 0, 1, 0, 0, 0, 2, 0 };
    int h_in = 4;
    int w_in = 4;
    /* Create conv1 layer */
    int k = 2;
    int stride = 1;
    int c = 1;
    int pad = 0;
    int group = 1;
    int num = 2;
    Convolution *conv1 = new Convolution(k, stride, c, pad, group, num);
    arma::mat binConvResult = conv1->binConv(toy, h_in, w_in);
    cout << binConvResult << endl;
    
}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

- (unsigned char *)getImages:(NSString*)path {
    int imageOffset = 16;
    NSBundle *main = [NSBundle mainBundle];
    NSURL *URLI = [main URLForResource:path withExtension:@"data"];
    NSData *imageData = [[NSData alloc] initWithContentsOfURL:URLI];
    NSString *imagePath = [main pathForResource:path ofType:@"data"];
    std::string imagePathString = std::string([imagePath UTF8String]);
    int numImageBytes = imageData.length;
    int fd_i = open(imagePathString.c_str(), O_RDONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);
    if (fd_i == -1) {
        std::cout << "Error: failed to open output file at " << imagePathString << ". errno = " << errno << std::endl;
    }
    unsigned char *images = (unsigned char *)mmap(NULL, numImageBytes, PROT_READ, MAP_FILE | MAP_SHARED, fd_i, 0);
    images = images + imageOffset;
    return images;
}

- (uint8_t *)getLabels:(NSString*)path {
    int labelOffset = 8;
    NSBundle *main = [NSBundle mainBundle];
    NSURL *URLL = [main URLForResource:path withExtension:@"data"];
    NSData *labelData = [[NSData alloc] initWithContentsOfURL:URLL];
    NSString *labelPath = [main pathForResource:path ofType:@"data"];
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
