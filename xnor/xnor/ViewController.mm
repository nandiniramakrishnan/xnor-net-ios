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

@interface ViewController () {
    UIImageView *imageView_;
}
@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    using namespace std;

    UIImage *image = [UIImage imageNamed:@"image.jpeg"];
    if(image == nil) cout << "Cannot read in the file image.jpeg!!" << endl;
    
    // Setup the display
    // Setup the your imageView_ view, so it takes up the entire App screen......
    imageView_ = [[UIImageView alloc] initWithFrame:CGRectMake(0.0, 0.0, self.view.frame.size.width, self.view.frame.size.height)];
    // Important: add OpenCV_View as a subview
    [self.view addSubview:imageView_];
    // Ensure aspect ratio looks correct
    imageView_.contentMode = UIViewContentModeScaleAspectFit;
    UIImage *grey_im = [self convertToGreyscale:image];
    
    
    CFDataRef rawData = CGDataProviderCopyData(CGImageGetDataProvider(grey_im.CGImage));
    UInt8 * buf = (UInt8 *) CFDataGetBytePtr(rawData);

    /* Image dimension details */
//    int imageWidth = 28;
//    int imageHeight = 28;
//    int numPixelsPerImage = imageWidth * imageHeight;
//    
//    /* Dataset size details */
//    int numTrainImages = 60000;
//    int numTestImages = 10000;
//    
//    /* Get training data */
//    uint8_t *images = [self getImages:@"train-images-idx3-ubyte"];
//    uint8_t *labels = [self getLabels:@"train-labels-idx1-ubyte"];
//    
//    /* Convert image vector into matrix where each column is an image */
//    arma::uchar_mat train_mat(images, numPixelsPerImage, numTrainImages);
//    
//    /* Perform binary convolution on first image (for testing) */
//    uint8_t *image1 = train_mat.colptr(0);
//    /* Toy example! */
//    uint8_t toy[16] = { 0, 0, 4, 0, 0, 2, 5, 0, 0, 0, 1, 0, 0, 0, 2, 0 };
    
    
    
    int h_in = 636;
    int w_in = 951;
    /* Create conv1 layer */
    int k = 3;
    int stride = 1;
    int c = 1;
    int pad = 0;
    int group = 1;
    int num = 1;
    Convolution *conv1 = new Convolution(k, stride, c, pad, group, num);
    arma::mat binConvResult = conv1->binConv(buf, h_in, w_in);
    cout << binConvResult << endl;

    
    imageView_.image = grey_im;
}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

- (UIImage *) convertToGreyscale:(UIImage *)i {
    
    int kRed = 1;
    int kGreen = 2;
    int kBlue = 4;
    
    int colors = kGreen | kBlue | kRed;
    int m_width = i.size.width;
    int m_height = i.size.height;
    
    uint32_t *rgbImage = (uint32_t *) malloc(m_width * m_height * sizeof(uint32_t));
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(rgbImage, m_width, m_height, 8, m_width * 4, colorSpace, kCGBitmapByteOrder32Little | kCGImageAlphaNoneSkipLast);
    CGContextSetInterpolationQuality(context, kCGInterpolationHigh);
    CGContextSetShouldAntialias(context, NO);
    CGContextDrawImage(context, CGRectMake(0, 0, m_width, m_height), [i CGImage]);
    CGContextRelease(context);
    CGColorSpaceRelease(colorSpace);
    
    // now convert to grayscale
    uint8_t *m_imageData = (uint8_t *) malloc(m_width * m_height);
    for(int y = 0; y < m_height; y++) {
        for(int x = 0; x < m_width; x++) {
            uint32_t rgbPixel=rgbImage[y*m_width+x];
            uint32_t sum=0,count=0;
            if (colors & kRed) {sum += (rgbPixel>>24)&255; count++;}
            if (colors & kGreen) {sum += (rgbPixel>>16)&255; count++;}
            if (colors & kBlue) {sum += (rgbPixel>>8)&255; count++;}
            m_imageData[y*m_width+x]=sum/count;
        }
    }
    free(rgbImage);
    
    // convert from a gray scale image back into a UIImage
    uint8_t *result = (uint8_t *) calloc(m_width * m_height *sizeof(uint32_t), 1);
    
    // process the image back to rgb
    for(int i = 0; i < m_height * m_width; i++) {
        result[i*4]=0;
        int val=m_imageData[i];
        result[i*4+1]=val;
        result[i*4+2]=val;
        result[i*4+3]=val;
    }
    
    // create a UIImage
    colorSpace = CGColorSpaceCreateDeviceRGB();
    context = CGBitmapContextCreate(result, m_width, m_height, 8, m_width * sizeof(uint32_t), colorSpace, kCGBitmapByteOrder32Little | kCGImageAlphaNoneSkipLast);
    CGImageRef image = CGBitmapContextCreateImage(context);
    CGContextRelease(context);
    CGColorSpaceRelease(colorSpace);
    UIImage *resultUIImage = [UIImage imageWithCGImage:image];
    CGImageRelease(image);
    
    free(m_imageData);
    
    // make sure the data will be released by giving it to an autoreleased NSData
    [NSData dataWithBytesNoCopy:result length:m_width * m_height];
    
    return resultUIImage;
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
