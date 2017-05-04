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
#import <GPUImage/GPUImage.h>

#ifdef __cplusplus
#include "armadillo"
#include <stdlib.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/ios.h"
#include "Convolution.hpp"
#endif

@interface ViewController () {
    UIImageView *imageView_;
}
@end

@implementation ViewController

-(UIImage *)boxblurImage:(UIImage *)image k:(int16_t *)k ksize:(int)ksize {
    //Get CGImage from UIImage
    CGImageRef img = image.CGImage;
    
    //setup variables
    vImage_Buffer inBuffer, outBuffer;
    
    vImage_Error error;
    
    void *pixelBuffer;
    
    //create vImage_Buffer with data from CGImageRef
    
    //These two lines get get the data from the CGImage
    CGDataProviderRef inProvider = CGImageGetDataProvider(img);
    CFDataRef inBitmapData = CGDataProviderCopyData(inProvider);
    
    //The next three lines set up the inBuffer object based on the attributes of the CGImage
    //uint8_t toy[16] = { 0, 0, 4, 0, 0, 2, 5, 0, 0, 0, 1, 0, 0, 0, 2, 0 };
    inBuffer.width = CGImageGetWidth(img);
    inBuffer.height = CGImageGetHeight(img);
    inBuffer.rowBytes = CGImageGetBytesPerRow(img);
//    inBuffer.width = 4;
//    inBuffer.height = 4;
//    inBuffer.rowBytes = 4;
    
    //This sets the pointer to the data for the inBuffer object
    inBuffer.data = (void*)CFDataGetBytePtr(inBitmapData);
    //inBuffer.data = (void*)toy;
    //create vImage_Buffer for output
    
    //allocate a buffer for the output image and check if it exists in the next three lines
    
    pixelBuffer = malloc(CGImageGetBytesPerRow(img) * CGImageGetHeight(img));
    //pixelBuffer = malloc(16);
    if(pixelBuffer == NULL)
        NSLog(@"No pixelbuffer");
    
    //set up the output buffer object based on the same dimensions as the input image
    outBuffer.data = pixelBuffer;
    outBuffer.width = CGImageGetWidth(img);
    outBuffer.height = CGImageGetHeight(img);
    outBuffer.rowBytes = CGImageGetBytesPerRow(img);
//    outBuffer.data = pixelBuffer;
//    outBuffer.width = 4;
//    outBuffer.height = 4;
//    outBuffer.rowBytes = 4;
    
    //perform convolution - this is the call for our type of data
    //error = vImageBoxConvolve_ARGB8888(&inBuffer, &outBuffer, NULL, 0, 0, boxSize, boxSize, NULL, kvImageEdgeExtend);
    //error = vImageConvolve_ARGB8888(&inBuffer, &outBuffer, NULL, 0, 0, k, ksize, ksize, 2, NULL, kvImageEdgeExtend);

    error = vImageConvolve_Planar8(&inBuffer, &outBuffer, NULL, 0, 0, k, ksize, ksize, 2, NULL, kvImageEdgeExtend);

    
    //check for an error in the call to perform the convolution
    if (error) {
        NSLog(@"error from convolution %ld", error);
    }
    
    //create CGImageRef from vImage_Buffer output
    //1 - CGBitmapContextCreateImage -
    
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceGray();
    
    CGContextRef ctx = CGBitmapContextCreate(outBuffer.data,
                                             outBuffer.width,
                                             outBuffer.height,
                                             8,
                                             outBuffer.rowBytes,
                                             colorSpace,
                                             kCGImageAlphaNone);
    CGImageRef imageRef = CGBitmapContextCreateImage (ctx);
    
    UIImage *returnImage = [UIImage imageWithCGImage:imageRef];
    
    //clean up
    CGContextRelease(ctx);
    CGColorSpaceRelease(colorSpace);
    
    free(pixelBuffer);
    CFRelease(inBitmapData);
    
    CGColorSpaceRelease(colorSpace);
    CGImageRelease(imageRef);
    
    return returnImage;
}

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
    
    cv::Mat cvImage;
    UIImageToMat(image, cvImage);
    
    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(cvImage, gray, CV_RGBA2GRAY);
    UIImage *finalImage = MatToUIImage(gray);
    
    CFDataRef rawData = CGDataProviderCopyData(CGImageGetDataProvider(finalImage.CGImage));
    CFIndex ind = CFDataGetLength(rawData);
    uint8_t * buf = (uint8_t *) CFDataGetBytePtr(rawData);
    int h_in = image.size.height;
    int w_in = image.size.width;

    /* Create conv1 layer */
    int stride = 1;
    int c = 1;
    int pad = 0;
    int group = 1;
    int num = 1;


    //    /* Toy example! */
//    uint8_t toy[16] = { 0, 0, 4, 0, 0, 2, 5, 0, 0, 0, 1, 0, 0, 0, 2, 0 };
//    int ksize = 2;
    //        int h_in = 4;
    //        int w_in = 4;
//    int16_t kernel[4] = {1, 1, 2, 0};
    int ksize = 3;
    int16_t kernel[9] = { 1, 2, 1, 2, 8, 2, 1, 2, 1 };
    
    UIImage *pp = [self boxblurImage:finalImage k:kernel ksize:ksize];

    Convolution *conv1 = new Convolution(ksize, stride, c, pad, group, num);
    arma::fmat binConvResult = conv1->binConv(buf, h_in, w_in);
    binConvResult = binConvResult / arma::max(arma::max(binConvResult));
    cout << arma::max(arma::max(binConvResult)) << endl;
    arma::uchar_mat Q = arma::conv_to<arma::uchar_mat>::from(binConvResult * 255);
    cv::Mat res = Arma2Cv(Q);
    UIImage *resim = MatToUIImage(res);
    imageView_.image = resim;
}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}
//==============================================================================
// Quick function to convert to Armadillo matrix header
arma::fmat Cv2Arma(cv::Mat &cvX)
{
    arma::fmat X(cvX.ptr<float>(0), cvX.cols, cvX.rows, false); // This is the transpose of the OpenCV X_
    return X; // Return the new matrix (no new memory allocated)
}
//==============================================================================
// Quick function to convert to OpenCV (floating point) matrix header
cv::Mat Arma2Cv(arma::uchar_mat &X)
{
    cv::Mat cvX = cv::Mat(X.n_cols, X.n_rows,CV_8U, X.memptr()).clone();
    return cvX; // Return the new matrix (new memory allocated)
}

@end
