//
//  ViewController.m
//  xnor
//
//  Created by Nandini Ramakrishnan on 08/04/17.
//  Copyright © 2017 Shalini Ramakrishnan. All rights reserved.
//

#import "ViewController.h"

#ifdef __cplusplus
#include "armadillo"
#include <stdlib.h>

#endif

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    using namespace arma;

    int D = 3000; // Number of columns in A
    int M = 400; // Number of rows in A
    int trials = 3000; // Number of trials
    arma::wall_clock timer;
    
    
    // initialize random data
    arma::fmat x; x.randn(D,1);
    arma::fmat A; A.randn(M,D);
    // intialize the clock
    
    
    
    timer.tic();
    // Step 4. apply matrix multiplication in Armadillo
    arma::fmat y(M,1); // Allocate space first
    timer.tic();
    for(int i=0; i<trials; i++) {
        y = A*x; // Apply multiplication in Armadillo
    }
    double arma_n_secs = timer.toc();
    cout << "Armadillo took " << arma_n_secs << " seconds." << endl;
    
}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


@end
