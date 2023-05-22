//
//  AggClusteringWrapper.m
//  WhisperDiarization
//
//  Created by fuhao on 2023/5/6.
//

#import <Foundation/Foundation.h>
#import "AggClusteringWrapper.h"
#import "AggClustering.hpp"


@implementation AggClusteringWrapper

-(void) agglomerativeClustering:(float*) dist Row:(int) row Labels:(int*) labels {
    float sim_threshold = 0.4;
    CSAlgorithm::AggClustering* clusterer = new CSAlgorithm::AggClustering();
    if (!clusterer->init(dist, row)) {
        fprintf(stderr, "====init clusterer failed!\n");
        return ;
    }
    if (!clusterer->doCluster()) {
        fprintf(stderr, "====cluster nodes failed!\n");
        return ;
    }
    clusterer->output(sim_threshold, labels);
    delete clusterer;
}


-(void) agglomerativeClustering:(float*) dist Row:(int) row MinClusterNum:(int) minClusterNum MaxClusterNum:(int) maxClusterNum Labels:(int*) labels {
    CSAlgorithm::AggClustering* clusterer = new CSAlgorithm::AggClustering();
    if (!clusterer->init(dist, row)) {
        fprintf(stderr, "====init clusterer failed!\n");
        return ;
    }
    if (!clusterer->doCluster()) {
        fprintf(stderr, "====cluster nodes failed!\n");
        return ;
    }
    clusterer->cutTree(minClusterNum, maxClusterNum, labels);
    delete clusterer;
}

@end
