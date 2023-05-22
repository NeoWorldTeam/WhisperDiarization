//
//  AggClusteringWrapper.m
//  WhisperDiarization
//
//  Created by fuhao on 2023/5/6.
//

#import <Foundation/Foundation.h>
#import "AggClusteringWrapper2.h"
#import <AggClustering/AggClusteringWrapper.h>


@implementation AggClusteringWrapper2

AggClusteringWrapper * _delegate;

- (instancetype)init {
    if (self = [super init]){
        _delegate = [[AggClusteringWrapper alloc] init];
    }
    return self;
}


//-(void) agglomerativeClustering:(float*) dist Row:(int) row Labels:(int*) labels {
//   
//}


-(void) agglomerativeClustering:(float*) dist Row:(int) row MinClusterNum:(int) minClusterNum MaxClusterNum:(int) maxClusterNum Labels:(int*) labels {
    [_delegate agglomerativeClustering:dist Row:row MinClusterNum:minClusterNum MaxClusterNum:maxClusterNum Labels:labels];
}

@end
