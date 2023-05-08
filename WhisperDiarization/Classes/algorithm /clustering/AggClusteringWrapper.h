//
//  AggClusteringWrapper.h
//  WhisperDiarization
//
//  Created by fuhao on 2023/5/6.
//
#import <Foundation/Foundation.h>
@interface AggClusteringWrapper : NSObject

-(void) agglomerativeClustering:(float*) dist Row:(int) row Labels:(int*) labels;

-(void) agglomerativeClustering:(float*) dist Row:(int) row ClusterNum:(int) clusterNum Labels:(int*) labels;

@end
