//
//  SilhouetteScoreWrapper.m
//  WhisperDiarization
//
//  Created by fuhao on 2023/5/8.
//

#import <Foundation/Foundation.h>
#import "silhouette_score.hpp"
#import "SilhouetteScoreWrapper.h"


@implementation SilhouetteScoreWrapper

- (float)score:(float**) distances Labels:(int*) labels ItemNum:(int)itemNum Cluster:(int)k {
    
    std::vector<std::vector<float>> vec(itemNum);
    for (int i = 0; i < itemNum; i++) {
        std::vector<float> row(itemNum);
        for (int j = 0; j < itemNum; j++) {
            row[j] = distances[i][j];
        }
        vec[i] = row;
    }
    
    std::vector<int> y(itemNum);
    for (int i = 0; i < itemNum; i++) {
        y[i] = labels[i];
    }
    float score = silhouetteScore(vec, y, k);
    return score;
}

@end
