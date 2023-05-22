//
//  SilhouetteScoreWrapper.h
//  WhisperDiarization
//
//  Created by fuhao on 2023/5/8.
//

@interface SilhouetteScoreWrapper : NSObject

- (float)score:(float**) distances Labels:(int*) labels ItemNum:(int)itemNum Cluster:(int)k ;

@end
