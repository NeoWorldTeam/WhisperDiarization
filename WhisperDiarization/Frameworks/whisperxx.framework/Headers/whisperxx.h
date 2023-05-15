//
//  whisperxx.h
//  whisperxx
//
//  Created by fuhao on 2023/4/20.
//

#import <Foundation/Foundation.h>

//! Project version number for whisperxx.
FOUNDATION_EXPORT double whisperxxVersionNumber;

//! Project version string for whisperxx.
FOUNDATION_EXPORT const unsigned char whisperxxVersionString[];

// In this header, you should import all the public headers of your framework using statements like #import <whisperxx/PublicHeader.h>


#import <AVFoundation/AVFoundation.h>
#import <AudioToolbox/AudioQueue.h>

#define NUM_BUFFERS 3
#define MAX_AUDIO_SEC 30
#define SAMPLE_RATE 16000

struct whisper_context;

typedef struct
{
    int ggwaveId;
    bool isCapturing;
    bool isTranscribing;
    bool isRealtime;


//    AudioQueueRef queue;
//    AudioStreamBasicDescription dataFormat;
//    AudioQueueBufferRef buffers[NUM_BUFFERS];
//
//    int n_samples;
//    int16_t * audioBufferI16;
//    float   * audioBufferF32;

    struct whisper_context * ctx;

//    void * vc;
} StateInp;




@interface WhisperWrapper : NSObject
{
    StateInp stateInp;
    NSString* modelPath;
    BOOL isLoaded;
}

-(instancetype)initWithModel:(NSString *)modelPath;

- (BOOL) LoadModel;
- (BOOL) process: (const float *) samples SampleNum:(int) n_samples;
- (int) getSegmentsNum;
- (NSString*) getSpeechBySegmentIndex:(int) index;
- (long long) getSpeechStartTimeBySegmentIndex:(int) index;
- (long long) getSpeechEndTimeBySegmentIndex:(int) index;
@end
