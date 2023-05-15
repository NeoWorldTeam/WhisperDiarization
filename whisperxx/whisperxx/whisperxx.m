//
//  WhisperWrapper.m
//  WhisperDiarization
//
//  Created by fuhao on 2023/4/20.
//

#import "whisperxx.h"
#include "whisper.h"

@implementation WhisperWrapper

-(instancetype)initWithModel:(NSString *)modelPath {
    if([super init])
    {
        self->isLoaded = false;
        self->modelPath = modelPath;
    }
    return self;
}

- (void)dealloc {
    whisper_free(stateInp.ctx);
    NSLog(@"WhisperWrapper dealloc");
}

- (BOOL) LoadModel {
    if (self->isLoaded){
        return true;
    }

    if (![[NSFileManager defaultManager] fileExistsAtPath:self->modelPath]) {
        NSLog(@"Model file not found");
        return false;
    }
    
    // create ggml context
    stateInp.ctx = whisper_init_from_file([modelPath UTF8String]);

    // check if the model was loaded successfully
    if (stateInp.ctx == NULL) {
        NSLog(@"Failed to load model");
        return false;
    }
    self->isLoaded = true;
    return true;
}

-(struct whisper_full_params) initContext {
    // run the model
    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    // get maximum number of threads on this device (max 8)
    const int max_threads = MIN(8, (int)[[NSProcessInfo processInfo] processorCount]);

    params.print_realtime   = false;
    params.print_progress   = false;
    params.print_timestamps = true;
    params.print_special    = false;
    params.translate        = false;
    params.language         = "en";
    params.n_threads        = max_threads;
    params.offset_ms        = 0;
    params.no_context       = true;
    params.single_segment   = false;

    return params;
}

- (BOOL) process: (const float *) samples SampleNum:(int) n_samples {
    if ([self LoadModel] == false) {
        return FALSE;
    }
    
    struct whisper_full_params params = [self initContext];
    whisper_reset_timings(self->stateInp.ctx);
    if (whisper_full(self->stateInp.ctx, params, samples, n_samples) != 0) {
        NSLog(@"Failed to run the model");
        return FALSE;
    }
    return TRUE;
}
- (int) getSegmentsNum {
    int n_segments = whisper_full_n_segments(self->stateInp.ctx);
    return n_segments;
}
- (NSString*) getSpeechBySegmentIndex:(int) index {
    const char * text_cur = whisper_full_get_segment_text(self->stateInp.ctx, index);
    return [NSString stringWithUTF8String:text_cur];
}
- (long long) getSpeechStartTimeBySegmentIndex:(int) index {
    int64_t time = whisper_full_get_segment_t0(self->stateInp.ctx, index);
    return time;
}
- (long long) getSpeechEndTimeBySegmentIndex:(int) index {
    int64_t time = whisper_full_get_segment_t1(self->stateInp.ctx, index);
    return time;
}
@end
