import Foundation
import whisperxx

//public typealias TranscriptCallBack = (_ result: [TranscriptSegment], _ error: WhisperError?)->Void


public enum WhisperError: Error {
    case error(message: String)
}


public struct TranscriptSegment {
    public var speech:String
    public var start: Int
    public var end: Int
    public init() {
        speech = ""
        start = 0
        end = 0
    }
    
    public init(speech:String,start:Int,end:Int) {
        self.speech = speech
        self.start = start
        self.end = end
    }
}

//public struct TranscriptResult{
//    public var speechs: [TranscriptSegment] = []
//    init() {}
//}

class WhisperDiarization {
//    let _queue: DispatchQueue
    var _whisper: WhisperWrapper?
//    var _isTranscripting: Bool = false
    
//    var _cacheBuffer: AVAudioPCMBuffer?
    
//    var  _callBack: TranscriptCallBack?
    init() {
//        _queue = DispatchQueue(label: "WhisperDiarization")
        
        
        guard let associateBundleURL2 = Bundle.main.url(forResource: "WhisperDiarization", withExtension: "bundle") else {
            return
        }
        
        guard let podBundle = Bundle(url: associateBundleURL2) else {
            return
        }
        
        guard let modelPath = podBundle.path(forResource: "ggml-tiny.en", ofType: "bin") else {
            print("Failed to get model file path with name.")
            return
        }
        
        
        
//        guard var associateBundleURL = Bundle.main.url(forResource: "Frameworks", withExtension: nil) else {
//            return
//        }
//
//        associateBundleURL.appendPathComponent("WhisperDiarization")
//        associateBundleURL.appendPathExtension("framework")
//
//        guard let podBundle = Bundle(url: associateBundleURL) else {
//            return
//        }
//        guard let associateBundleURL2 = podBundle.url(forResource: "WhisperDiarization", withExtension: "bundle") else {
//            returnwhisperxx.framework
//        }
//        guard let podBundle2 = Bundle(url: associateBundleURL2) else {
//            return
//        }
//        let modelPath = podBundle2.path(forResource: "ggml-tiny", ofType: "bin")
        _whisper = WhisperWrapper(model: modelPath, lang: "en")
    }
    
    
    deinit {
        _whisper = nil
        
        print("释放whisper")
    }
    

    
//    private func readDataFromFile(wavFile: URL) -> (Data,Int){
//        // 打开文件进行读取
//        let file = try! FileHandle(forReadingFrom: wavFile)
//
//        defer {
//            // 关闭文件
//            file.closeFile()
//        }
//
//        // 读取文件数据
//        let data = file.readDataToEndOfFile()
//        let numSamples = data.count / MemoryLayout<Float>.size
//
//        return (data,numSamples)
//    }
    
//    private func generateTranscriptSeg(samples: UnsafePointer<Float>, speech: String, t0: Int64, t1: Int64) -> TranscriptSegment {
//        var segment = TranscriptSegment()
//        segment.speech = speech
//        segment.startTimeStamp = t0
//        segment.endTimeStamp = t1
//
//        return segment
//    }
//
    
//    private func _transcript(wavFile: URL) -> [TranscriptSegment] {
//        //读取文件
//        let (data,numSamples) = readDataFromFile(wavFile: wavFile)
//        // 转换data为UnsafePointer<Float>
//        let samples = data.withUnsafeBytes { (bytes: UnsafeRawBufferPointer) -> UnsafePointer<Float> in
//            let floatPtr = bytes.bindMemory(to: Float.self)
//            return UnsafePointer<Float>(floatPtr.baseAddress!)
//        }
//
//        return _transcript(samples: samples,numSamples: numSamples)
//    }
    
    private func _transcript(samples: UnsafePointer<Float>,numSamples: Int) -> [TranscriptSegment] {
        guard let _whisper = _whisper else {
            return []
        }
        
        
        let result = _whisper.process(samples, sampleNum: Int32(numSamples))
        guard result else {
            return []
        }
        //获取段落
        let segmentNum = _whisper.getSegmentsNum()
        guard segmentNum > 0 else{
            return []
        }
        
        var transcriptResult: [TranscriptSegment] = []
        for index in 0..<segmentNum {
            var speech:String
            if let trySpeech = _whisper.getSpeechBySegmentIndex(index) {
                speech = trySpeech
            }else {
                speech = ""
            }
            
            
            let t0 = _whisper.getSpeechStartTime(bySegmentIndex: index)
            let t1 = _whisper.getSpeechEndTime(bySegmentIndex: index)
            
            let t0_ms = t0 * 10
            let t1_ms = t1 * 10
            
            let t0_index = Int(t0_ms * 16)
            let t1_index = Int(t1_ms * 16)
            
            
            
            let segmentData = TranscriptSegment(speech: speech, start: t0_index, end: t1_index)
            transcriptResult.append(segmentData)
        }
        return transcriptResult
    }
}



extension WhisperDiarization {
//    func transcript(wavFile: URL, callBack: TranscriptCallBack?) {
//        guard let callBack = callBack else {
//            return
//        }
//
//        guard _isTranscripting == false else {
//            callBack(nil, WhisperError.error(message: "transcripting"))
//            return
//        }
//        _isTranscripting = true
//        _queue.async { [weak self] in
//            let result = self?._transcript(wavFile: wavFile)
//            self?._isTranscripting = false
//            callBack(result, nil)
//        }
//    }
//
//
//    func transcript(samples: UnsafePointer<Float>,numSamples: Int, callBack: TranscriptCallBack?) {
//        guard let callBack = callBack else {
//            return
//        }
//
//        guard _isTranscripting == false else {
//            callBack(nil, WhisperError.error(message: "transcripting"))
//            return
//        }
//        _isTranscripting = true
//
//        _queue.async { [weak self] in
//            let result = self?._transcript(samples: samples, numSamples: numSamples)
//            self?._isTranscripting = false
//            callBack(result, nil)
//        }
//    }
    
    
    func transcriptSync(buffer: Data) -> [TranscriptSegment] {
        var result:[TranscriptSegment] = []
        buffer.withUnsafeBytes { (ptr: UnsafePointer<Float>) in
            let numSamples = buffer.count / MemoryLayout<Float>.stride
            result = self._transcript(samples: ptr, numSamples: numSamples)
        }
        
        return result
    }
    
    
//    func transcript(audioPCMBuffer: AVAudioPCMBuffer, callBack: TranscriptCallBack?) {
//        guard let callBack = callBack else {
//            return
//        }
//
//        guard _isTranscripting == false else {
//            callBack([], WhisperError.error(message: "transcripting"))
//            return
//        }
//        _isTranscripting = true
//
//        _cacheBuffer = audioPCMBuffer
//        _callBack = callBack
//
//
//        _queue.async { [weak self] in
//            guard let self = self else {
//                return
//            }
//
//            guard let cacheBuffer = self._cacheBuffer else {
//                self._cacheBuffer = nil
//                self._callBack = nil
//                self._isTranscripting = false
//                return
//            }
//
//            guard let cacheBuffer = self._cacheBuffer else {
//                return
//            }
//
//            let floatChannelData = cacheBuffer.floatChannelData!
//            let samples = UnsafePointer<Float>(floatChannelData[0])
//            let numSamples = Int(cacheBuffer.frameLength)
//
//            let result = self._transcript(samples: samples, numSamples: numSamples)
//
//            self._cacheBuffer = nil
//            self._callBack?(result, nil)
//            self._callBack = nil
//            self._isTranscripting = false
//        }
//    }
    
}
