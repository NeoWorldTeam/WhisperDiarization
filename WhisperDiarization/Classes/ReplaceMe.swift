import Foundation
import whisperxx

public typealias TranscriptCallBack = (_ result: TranscriptResult?, _ error: WhisperError?)->Void


public enum WhisperError: Error {
    case error(message: String)
}


public struct TranscriptSegment : Codable {
    public var label:Int
    public var speech:String
    public var startTimeStamp:Int64
    public var endTimeStamp:Int64
    public var features:[Float]
    public init() {
        label = 0
        speech = ""
        startTimeStamp = 0
        endTimeStamp = 0
        features = Array<Float>(repeating: 0.0, count: 192)
    }
    
    public init(label:Int,speech:String,startTimeStamp:Int64,endTimeStamp:Int64,features:[Float]) throws {
        self.label = label
        self.speech = speech
        self.startTimeStamp = startTimeStamp
        self.endTimeStamp = endTimeStamp
        self.features = features
    }
}

public struct TranscriptResult : Codable {
    public var speechs: [TranscriptSegment] = []
    init() {}
}

public class WhisperDiarization {
    let _queue: DispatchQueue
    var _whisper: WhisperWrapper?
    var _isTranscripting: Bool = false
    
    var _cacheBuffer: AVAudioPCMBuffer?
    
    var  _callBack: TranscriptCallBack?
    public init() {
        _queue = DispatchQueue(label: "WhisperDiarization")
        
        
        guard var associateBundleURL = Bundle.main.url(forResource: "Frameworks", withExtension: nil) else {
            return
        }

        associateBundleURL.appendPathComponent("WhisperDiarization")
        associateBundleURL.appendPathExtension("framework")
        
        guard let podBundle = Bundle(url: associateBundleURL) else {
            return
        }
        guard let associateBundleURL2 = podBundle.url(forResource: "WhisperDiarization", withExtension: "bundle") else {
            return
        }
        guard let podBundle2 = Bundle(url: associateBundleURL2) else {
            return
        }
        let modelPath = podBundle2.path(forResource: "ggml-tiny", ofType: "bin")
        _whisper = WhisperWrapper(model: modelPath)
    }
    

    

    
    private func readDataFromFile(wavFile: URL) -> (Data,Int){
        // 打开文件进行读取
        let file = try! FileHandle(forReadingFrom: wavFile)
        
        defer {
            // 关闭文件
            file.closeFile()
        }
        
        // 读取文件数据
        let data = file.readDataToEndOfFile()
        let numSamples = data.count / MemoryLayout<Float>.size
        
        return (data,numSamples)
    }
    
    private func generateTranscriptSeg(samples: UnsafePointer<Float>, speech: String, t0: Int64, t1: Int64) -> TranscriptSegment {
        var segment = TranscriptSegment()
        segment.speech = speech
        segment.startTimeStamp = t0
        segment.endTimeStamp = t1
        
        return segment
    }
    
    
    private func _transcript(wavFile: URL) -> TranscriptResult? {
        //读取文件
        let (data,numSamples) = readDataFromFile(wavFile: wavFile)
        // 转换data为UnsafePointer<Float>
        let samples = data.withUnsafeBytes { (bytes: UnsafeRawBufferPointer) -> UnsafePointer<Float> in
            let floatPtr = bytes.bindMemory(to: Float.self)
            return UnsafePointer<Float>(floatPtr.baseAddress!)
        }
        
        return _transcript(samples: samples,numSamples: numSamples)
    }
    
    private func _transcript(samples: UnsafePointer<Float>,numSamples: Int) -> TranscriptResult? {
        guard let _whisper = _whisper else {
            return nil
        }
        
        
        let result = _whisper.process(samples, sampleNum: Int32(numSamples))
        guard result else {
            return nil
        }
        //获取段落
        let segmentNum = _whisper.getSegmentsNum()
        guard segmentNum > 0 else{
            return nil
        }
        
        var transcriptResult = TranscriptResult()
        for index in 0..<segmentNum {
            var speech:String
            if let trySpeech = _whisper.getSpeechBySegmentIndex(index) {
                speech = trySpeech
            }else {
                speech = ""
            }
            
            
            let t0 = _whisper.getSpeechStartTime(bySegmentIndex: index)
            let t1 = _whisper.getSpeechEndTime(bySegmentIndex: index)
            
            let segmentData = generateTranscriptSeg(samples: samples, speech: speech, t0: t0, t1: t1)
            transcriptResult.speechs.append(segmentData)
        }
        return transcriptResult
    }
}



public extension WhisperDiarization {
    func transcript(wavFile: URL, callBack: TranscriptCallBack?) {
        guard let callBack = callBack else {
            return
        }
        
        guard _isTranscripting == false else {
            callBack(nil, WhisperError.error(message: "transcripting"))
            return
        }
        _isTranscripting = true
        _queue.async { [weak self] in
            let result = self?._transcript(wavFile: wavFile)
            self?._isTranscripting = false
            callBack(result, nil)
        }
    }
    
    
    func transcript(samples: UnsafePointer<Float>,numSamples: Int, callBack: TranscriptCallBack?) {
        guard let callBack = callBack else {
            return
        }
        
        guard _isTranscripting == false else {
            callBack(nil, WhisperError.error(message: "transcripting"))
            return
        }
        _isTranscripting = true
        
        _queue.async { [weak self] in
            let result = self?._transcript(samples: samples, numSamples: numSamples)
            self?._isTranscripting = false
            callBack(result, nil)
        }
    }
    
    
    func transcript(audioPCMBuffer: AVAudioPCMBuffer, callBack: TranscriptCallBack?) {
        guard let callBack = callBack else {
            return
        }
        
        guard _isTranscripting == false else {
            callBack(nil, WhisperError.error(message: "transcripting"))
            return
        }
        _isTranscripting = true
        
        _cacheBuffer = audioPCMBuffer
        _callBack = callBack

        
        _queue.async { [weak self] in
            guard let self = self else {
                return
            }
            
            guard let cacheBuffer = self._cacheBuffer else {
                self._cacheBuffer = nil
                self._callBack = nil
                self._isTranscripting = false
                return
            }
            
            guard let cacheBuffer = self._cacheBuffer else {
                return
            }
            
            let floatChannelData = cacheBuffer.floatChannelData!
            let samples = UnsafePointer<Float>(floatChannelData[0])
            let numSamples = Int(cacheBuffer.frameLength)
                  
            let result = self._transcript(samples: samples, numSamples: numSamples)
            
            self._cacheBuffer = nil
            self._callBack?(result, nil)
            self._callBack = nil
            self._isTranscripting = false
        }
    }
    
}
