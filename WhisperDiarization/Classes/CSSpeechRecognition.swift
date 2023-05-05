//
//  CSSpeechRecognition.swift
//  WhisperDiarization
//
//  Created by fuhao on 2023/4/27.
//

import Foundation
import AVFAudio
import Silero_VAD_for_iOS
import SpeakerEmbeddingForiOS


public struct TranscriptItem {
    public var label:Int
    public var speech: String
    public var startTimeStamp:Int64
    public var endTimeStamp:Int64
    public var features:[Float]
}

extension Data {
    func toFloatArray() -> [Float] {
        var floatArray = [Float](repeating: 0, count: count/MemoryLayout<Float>.stride)
        _ = floatArray.withUnsafeMutableBytes { mutableFloatBytes in
            self.copyBytes(to: mutableFloatBytes)
        }
        return floatArray
    }
}

public class CSSpeechRecognition {
    var whisper: WhisperDiarization?
    let _queue = DispatchQueue(label: "CSSpeechRecognition")
    let audioPreprocess = AudioPreprocess(maxItemCount: 2)
    var isRunning = true
    
    var vadFrameFixByte = MemoryLayout<Float>.size * 16000 * 29
//    var vadFrameFixByte = 511 * MemoryLayout<Float>.size
    var cahceFrameSize = 0
    var cacheAudioData = Data()
    
    let processFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
    var vad: VoiceActivityDetector?
    var pcmBuffers = [AVAudioPCMBuffer]()
    
    var featureExtarer: SpeakerEmbedding?
    
    
    var speechsCache: [TranscriptItem] = []
    
    public init() {
        _queue.async {
            self._preload()
            self._run()
        }
    }
    
    func _preload() {
        if whisper == nil {
            whisper = WhisperDiarization()
        }
        
        if vad == nil {
            vad = VoiceActivityDetector()
        }
        
        if featureExtarer == nil {
            featureExtarer = SpeakerEmbedding()
        }
    }
    
    func _isloaded() -> Bool{
        guard whisper != nil else {
            return false
        }
        
        guard vad != nil else {
            return false
        }
        
        guard featureExtarer != nil else {
            return false
        }
        
        return true
    }
    
    func divideIntoSegments(_ x: Int, step: Int) -> [(start: Int, count: Int)] {
        var result: [(start: Int, count: Int)] = []
        var remaining = x
        var start = 0
        
        while remaining > 0 && remaining >= step{
            result.append((start, step))
            remaining -= step
            start += step
        }
        
        return result
    }
    
    func _vadHandle() -> [VADTimeResult] {
        guard let vad = vad else {
            return []
        }
        
        guard cacheAudioData.count >= vadFrameFixByte else{
            return []
        }
        
        let chunkCount = cacheAudioData.count / (512 * MemoryLayout<Float>.size)
        let audioFrameCount = AVAudioFrameCount(chunkCount * 512)
        let audioFrameSize = Int(audioFrameCount) * MemoryLayout<Float>.size
        
        guard let pcmBuffer = AVAudioPCMBuffer(pcmFormat: processFormat, frameCapacity: audioFrameCount) else {
            fatalError("Unable to create PCM buffer")
        }
        pcmBuffer.frameLength = audioFrameCount
        
        let pcmFloatPointer: UnsafeMutablePointer<Float> = pcmBuffer.floatChannelData![0]
        let pcmRawPointer = pcmFloatPointer.withMemoryRebound(to: UInt8.self, capacity: audioFrameSize) {
            return UnsafeMutableRawPointer($0)
        }
        
        cacheAudioData.withUnsafeBytes { (bytes: UnsafeRawBufferPointer) in
            pcmRawPointer.copyMemory(from: bytes.baseAddress!, byteCount: audioFrameSize)
        }
        
        guard let result = vad.detectForTimeStemp(buffer: pcmBuffer) else {
            return []
        }
        return result
    }
    
    struct AudioSegment {
        var data: Data
        var start: Int
        var end: Int
    }
    
    struct AudioEmbedsSegment {
        var embeding: [Float]
        var start: Int
        var end: Int
    }
    
    struct AudioCombianEmbedsSegment {
        var embeding: [Float]
        var start: Int
        var end: Int
        var label: Int
    }
    
    
    func _windowed_embeds(featureExtarer: SpeakerEmbedding, startIndex: Int, signal: Data, fs: Int, window: Double = 0.5, period: Double = 0.3) -> [AudioEmbedsSegment] {
        let lenWindow = Int(window * Double(fs))
        let lenPeriod = Int(period * Double(fs))
        let lenSignal = signal.count / MemoryLayout<Float>.size

        // Get the windowed segments
        var segments: [[Int]] = []
        var start = 0
        while start + lenWindow < lenSignal {
            segments.append([start, start + lenWindow])
            start += lenPeriod
        }
        segments.append([start, lenSignal])
        
        
    
        var embeds: [AudioEmbedsSegment] = []
        for segment in segments {
            let i = segment[0]
            let j = segment[1]
            let signalSeg = signal.subdata(in: i*MemoryLayout<Float>.size..<(j)*MemoryLayout<Float>.size)
            let tempCheck = signalSeg.toFloatArray()
            
            guard let segEmbed = featureExtarer.extractFeature(data: signalSeg) else {
                continue
            }
            let segAudioEmbed = AudioEmbedsSegment(embeding: segEmbed, start: startIndex+i, end: startIndex+j)
            embeds.append(segAudioEmbed)
        }

        return embeds
    }
    
    func _featuresHandle(audioSegments: [AudioSegment]) -> [AudioEmbedsSegment]{
        guard let featureExtarer = featureExtarer else {
            return []
        }
        var allEmbeds: [AudioEmbedsSegment] = []
        
        audioSegments.forEach { audioSegment in
            let audioEmbedsSegments = _windowed_embeds(featureExtarer: featureExtarer,startIndex: audioSegment.start, signal: audioSegment.data, fs: 16000)
            allEmbeds.append(contentsOf: audioEmbedsSegments)
        }
        
        return allEmbeds
    }
    
    func _analyzeSpeaker(features: [[Float]]) -> (Int, [Int]) {
//        guard features.count > 0 else {
//            return (0, [])
//        }
//
//        guard features.count > 1 else {
//            return (1, [0])
//        }
//
//
//        let mormalFeatureDis = MLTools.pairwise_distances(features)
//
//        var speakersNumScore:[Float] = []
//        var speakersLabels:[[Int]] = []
//
//        for k in 2...5 {
//            let labels = MLTools.agglomerativeClustering(mormalFeatureDis, k)
//            let score = MLTools.silhouetteScore(mormalFeatureDis, labels)
//            speakersNumScore.append(score)
//            speakersLabels.append(labels)
//        }
//
//        let maxIndex = speakersNumScore.enumerated().max { $0.1 < $1.1 }?.0
//        guard let index = maxIndex else {
//            return (0, [])
//        }
//
//        return (index+2, speakersLabels[index])
        return (1, [Int](repeating: 0, count: features.count))
    }
    
    func _joinSegments(clusterLabels: [Int], segments: [AudioEmbedsSegment], tolerance: Int = 5) -> [AudioCombianEmbedsSegment] {
        assert(clusterLabels.count == segments.count)

        var newSegments = [AudioCombianEmbedsSegment]()
        guard let firstSeg = segments.first else {
            return newSegments
        }
        newSegments.append(AudioCombianEmbedsSegment(embeding: firstSeg.embeding, start: firstSeg.start, end: firstSeg.end, label: clusterLabels[0]))
        
        

        for i in 1..<segments.count {
            let l = clusterLabels[i]
            let seg = segments[i]
            let start = seg.start
            let end = seg.end
        
            var protoseg = AudioCombianEmbedsSegment(embeding: seg.embeding, start: seg.start, end: seg.end, label: l)

            if start <= newSegments.last!.end {
                // If segments overlap
                if l == newSegments.last!.label {
                    // If overlapping segment has same label
                    newSegments[newSegments.count - 1].end = end
                } else {
                    // If overlapping segment has diff label
                    // Resolve by setting new start to midpoint
                    // And setting last segment end to midpoint
                    let overlap = newSegments.last!.end - start
                    let midpoint = start + overlap / 2
                    newSegments[newSegments.count - 1].end = midpoint
                    protoseg.start = midpoint
                    newSegments.append(protoseg)
                }
            } else {
                // If there's no overlap just append
                newSegments.append(protoseg)
            }
        }

        return newSegments
    }
    
    
    func _joinSamespeakerSegments(_ segments: [AudioCombianEmbedsSegment], silenceTolerance: Double = 0.2) -> [AudioCombianEmbedsSegment] {
        var newSegments: [AudioCombianEmbedsSegment] = []
        guard let firstItem = segments.first else {
            return newSegments
        }
        newSegments.append(firstItem)
        let silenceToleranceSize = Int(silenceTolerance * 16000)

        for i in 1..<segments.count {
            let seg = segments[i]
            if seg.label == newSegments[newSegments.count - 1].label {
                if newSegments[newSegments.count - 1].end + silenceToleranceSize >= seg.start {
                    newSegments[newSegments.count - 1].end = seg.end
                } else {
                    newSegments.append(seg)
                }
            } else {
                newSegments.append(seg)
            }
        }
        return newSegments
    }

    
    func _run() {
        
        while isRunning {
            guard let audioBuffer = audioPreprocess.dequeue() else {
                continue
            }
            guard let whisper = whisper else {
                continue
            }
            let buffer = audioBuffer.buffer
            let segmentTimeStamp = audioBuffer.timeStamp
            
            let bufferByteSize = Int(buffer.frameLength) * MemoryLayout<Float>.size
            guard let floatChannel = buffer.floatChannelData else {
                continue
            }
            floatChannel[0].withMemoryRebound(to: UInt8.self, capacity: bufferByteSize) { pointer in
                cacheAudioData.append(pointer, count: bufferByteSize)
            }
        

            let vadResults = _vadHandle()
            
            let audioSegments = vadResults.map { result in
                
                let startIndex = result.start * MemoryLayout<Float>.size
                let endIndex = result.end * MemoryLayout<Float>.size
                
                print("startIndex:\(startIndex), endIndex: \(endIndex)")
                
                let audioSegment = cacheAudioData.subdata(in: startIndex..<endIndex)
                return AudioSegment(data: audioSegment, start: result.start, end: result.end)
            }
            
            let featuresSegments = _featuresHandle(audioSegments: audioSegments)
            
            let featuresX:[[Float]] = featuresSegments.map { segment in
                segment.embeding
            }

            let (speakerNum, speakerLabel) = _analyzeSpeaker(features: featuresX)
            
            //合并语句
            let joinAudioSegment = _joinSegments(clusterLabels: speakerLabel, segments: featuresSegments)
            let separateAudioSegment = _joinSamespeakerSegments(joinAudioSegment)
            
            //识别内容
            var speechs: [TranscriptItem] = []
            for segment in separateAudioSegment {
                let startIndex = segment.start * MemoryLayout<Float>.size
                let endIndex = segment.end * MemoryLayout<Float>.size
                let segmentData = cacheAudioData.subdata(in: startIndex..<endIndex)

                let speechTranscripts = whisper.transcriptSync(buffer: segmentData)
                speechTranscripts.forEach { transcriptSeg in
                    //TODO: 计算正确的时间戳
                    let speechItemStartTimeStamp = Int64((transcriptSeg.start + segment.start) / 16)
                    let speechItemEndTimeStamp = Int64((transcriptSeg.end + segment.start) / 16)
                    let startTimeStamp =  segmentTimeStamp + speechItemStartTimeStamp
                    let endTimeStamp = segmentTimeStamp + speechItemEndTimeStamp
                    
                    let transcript = TranscriptItem(label: segment.label, speech: transcriptSeg.speech,startTimeStamp: startTimeStamp, endTimeStamp: endTimeStamp, features: segment.embeding)
                    print("识别语音:\(transcript.speech), 时间: \(Date(timeIntervalSince1970: (TimeInterval(startTimeStamp) * 0.001)).description)")
                    speechs.append(transcript)
                }
            }
            speechsCache.append(contentsOf: speechs)
            
            //clean cache
            guard let lastAudioSegment = separateAudioSegment.last else {
                cacheAudioData.removeAll()
                return
            }
            let lastAduioSegEndBytes = min(bufferByteSize, lastAudioSegment.end * MemoryLayout<Float>.size)
            cacheAudioData.removeSubrange(0..<lastAduioSegEndBytes)
        }
    }
    
}


public extension CSSpeechRecognition {
    
    func pushAudioBuffer(buffer: AVAudioPCMBuffer, timeStamp: Int64) {
        guard _isloaded() else {
            return
        }
        audioPreprocess.enqueues(buffer, timeStamp: timeStamp)
    }
    
    func pullRecognition() -> [TranscriptItem]{
        let count = speechsCache.count
        guard count > 0 else {
            return []
        }
        
        let pullSpeechs = Array(speechsCache[0..<count])
        speechsCache.removeSubrange(0..<count)
        
        return pullSpeechs
    }
}
