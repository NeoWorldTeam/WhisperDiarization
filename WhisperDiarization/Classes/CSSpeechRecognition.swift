//
//  CSSpeechRecognition.swift
//  WhisperDiarization
//
//  Created by fuhao on 2023/4/27.
//

import Foundation
import AVFAudio
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
    
    var vadMoudle: VADModule = VADModule()
    
    
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
    
//    func _vadHandle() -> [VADTimeResult] {
//        guard let vad = vad else {
//            return []
//        }
//
//        guard cacheAudioData.count >= vadFrameFixByte else{
//            return []
//        }
//
//        let chunkCount = cacheAudioData.count / (512 * MemoryLayout<Float>.size)
//        let audioFrameCount = AVAudioFrameCount(chunkCount * 512)
//        let audioFrameSize = Int(audioFrameCount) * MemoryLayout<Float>.size
//
//        guard let pcmBuffer = AVAudioPCMBuffer(pcmFormat: processFormat, frameCapacity: audioFrameCount) else {
//            fatalError("Unable to create PCM buffer")
//        }
//        pcmBuffer.frameLength = audioFrameCount
//
//        let pcmFloatPointer: UnsafeMutablePointer<Float> = pcmBuffer.floatChannelData![0]
//        let pcmRawPointer = pcmFloatPointer.withMemoryRebound(to: UInt8.self, capacity: audioFrameSize) {
//            return UnsafeMutableRawPointer($0)
//        }
//
//        cacheAudioData.withUnsafeBytes { (bytes: UnsafeRawBufferPointer) in
//            pcmRawPointer.copyMemory(from: bytes.baseAddress!, byteCount: audioFrameSize)
//        }
//
//        guard let result = vad.detectForTimeStemp(buffer: pcmBuffer) else {
//            return []
//        }
//        return result
//    }
    
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
        guard features.count > 0 else {
            return (0, [])
        }

        guard features.count > 1 else {
            return (1, [0])
        }


        let mormalFeatureDis = MLTools.pairwise_distances(features)
        var lastetScore:Float = 0
        var lastLabels:[Int] = []
        for k in 2...5 {
            let labels = MLTools.agglomerativeClustering(mormalFeatureDis, k)
            let score = MLTools.silhouetteScore(mormalFeatureDis, labels, k)
            if lastetScore > score {
                break
            }
            
            lastetScore = score
            lastLabels = labels
        }

        let speakersCount = Set(lastLabels).count
        return (speakersCount, lastLabels)
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
    
    func test_SaveToWav(data: Data, index: Int) {
        let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let fileURL = documentsDirectory.appendingPathComponent("audio_" + String(index) + ".wav")

        // 创建AVAudioFile
        let audioFile = try! AVAudioFile(forWriting: fileURL, settings: [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: 16000,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsFloatKey: true
        ])

        // 写入音频数据
        let audioBuffer = AVAudioPCMBuffer(pcmFormat: audioFile.processingFormat, frameCapacity: UInt32(data.count) / audioFile.processingFormat.streamDescription.pointee.mBytesPerFrame)!
        audioBuffer.frameLength = audioBuffer.frameCapacity
        let audioBufferData = audioBuffer.floatChannelData![0]
        audioBufferData.withMemoryRebound(to: UInt8.self, capacity: data.count) { pointer in
            data.copyBytes(to: pointer, count: data.count)
        }

        try! audioFile.write(from: audioBuffer)

        print("文件已经保存到：\(fileURL)")
    }

    
    func _run() {
        
        while isRunning {
            guard let audioBuffer = audioPreprocess.dequeue() else {
                continue
            }
            guard let whisper = whisper else {
                continue
            }
            
            let vadResults:[VADBuffer] = vadMoudle.checkAudio(buffer: audioBuffer.buffer, timeStamp: Int64(audioBuffer.timeStamp))
            guard vadResults.isEmpty == false else {
                continue
            }
            
            

            vadResults.forEach { vadBuffer in
                var speechTranscripts = whisper.transcriptSync(buffer: vadBuffer.buffer)
                //移除无效识别
                speechTranscripts = try! speechTranscripts.filter { transcripSeg in
                    guard !transcripSeg.speech.isEmpty else {
                        return false
                    }
//                    let pattern = #"^\s?\(\w+\)\s?$"#
                    let pattern = "\\([a-zA-Z0-9_!#]+\\)|\\[[a-zA-Z0-9_!#]+\\]"
                    let regex = try NSRegularExpression(pattern: pattern)
                    let range = NSRange(location: 0, length: transcripSeg.speech.utf16.count)
                    guard regex.firstMatch(in: transcripSeg.speech, options: [], range: range) == nil else {
                        return false
                    }
                    
                    return true
                }
                
                //识别和匹配
                var matchIndex = 0
                var matchSegment: [Int:Int] = [Int:Int]()
                
                var test_tttt_index = 200
                for sppechIndex in 0..<speechTranscripts.count {
                    let transcriptSeg = speechTranscripts[sppechIndex]
                    
                    
                    //检查分割数据准确性
                    let testData = vadBuffer.buffer.subdata(in: transcriptSeg.start * MemoryLayout<Float>.size..<transcriptSeg.end*MemoryLayout<Float>.size)
                    test_SaveToWav(data: testData, index: test_tttt_index)
                    test_tttt_index+=1
                    
                    
                    //中间值是否在范围内,每次从上一个定位点开始
                    for index in matchIndex..<vadBuffer.rangeTimes.count {
                        let vadRange = Int(vadBuffer.rangeTimes[index].sampleRange.start)..<Int(vadBuffer.rangeTimes[index].sampleRange.end)
                        let scriptRange = transcriptSeg.start..<transcriptSeg.end
                        
                        if max(vadRange.lowerBound, scriptRange.lowerBound) < min(vadRange.upperBound, scriptRange.upperBound), // 判断是否有交汇
                            min(vadRange.upperBound, scriptRange.upperBound) - max(vadRange.lowerBound, scriptRange.lowerBound) >= 512 { // 判断交汇数量是否为512
                            matchSegment[sppechIndex] = index
                            matchIndex = index
                            break
                        }
                    }
                }
                
                matchSegment.forEach { (key: Int, value: Int) in
                    let transcript = speechTranscripts[key]
                    let vadRange = vadBuffer.rangeTimes[value]
                    
                    let relativeStartIndex = Int64(transcript.start) - vadRange.sampleRange.start
                    let startTimeStemp = vadRange.realTimeStamp.start + relativeStartIndex / 16
                    let endTimeStemp = startTimeStemp + Int64(transcript.end - transcript.start) / 16
                    
                    let transcriptItem = TranscriptItem(label: 0, speech: transcript.speech, startTimeStamp: startTimeStemp, endTimeStamp: endTimeStemp, features: [Float](repeating: 0, count: 192))
                    speechsCache.append(transcriptItem)
                }
                
                
                

            }
            
            
            
//            let buffer = audioBuffer.buffer
//            let segmentTimeStamp = audioBuffer.timeStamp
//
//
//            vadMoudle.checkAudio(buffer: audioBuffer.buffer, timeStamp: audioBuffer.timeStamp)
//
//            let bufferByteSize = Int(buffer.frameLength) * MemoryLayout<Float>.size
//            guard let floatChannel = buffer.floatChannelData else {
//                continue
//            }
//            floatChannel[0].withMemoryRebound(to: UInt8.self, capacity: bufferByteSize) { pointer in
//                cacheAudioData.append(pointer, count: bufferByteSize)
//            }
//
//            // vad
//            let vadResults = _vadHandle()
//            let audioSegments = vadResults.map { result in
//                let startIndex = result.start * MemoryLayout<Float>.size
//                let endIndex = result.end * MemoryLayout<Float>.size
//
//                print("startIndex:\(startIndex), endIndex: \(endIndex)")
//
//                let audioSegment = cacheAudioData.subdata(in: startIndex..<endIndex)
//                return AudioSegment(data: audioSegment, start: result.start, end: result.end)
//            }
//
//            //transcript
//            for segment in separateAudioSegment {
//                let startIndex = segment.start * MemoryLayout<Float>.size
//                let endIndex = segment.end * MemoryLayout<Float>.size
//                let segmentData = cacheAudioData.subdata(in: startIndex..<endIndex)
//
//                test_SaveToWav(data: segmentData, index: test_index)
//                test_index+=1
//
//                let speechTranscripts = whisper.transcriptSync(buffer: segmentData)
//                speechTranscripts.forEach { transcriptSeg in
//                    //TODO: 计算正确的时间戳
//                    let speechItemStartTimeStamp = Int64((transcriptSeg.start + segment.start) / 16)
//                    let speechItemEndTimeStamp = Int64((transcriptSeg.end + segment.start) / 16)
//                    let startTimeStamp =  segmentTimeStamp + speechItemStartTimeStamp
//                    let endTimeStamp = segmentTimeStamp + speechItemEndTimeStamp
//
//                    let transcript = TranscriptItem(label: segment.label, speech: transcriptSeg.speech,startTimeStamp: startTimeStamp, endTimeStamp: endTimeStamp, features: segment.embeding)
//                    print("识别语音:\(transcript.speech), 时间: \(Date(timeIntervalSince1970: (TimeInterval(startTimeStamp) * 0.001)).description)")
//                    speechs.append(transcript)
//                }
//            }
//
//
//
////            var test_vad_Index = 100
////            audioSegments.forEach { segment in
////                test_SaveToWav(data: segment.data, index: test_vad_Index)
////                test_vad_Index += 1
////            }
//
//
//
//
//            let featuresSegments = _featuresHandle(audioSegments: audioSegments)
//
//            let featuresX:[[Float]] = featuresSegments.map { segment in
//                segment.embeding
//            }
//
//            let (speakerNum, speakerLabel) = _analyzeSpeaker(features: featuresX)
//
//            //合并语句
//            let joinAudioSegment = _joinSegments(clusterLabels: speakerLabel, segments: featuresSegments)
//            let separateAudioSegment = _joinSamespeakerSegments(joinAudioSegment)
//
//            //识别内容
//            var test_index = 0
//            var speechs: [TranscriptItem] = []
//            for segment in separateAudioSegment {
//                let startIndex = segment.start * MemoryLayout<Float>.size
//                let endIndex = segment.end * MemoryLayout<Float>.size
//                let segmentData = cacheAudioData.subdata(in: startIndex..<endIndex)
//
////                test_SaveToWav(data: segmentData, index: test_index)
////                test_index+=1
//
//                let speechTranscripts = whisper.transcriptSync(buffer: segmentData)
//                speechTranscripts.forEach { transcriptSeg in
//                    //TODO: 计算正确的时间戳
//                    let speechItemStartTimeStamp = Int64((transcriptSeg.start + segment.start) / 16)
//                    let speechItemEndTimeStamp = Int64((transcriptSeg.end + segment.start) / 16)
//                    let startTimeStamp =  segmentTimeStamp + speechItemStartTimeStamp
//                    let endTimeStamp = segmentTimeStamp + speechItemEndTimeStamp
//
//                    let transcript = TranscriptItem(label: segment.label, speech: transcriptSeg.speech,startTimeStamp: startTimeStamp, endTimeStamp: endTimeStamp, features: segment.embeding)
//                    print("识别语音:\(transcript.speech), 时间: \(Date(timeIntervalSince1970: (TimeInterval(startTimeStamp) * 0.001)).description)")
//                    speechs.append(transcript)
//                }
//            }
//            speechsCache.append(contentsOf: speechs)
//
//            //clean cache
//            guard let lastAudioSegment = separateAudioSegment.last else {
//                cacheAudioData.removeAll()
//                continue
//            }
//            let lastAduioSegEndBytes = min(bufferByteSize, lastAudioSegment.end * MemoryLayout<Float>.size)
//            cacheAudioData.removeSubrange(0..<lastAduioSegEndBytes)
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
    
    
//    ClusterA,ClusterB,0.56
//    ClusterA,ClusterC,0.32
//    ClusterA,ClusterD,0.51
//    ClusterA,ClusterE,0.12
//    ClusterB,ClusterC,0.56
//    ClusterB,ClusterD,0.32
//    ClusterB,ClusterE,0.64
//    ClusterC,ClusterD,0.21
//    ClusterC,ClusterE,0.18
//    ClusterE,ClusterD,0.51

    
//    func test() -> Bool {
//        let xxx = AggClusteringWrapper()
//        var testData:[[Float]] = [  [0, 0.56,   0.32,   0.51,   0.12],
//                                    [0.56, 0,   0.56,   0.32,   0.64],
//                                    [0.32, 0.56,   0,   0.21,   0.18],
//                                    [0.51, 0.32,   0.21,   0,   0.51],
//                                    [0.12, 0.64,   0.18,   0.51,   0],]
//        var flagTestData = Array<Float>(testData.joined())
//        var labels = [Int32](repeating: 0, count: 5)
//        
//        
//        flagTestData.withUnsafeMutableBufferPointer({ (cccc:inout UnsafeMutableBufferPointer<Float>) in
//            var dataPtr = cccc.baseAddress
//            labels.withUnsafeMutableBufferPointer { (dddd:inout UnsafeMutableBufferPointer<Int32>) in
//                var labelsPtr = dddd.baseAddress
//                xxx.agglomerativeClustering(dataPtr, row: 5, labels: labelsPtr)
//            }
//        })
//
//        
//        print(labels)
//        
//        
//        
//        
//        
//        return false
//    }
}


