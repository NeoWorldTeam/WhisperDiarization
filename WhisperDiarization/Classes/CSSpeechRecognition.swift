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


struct VADAndTranscriptMatchSegment {
    var vadIndex: Int
    var speechIndex: [Int]
    
    init(vadIndex: Int) {
        self.vadIndex = vadIndex
        self.speechIndex = []
    }
}

public class CSSpeechRecognition {
    var whisper: WhisperDiarization?
    let _queue = DispatchQueue(label: "CSSpeechRecognition")
    let audioPreprocess = AudioPreprocess(maxItemCount: 2)
    var isRunning = true
    
    var vadMoudle: VADModule?
    
    
    var vadFrameFixByte = MemoryLayout<Float>.size * 16000 * 29
//    var vadFrameFixByte = 511 * MemoryLayout<Float>.size
    var cahceFrameSize = 0
    var cacheAudioData = Data()
    
//    let processFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
//    var vad: VoiceActivityDetector?
    var pcmBuffers = [AVAudioPCMBuffer]()
    
    var featureExtarer: SpeakerEmbedding?
    var speakerAnalyse: SpeakerAnalyseTempModule?
    
    var speechsCache: [TranscriptItem] = []
//    var test_tttt_index = 200
    
    
    public init() {
        _queue.async {
            self._preload()
            self._run()
        }
    }
    
    func _preload() {
        
        if speakerAnalyse == nil {
            speakerAnalyse = SpeakerAnalyseTempModule()
            speakerAnalyse?.preload()
        }
        
        
        if whisper == nil {
            whisper = WhisperDiarization()
        }
        
        if vadMoudle == nil {
            vadMoudle = VADModule()
        }
        
        if featureExtarer == nil {
            featureExtarer = SpeakerEmbedding()
        }
    }
    
    func _isloaded() -> Bool{
        guard whisper != nil else {
            return false
        }
        
        guard vadMoudle != nil else {
            return false
        }
        
        guard featureExtarer != nil else {
            return false
        }
        
        return true
    }
    
    
    struct AudioSegment {
        var data: Data
        var start: Int
        var end: Int
        var startTimeStamp: Int64
        var endTimeStamp: Int64
    }
    
    struct AudioEmbedsSegment {
        var embeding: [Float]
        var start: Int
        var end: Int
        var sourceIndex: Int
    }
    
    struct AudioCombianEmbedsSegment {
        var embeding: [Float]
        var start: Int
        var end: Int
        var label: Int
    }
    
    
    func _windowed_embeds(featureExtarer: SpeakerEmbedding, sourceIndex: Int, signal: Data, fs: Int, window: Double = 0.9, period: Double = 0.3) -> [AudioEmbedsSegment] {
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
            let startIndex = i * MemoryLayout<Float>.size
            let endIndex = j * MemoryLayout<Float>.size
            let signalSeg = signal.subdata(in: startIndex..<endIndex)
//            let tempCheck = signalSeg.toFloatArray()
            
            guard let segEmbed = featureExtarer.extractFeature(data: signalSeg) else {
                continue
            }
            let segAudioEmbed = AudioEmbedsSegment(embeding: segEmbed, start: i, end: j, sourceIndex: sourceIndex )
            embeds.append(segAudioEmbed)
        }

        return embeds
    }
    
    func _featuresHandle(audioSegments: [AudioSegment]) -> [AudioEmbedsSegment]{
        guard let featureExtarer = featureExtarer else {
            return []
        }
        var allEmbeds: [AudioEmbedsSegment] = []
        
        for (index, audioSegment) in audioSegments.enumerated() {
            let audioEmbedsSegments = _windowed_embeds(featureExtarer: featureExtarer,sourceIndex: index, signal: audioSegment.data, fs: 16000)
            print("audioEmbedsSegments count: \(audioEmbedsSegments.count)")
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
//        let l2FeatureDis = MLTools.pairwise_distances(mormalFeatureDis)
//        print(mormalFeatureDis)
//        var lastetScore:Float = 0
//        var lastLabels:[Int] = []
        let labels = MLTools.agglomerativeClustering(mormalFeatureDis, 5)
//        for k in 2...5 {
//            let labels = MLTools.agglomerativeClustering(mormalFeatureDis, k)
//            let score = MLTools.silhouetteScore(l2FeatureDis, labels, k)
//            if lastetScore > score {
//                break
//            }
//
//            lastetScore = score
//            lastLabels = labels
//        }

        let speakersCount = Set(labels).count
        return (speakersCount, labels)
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

    func fillterSpeechTranscript(_ transcripts: inout [TranscriptSegment]) -> [TranscriptSegment] {
        let speechTranscripts = try! transcripts.filter { transcripSeg in
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
        return speechTranscripts
    }
    
    func matchVADAndTranscript(_ transcripts: inout [TranscriptSegment], _ vadBuffer: VADBuffer) -> [VADAndTranscriptMatchSegment] {
        var matchIndex = 0
        var matchSegments: [VADAndTranscriptMatchSegment] = []
        
        
        for (sppechIndex, transcriptSeg) in transcripts.enumerated() {
            //检查分割数据准确性
//                    let testData = vadBuffer.buffer.subdata(in: transcriptSeg.start * MemoryLayout<Float>.size..<transcriptSeg.end*MemoryLayout<Float>.size)
//                    test_SaveToWav(data: testData, index: test_tttt_index)
//                    test_tttt_index+=1
            
            //中间值是否在范围内,每次从上一个定位点开始
            for index in matchIndex..<vadBuffer.rangeTimes.count {
                let vadRange = Int(vadBuffer.rangeTimes[index].sampleRange.start)..<Int(vadBuffer.rangeTimes[index].sampleRange.end)
                let scriptRange = transcriptSeg.start..<transcriptSeg.end
                
                if max(vadRange.lowerBound, scriptRange.lowerBound) < min(vadRange.upperBound, scriptRange.upperBound), // 判断是否有交汇
                    min(vadRange.upperBound, scriptRange.upperBound) - max(vadRange.lowerBound, scriptRange.lowerBound) >= 512 { // 判断交汇数量是否为512
                    
                    matchIndex = index
                    if let matchItemIndex = matchSegments.firstIndex(where: {$0.vadIndex == index}) {
                        matchSegments[matchItemIndex].speechIndex.append(sppechIndex)
                    }else {
                        var matchItem = VADAndTranscriptMatchSegment(vadIndex: index)
                        matchItem.speechIndex.append(sppechIndex)
                        matchSegments.append(matchItem)
                    }
                    break
                }
            }
        }
        return matchSegments
    }
    
    func extractAudioRaw(_ transcripts: inout [TranscriptSegment], _ vadBuffer: VADBuffer, _ matchSegments: inout [VADAndTranscriptMatchSegment] ) -> [AudioSegment] {
        let trancriptAudioSegments:[AudioSegment] = transcripts.enumerated().map { (index, seg) in

            let matchSegment = matchSegments.first(where: {$0.speechIndex.contains(where: {$0 == index})})!
            let vadRange = vadBuffer.rangeTimes[matchSegment.vadIndex]
            let caculateStart = max(seg.start, Int(vadRange.sampleRange.start)) * MemoryLayout<Float>.size
            let caculateEnd = min(seg.end, Int(vadRange.sampleRange.end)) * MemoryLayout<Float>.size

            let segData = vadBuffer.buffer.subdata(in: caculateStart..<caculateEnd)
            
            let startSampleIndex = 0
            let endSampleIndex = (caculateEnd - caculateStart) / MemoryLayout<Float>.size
            
            
            let caculateStartIndex = caculateStart / MemoryLayout<Float>.size
            let startTimeStamp:Int64 = vadRange.realTimeStamp.start + ((Int64(caculateStartIndex) - vadRange.sampleRange.start) / 16)
            let endTimeStamp:Int64 = startTimeStamp + Int64(endSampleIndex / 16)
            
            return AudioSegment(data: segData, start: startSampleIndex, end: endSampleIndex, startTimeStamp: startTimeStamp, endTimeStamp: endTimeStamp)
        }
        
        return trancriptAudioSegments
    }
    
    
    
    func extractFeature(_ featureExtarer: SpeakerEmbedding, _ datas: inout [Data] ) -> [[Float]] {
//                test_tttt_index = 300
        let transcriptFeature:[[Float]] = datas.map { data in
//                    test_SaveToWav(data: data, index: test_tttt_index)
//                    test_tttt_index += 1

            guard let feature = featureExtarer.extractFeature(data: data) else {
                return [Float](repeating: 0, count: 192)
            }
            return feature
        }
        return transcriptFeature
    }
    
    func _run() {
        
        while isRunning {
            guard let audioBuffer = audioPreprocess.dequeue() else {
                continue
            }
            guard let whisper = whisper else {
                continue
            }
            guard let featureExtarer = featureExtarer else {
                continue
            }
            guard let vadMoudle = vadMoudle else {
                continue
            }
            
            
            let vadResults:[VADBuffer] = vadMoudle.checkAudio(buffer: audioBuffer.buffer, timeStamp: Int64(audioBuffer.timeStamp))
            guard vadResults.isEmpty == false else {
                continue
            }
            
            

            vadResults.forEach { vadBuffer in
//                test_SaveToWav(data: vadBuffer.buffer, index: 1000)
                
                var speechTranscripts = whisper.transcriptSync(buffer: vadBuffer.buffer)
                print("before:\(speechTranscripts)")
                speechTranscripts = fillterSpeechTranscript(&speechTranscripts)
                print("after:\(speechTranscripts)")
                var matchSegments:[VADAndTranscriptMatchSegment] = matchVADAndTranscript(&speechTranscripts, vadBuffer)
                print(matchSegments)
//                //第二次增强识别
//                let needEnhanceRecognizeMatch = matchSegments.filter { seg in
//                    seg.speechIndex.count > 0
//                }
//
//                let singleRecongizeMatch = matchSegments.filter { seg in
//                    seg.speechIndex.count == 0
//                }
//
//
//                let enhanceRecognizeData = Date()
//                needEnhanceRecognizeMatch.forEach { seg in
//                    let vadRange = vadBuffer.rangeTimes[seg.vadIndex]
//
//                }
//
                
                let trancriptAudioSeg:[AudioSegment] = extractAudioRaw(&speechTranscripts, vadBuffer, &matchSegments)
                var trancriptRowData = trancriptAudioSeg.map({$0.data})
                let transcriptFeature:[[Float]] = extractFeature(featureExtarer, &trancriptRowData)
                
                guard !transcriptFeature.isEmpty else {
                    return
                }
                
                //加入存在用户
                let (existSpeakerIndex,existSpeakerFeatures) = speakerAnalyse!.getTopSpeakerFeature(num: 2)
                var mergeFeatures:[[Float]] = []
                existSpeakerFeatures.forEach { features in
                    mergeFeatures.append(contentsOf: features)
                }
                mergeFeatures.append(contentsOf: transcriptFeature)

                
                
                var (speakerNum, speakerLabel) = _analyzeSpeaker(features: mergeFeatures)
                
                //存在用户remark
                var existSpeakerLabels: [[Int]] = []
                if existSpeakerFeatures.isEmpty == false {
                    existSpeakerFeatures.forEach { features in
                        let labels:[Int] = Array(speakerLabel[0..<features.count])
                        speakerLabel.removeSubrange(0..<features.count)
                        existSpeakerLabels.append(labels)
                    }
                    
                    for (index, labels) in existSpeakerLabels.enumerated() {
                        //最大可能
                        let mostFrequent = labels.reduce(into: [:]) { counts, number in
                            counts[number, default: 0] += 1
                        }
                        .max { $0.value < $1.value }?.key
                        
                        let existSpeakerLabel = existSpeakerIndex[index]
                        //替换
                        speakerLabel = speakerLabel.map { (number) -> Int in
                            if number == mostFrequent {
                                return existSpeakerLabel
                            } else {
                                return number + 101
                            }
                        }
                    }
                }
                
                let unRecognizeSpeakerLabels = speakerLabel.filter { label in
                    label > 100
                }
                
                Set(unRecognizeSpeakerLabels).forEach { unRecogizeLabel in
                    lazy var newLabel = self.speakerAnalyse!.generateNewIndex()
                    speakerLabel = speakerLabel.map({ label in
                        if label == unRecogizeLabel {
                            return newLabel
                        }else {
                            return label
                        }
                    })
                }
                
                let speechDatas = speakerLabel.enumerated().map { elem in
                    let label = elem.element
                    let index = elem.offset
                    let speech = speechTranscripts[index].speech
                    let audioSeg = trancriptAudioSeg[index]
                    let embeding = transcriptFeature[index]
                    let transcript = TranscriptItem(label: label, speech: speech, startTimeStamp: audioSeg.startTimeStamp, endTimeStamp: audioSeg.endTimeStamp, features: embeding)
                    print("识别语音:\(transcript.speech),说话人:\(label) 时间: \(Date(timeIntervalSince1970: (TimeInterval(audioSeg.startTimeStamp) * 0.001)).description)")
                    return transcript
                }
 
                var store_featurePair:[(Int,[Float])] = []
                var store_wordNumPair:[(Int,Int)] = []
                speechDatas.forEach { item in
                    guard let f_p = store_featurePair.firstIndex(where: {$0.0 == item.label}) else{
                        store_featurePair.append((item.label, item.features))
                        return
                    }
                }
                
                speechDatas.forEach { item in
                    guard let w_p = store_wordNumPair.firstIndex(where: {$0.0 == item.label}) else{
                        store_wordNumPair.append((item.label, item.speech.count))
                        return
                    }
                    
                    store_wordNumPair[w_p].1 += item.speech.count
                }
                
                store_featurePair.forEach { (label: Int, feature: [Float]) in
                    guard let wordPair = store_wordNumPair.first(where: {$0.0 == label}) else {
                        return
                    }
                    speakerAnalyse?.updateSpeaker(index: label, feature: feature, word: wordPair.1)
                }
                speakerAnalyse?.store()

                //加入缓存
                speechsCache.append(contentsOf: speechDatas)
            }

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


