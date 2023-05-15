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
    
    let _queue = DispatchQueue(label: "CSSpeechRecognition")
    let audioPreprocess = AudioPreprocess(maxItemCount: 2)
    var isRunning = true
    
    var vadMoudle: VADModule?
    var featureExtarer: SpeakerEmbedding?
    var speakerAnalyse: SpeakerAnalyseTempModule?
    var whisper: SpeechRecognizeModule?
    
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
            whisper = SpeechRecognizeModule()
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
    
    

    //分窗特征
    struct AudioWindowEmbedsSegment {
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
    
    
    func _windowed_embeds(featureExtarer: SpeakerEmbedding, sourceIndex: Int, signal: Data, fs: Int, window: Double = 0.9, period: Double = 0.3) -> [AudioWindowEmbedsSegment] {
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
        
        
    
        var embeds: [AudioWindowEmbedsSegment] = []
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
            let segAudioEmbed = AudioWindowEmbedsSegment(embeding: segEmbed, start: i, end: j, sourceIndex: sourceIndex )
            embeds.append(segAudioEmbed)
        }

        return embeds
    }
    
    func _featuresHandle(audioSegments: [AudioSegment]) -> [AudioWindowEmbedsSegment]{
        guard let featureExtarer = featureExtarer else {
            return []
        }
        var allEmbeds: [AudioWindowEmbedsSegment] = []
        
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
    
//    func _joinSegments(clusterLabels: [Int], segments: [AudioWindowEmbedsSegment], tolerance: Int = 5) -> [AudioCombianEmbedsSegment] {
//        assert(clusterLabels.count == segments.count)
//
//        var newSegments = [AudioCombianEmbedsSegment]()
//        guard let firstSeg = segments.first else {
//            return newSegments
//        }
//        newSegments.append(AudioCombianEmbedsSegment(embeding: firstSeg.embeding, start: firstSeg.start, end: firstSeg.end, label: clusterLabels[0]))
//
//
//
//        for i in 1..<segments.count {
//            let l = clusterLabels[i]
//            let seg = segments[i]
//            let start = seg.start
//            let end = seg.end
//
//            var protoseg = AudioCombianEmbedsSegment(embeding: seg.embeding, start: seg.start, end: seg.end, label: l)
//
//            if start <= newSegments.last!.end {
//                // If segments overlap
//                if l == newSegments.last!.label {
//                    // If overlapping segment has same label
//                    newSegments[newSegments.count - 1].end = end
//                } else {
//                    // If overlapping segment has diff label
//                    // Resolve by setting new start to midpoint
//                    // And setting last segment end to midpoint
//                    let overlap = newSegments.last!.end - start
//                    let midpoint = start + overlap / 2
//                    newSegments[newSegments.count - 1].end = midpoint
//                    protoseg.start = midpoint
//                    newSegments.append(protoseg)
//                }
//            } else {
//                // If there's no overlap just append
//                newSegments.append(protoseg)
//            }
//        }
//
//        return newSegments
//    }
//
//
//    func _joinSamespeakerSegments(_ segments: [AudioCombianEmbedsSegment], silenceTolerance: Double = 0.2) -> [AudioCombianEmbedsSegment] {
//        var newSegments: [AudioCombianEmbedsSegment] = []
//        guard let firstItem = segments.first else {
//            return newSegments
//        }
//        newSegments.append(firstItem)
//        let silenceToleranceSize = Int(silenceTolerance * 16000)
//
//        for i in 1..<segments.count {
//            let seg = segments[i]
//            if seg.label == newSegments[newSegments.count - 1].label {
//                if newSegments[newSegments.count - 1].end + silenceToleranceSize >= seg.start {
//                    newSegments[newSegments.count - 1].end = seg.end
//                } else {
//                    newSegments.append(seg)
//                }
//            } else {
//                newSegments.append(seg)
//            }
//        }
//        return newSegments
//    }

    func extractFeature(_ featureExtarer: SpeakerEmbedding, _ datas: inout [Data] ) -> [[Float]] {
        let transcriptFeature:[[Float]] = datas.map { data in
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
            let recognizeResult:[RecognizeSegment] = whisper.recognize(vadBuffers: vadResults)
            var trancriptRowData = recognizeResult.map({$0.data})
            let transcriptFeature:[[Float]] = extractFeature(featureExtarer, &trancriptRowData)
            
            guard !transcriptFeature.isEmpty else {
                return
            }
            
            //加入存在用户
            let (existSpeakerIndex,existSpeakerFeatures) = speakerAnalyse!.getTopSpeakerFeature(num: 2)
            if existSpeakerIndex.isEmpty || existSpeakerFeatures.count < speakerAnalyse!.fixHostFeatureCount {
                speakerAnalyse!.saveToHost(transcriptFeature)
                //不需要分析直接说话人==0
                
                let speechDatas = recognizeResult.enumerated().map { elem in
                    let index = elem.offset
                    let segment:RecognizeSegment = elem.element
                    let label = 0
                    let speech = segment.speech
                    let embeding = transcriptFeature[index]
                    let transcript = TranscriptItem(label: label, speech: speech, startTimeStamp: segment.startTimeStamp, endTimeStamp: segment.endTimeStamp, features: embeding)
                    print("识别语音:\(transcript.speech),说话人:\(label) 时间: \(Date(timeIntervalSince1970: (TimeInterval(segment.startTimeStamp) * 0.001)).description)")
                    return transcript
                }
                //加入缓存
                speechsCache.append(contentsOf: speechDatas)
                return
                
            }
            
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
                let audioSeg = recognizeResult[index]
                let speech = audioSeg.speech
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


