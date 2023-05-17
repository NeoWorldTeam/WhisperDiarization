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
    var test_tttt_index = 1000
    
    
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
    
    func _analyzeSpeaker(features: [[Float]], k: Int = 2) -> (Int, [Int]) {
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
        let labels = MLTools.agglomerativeClustering(mormalFeatureDis, max(2, k), 5)
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
            
            
            var vadResults:[VADBuffer] = []
            DispatchQueue.main.sync {
                vadResults = vadMoudle.checkAudio(buffer: audioBuffer.buffer, timeStamp: Int64(audioBuffer.timeStamp))
            }
            
            guard !vadResults.isEmpty else {
                continue
            }

            print("checkAudio count:\(vadResults.count)")
//            vadResults.forEach { buffer in
//                whisper.test_SaveToWav(data: buffer.buffer, index: test_tttt_index)
//                test_tttt_index+=1
//            }
                        
            let recognizeResult:[RecognizeSegment] = whisper.recognize(vadBuffers: vadResults)
            print("recognizeResult count:\(recognizeResult.count)")
            var trancriptRowData = recognizeResult.map({$0.data})
            let transcriptFeature:[[Float]] = extractFeature(featureExtarer, &trancriptRowData)
            
            guard !transcriptFeature.isEmpty else {
                continue
            }
            
            //加入存在用户
            var (existSpeakerIndex,existSpeakerFeatures) = speakerAnalyse!.getTopSpeakerFeature(num: 3)
            existSpeakerIndex.enumerated().forEach { elem in
                print("label:\(elem.element), num:\(existSpeakerFeatures[elem.offset].count)")
            }
            
            //没有host情况
            let hostNeedFeature = speakerAnalyse!.fixHostFeatureCount - existSpeakerFeatures[0].count
            if existSpeakerFeatures.isEmpty || hostNeedFeature > 0 {
                speakerAnalyse!.saveToHost(transcriptFeature)
                
                 
                if hostNeedFeature >= transcriptFeature.count {
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
                    continue
                }
                
                //重新获取存在用户
                (existSpeakerIndex,existSpeakerFeatures) = speakerAnalyse!.getTopSpeakerFeature(num: 3)
            }
            
            //合并Feature
            var mergeFeatures:[[Float]] = []
            existSpeakerFeatures.forEach { features in
                mergeFeatures.append(contentsOf: features)
            }
            mergeFeatures.append(contentsOf: transcriptFeature)
            
            //分析
            var (speakerNum, speakerLabel) = _analyzeSpeaker(features: mergeFeatures, k: existSpeakerFeatures.count)
            print(speakerLabel)
            
            //获取存在用户的label
            var existSpeakerLabels: [[Int]] = []
            existSpeakerFeatures.forEach { features in
                var labels:[Int] = Array(speakerLabel[0..<features.count])
                speakerLabel.removeSubrange(0..<features.count)
                existSpeakerLabels.append(labels)
            }
            
            //分析label的概率
            print("========== 1 ================")
            existSpeakerLabels.enumerated().forEach { elem in
                let index: Int = elem.offset
                let labels: [Int] = elem.element
                print("label:\(existSpeakerIndex[index])  contain labels:\(labels)")
            }
            
            
            // 提取>1:1
            existSpeakerLabels.enumerated().forEach { elem in
                let index: Int = elem.offset
                let labels: [Int] = elem.element
                
                if existSpeakerLabels[index].last == -1 {
                    return
                }
                
                let counts: [Int: Int] = labels.reduce(into: [:]) { counts, element in
                    counts[element, default: 0] += 1
                }
                
                var highProbabilityLabels = Array<Int>(counts.filter { elem in
                    let a1 = Float(elem.value) / Float(labels.count)
                    let a2 = Float(1)/Float(counts.count)
                    return a1 >= a2
                }.keys)
                
                if highProbabilityLabels.isEmpty == false {
                    highProbabilityLabels.append(-1)
                    existSpeakerLabels[index] = highProbabilityLabels
                    return
                }
            }
            
            print("========== 2 ================")
            existSpeakerLabels.enumerated().forEach { elem in
                let index: Int = elem.offset
                let labels: [Int] = elem.element
                print("label:\(existSpeakerIndex[index])  contain labels:\(labels)")
            }
            
            
            //去除掉相同的高概率
            var useLabels:[Int] = []
            
            existSpeakerLabels = existSpeakerLabels.map({ labels in
                guard labels.contains(-1) else {
                    return []
                }
                
                let filtterLabels = labels.filter { label in
                    !useLabels.contains(label)
                }
                
                useLabels.append(contentsOf: filtterLabels)
                return filtterLabels
            })
            
            print("========== 3 ================")
            existSpeakerLabels.enumerated().forEach { elem in
                let index: Int = elem.offset
                let labels: [Int] = elem.element
                print("label:\(existSpeakerIndex[index])  contain labels:\(labels)")
            }

            //提取所有已知用户
            //替换
            speakerLabel = speakerLabel.map { (number) -> Int in
                for (index, labels) in existSpeakerLabels.enumerated() {
                    if labels.contains(number) {
                        return existSpeakerIndex[index]
                    }
                }
                
                return number + 101
            }
            
            //为不存在的用户创建新的id并替换
            let unRecognizeSpeakerLabels = speakerLabel.filter { label in
                label > 100
            }
            
            var recordNewLabel = existSpeakerIndex.max()!
            Set(unRecognizeSpeakerLabels).forEach { unRecogizeLabel in
                var newLabel = 0
                speakerLabel = speakerLabel.map({ label in
                    if label == unRecogizeLabel {
                        if newLabel == 0 {
                            recordNewLabel += 1
                            newLabel = recordNewLabel
                        }
                        return newLabel
                    }else {
                        return label
                    }
                })
            }
            
            //通过更新好的id生成数据
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

            //更新
            speechDatas.forEach { item in
                speakerAnalyse?.updateSpeaker(index: item.label, feature: item.features)
            }

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


