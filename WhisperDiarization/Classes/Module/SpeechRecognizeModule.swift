//
//  SpeechRecognizeModule.swift
//  WhisperDiarization
//
//  Created by fuhao on 2023/5/12.
//

import Foundation
import AVFAudio


struct VADAndTranscriptMatchSegment {
    var vadIndex: Int
    var speechIndex: [(Int,Int)]
    var speechs: [String]
    
    init(vadIndex: Int) {
        self.vadIndex = vadIndex
        self.speechIndex = []
        self.speechs = []
    }
}

struct AudioSegment {
    var data: Data
    var start: Int
    var end: Int
    var startTimeStamp: Int64
    var endTimeStamp: Int64
}


struct RecognizeSegment {
    var id: Int
    var speech: String
    var data: Data
    var startIndex: Int
    var endIndex: Int
    var startTimeStamp: Int64
    var endTimeStamp: Int64
    var shard: Bool
}

class SpeechRecognizeModule {
    let whisper: WhisperDiarization
    let clampFixBytes = 30 * 16000 * MemoryLayout<Float>.size
    init() {
        whisper = WhisperDiarization()
    }
    
    //    func extractAudioRaw(_ transcripts: inout [TranscriptSegment], _ vadBuffer: VADBuffer, _ matchSegments: inout [VADAndTranscriptMatchSegment] ) -> [AudioSegment] {
    //        let trancriptAudioSegments:[AudioSegment] = transcripts.enumerated().map { (index, seg) in
    //
    //            let matchSegment = matchSegments.first(where: {$0.speechIndex.contains(where: {$0 == index})})!
    //            let vadRange = vadBuffer.rangeTimes[matchSegment.vadIndex]
    //            let caculateStart = max(seg.start, Int(vadRange.sampleRange.start)) * MemoryLayout<Float>.size
    //            let caculateEnd = min(seg.end, Int(vadRange.sampleRange.end)) * MemoryLayout<Float>.size
    //
    //            let segData = vadBuffer.buffer.subdata(in: caculateStart..<caculateEnd)
    //
    //            let startSampleIndex = 0
    //            let endSampleIndex = (caculateEnd - caculateStart) / MemoryLayout<Float>.size
    //
    //
    //            let caculateStartIndex = caculateStart / MemoryLayout<Float>.size
    //            let startTimeStamp:Int64 = vadRange.realTimeStamp.start + ((Int64(caculateStartIndex) - vadRange.sampleRange.start) / 16)
    //            let endTimeStamp:Int64 = startTimeStamp + Int64(endSampleIndex / 16)
    //
    //            return AudioSegment(data: segData, start: startSampleIndex, end: endSampleIndex, startTimeStamp: startTimeStamp, endTimeStamp: endTimeStamp)
    //        }
    //
    //        return trancriptAudioSegments
    //    }
    
    
    
//    func findStartPosInWhichRange(statrIndex: Int, position: Int, rangeTimes: [VADRange]) -> (index: Int, volume: Int) {
//        for index in statrIndex..<rangeTimes.count {
//            let myRange = rangeTimes[index].sampleRange.start..<rangeTimes[index].sampleRange.end
//            if myRange.contains(Int64(position)) {
//                let volume = Int(rangeTimes[index].sampleRange.end - Int64(position))
//                return (index,volume)
//            }
//        }
//
//        return (-1,0)
//    }
    
    func findStartPosInWhichRange(statrIndex: Int, startPos: Int, endPos: Int, rangeTimes: [VADRange]) -> (index: Int, volume: Int) {
        for index in statrIndex..<rangeTimes.count {
            let rangeRealStart = Int(rangeTimes[index].sampleRange.start)
            let rangeRealEnd = Int(rangeTimes[index].sampleRange.end)
            var rangeStart = rangeRealStart
            var rangeEnd = rangeRealEnd
            
            if index != 0 {
                rangeStart = Int(rangeTimes[index - 1].sampleRange.end)
            }
            
            if index != (rangeTimes.count - 1) {
                rangeEnd = Int(rangeTimes[index + 1].sampleRange.start)
            }
            
            
            let myRange = rangeStart..<rangeEnd
            if myRange.contains(startPos) {
                let volume = min(Int(rangeRealEnd), endPos) - startPos
                return (index,volume)
            }
        }
        
        return (-1,0)
    }
    
    func findEndPosInWhichRange(statrIndex: Int, startPos: Int, endPos: Int, rangeTimes: [VADRange]) -> (index: Int, volume: Int) {
        var endPosByLimit = endPos
        if endPosByLimit >= rangeTimes[rangeTimes.count-1].sampleRange.end {
            endPosByLimit = Int(rangeTimes[rangeTimes.count-1].sampleRange.end - 1)
        }
        for index in statrIndex..<rangeTimes.count {
            let rangeRealStart = Int(rangeTimes[index].sampleRange.start)
            let rangeRealEnd = Int(rangeTimes[index].sampleRange.end)
            var rangeStart = rangeRealStart
            var rangeEnd = rangeRealEnd
            
            if index != 0 {
                rangeStart = Int(rangeTimes[index - 1].sampleRange.end)
            }
            
            if index != (rangeTimes.count - 1) {
                rangeEnd = Int(rangeTimes[index + 1].sampleRange.start)
            }
            

            
            
            let myRange = rangeStart..<rangeEnd
            if myRange.contains(endPosByLimit) {
                let volume = endPosByLimit - max(Int(rangeRealStart), startPos)
                return (index,volume)
            }
        }
        
        return (-1,0)
    }

    
    func overlapRange(a: (start: Int, end: Int), b: (start: Int, end: Int)) -> (start: Int, end: Int)? {
        // 计算两个区间的重合部分
        let start = max(a.start, b.start)
        let end = min(a.end, b.end)
        // 如果没有重合部分，则返回nil
        if start >= end {
            return nil
        }
        // 计算重合区间的上下区间
        let upper = min(a.end, b.end)
        let lower = max(a.start, b.start)
        // 返回重合区间的上下区间
        return (lower, upper)
    }
    
    func matchTranscriptLocationInBufferByVAD(_ transcripts: inout [TranscriptSegment], _ rangeTimes: [VADRange]) -> [VADAndTranscriptMatchSegment] {
        var matchIndex = 0
        var matchSegments: [VADAndTranscriptMatchSegment] = []
        
        
        var tempIndex = 0
//        var volume_1 = 0
//        var volume_2 = 0
        for (sppechIndex, transcriptSeg) in transcripts.enumerated() {
            //检查分割数据准确性
//                    let testData = vadBuffer.buffer.subdata(in: transcriptSeg.start * MemoryLayout<Float>.size..<transcriptSeg.end*MemoryLayout<Float>.size)
//                    test_SaveToWav(data: testData, index: test_tttt_index)
//                    test_tttt_index+=1
            
            var maxOverLapSize = 0
            let iterStartIndex = tempIndex
            var speechIndexPair:(Int,Int) = (0,0)
            for index in iterStartIndex..<rangeTimes.count {
                let meeasureRange = rangeTimes[index]
                guard let range = overlapRange(a: (transcriptSeg.start,transcriptSeg.end), b: (Int(meeasureRange.sampleRange.start),Int(meeasureRange.sampleRange.end))) else {
                    guard maxOverLapSize == 0 else {
                        break
                    }
                    continue
                }
                
                if (range.end - range.start) > maxOverLapSize {
                    maxOverLapSize = range.end - range.start
                    tempIndex = index
                    speechIndexPair = range
                }
            }
            
            if let matchItemIndex = matchSegments.firstIndex(where: {$0.vadIndex == tempIndex}) {
                matchSegments[matchItemIndex].speechIndex.append(speechIndexPair)
                matchSegments[matchItemIndex].speechs.append(transcriptSeg.speech)
            }else {
                var matchItem = VADAndTranscriptMatchSegment(vadIndex: tempIndex)
                matchItem.speechIndex.append(speechIndexPair)
                matchItem.speechs.append(transcriptSeg.speech)
                matchSegments.append(matchItem)
            }
            
            
//            for index in matchIndex..<rangeTimes.count {
//
//
//
//
//                let clampStart = max(Int(rangeTimes[index].sampleRange.start),transcriptSeg.start)
//                let clampEnd = min(Int(rangeTimes[index].sampleRange.end), transcriptSeg.end)
//                let speechIndexPair = (clampStart, clampEnd)
//
//                if clampStart < clampEnd,
//                   clampEnd - clampStart >= 512 {
//
//                    matchIndex = index
//                    if let matchItemIndex = matchSegments.firstIndex(where: {$0.vadIndex == index}) {
//                        matchSegments[matchItemIndex].speechIndex.append(speechIndexPair)
//                        matchSegments[matchItemIndex].speechs.append(transcriptSeg.speech)
//                    }else {
//                        var matchItem = VADAndTranscriptMatchSegment(vadIndex: index)
//                        matchItem.speechIndex.append(speechIndexPair)
//                        matchItem.speechs.append(transcriptSeg.speech)
//                        matchSegments.append(matchItem)
//                    }
//                    break
//                }
//            }
        }
        return matchSegments
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
    
    func caculateSegment(matchSegments: [VADAndTranscriptMatchSegment], vadBuffer: VADBuffer) -> [RecognizeSegment]{

        var ressss: [RecognizeSegment] = []
        var id_mark = 0
        
        for (index, seg) in matchSegments.enumerated() {
            let vadRange = vadBuffer.rangeTimes[index]
            let shard = seg.speechIndex.count > 1
            
            
            seg.speechIndex.enumerated().forEach { elemet in
                let segIndex = elemet.offset
                let start: Int = elemet.element.0
                let end: Int = elemet.element.1
                
                let startBytes = start * MemoryLayout<Float>.size
                let endBytes = end * MemoryLayout<Float>.size
                let buffer = vadBuffer.buffer.subdata(in: startBytes..<endBytes)
                
        
                let caculateStartIndex = start
                let startTimeStamp:Int64 = vadRange.realTimeStamp.start + ((Int64(caculateStartIndex) - vadRange.sampleRange.start) / 16)
                let endTimeStamp:Int64 = startTimeStamp + Int64(end / 16)
                
                let recogSeg = RecognizeSegment(id: id_mark, speech: seg.speechs[segIndex],data: buffer, startIndex: 0, endIndex: end - start, startTimeStamp: startTimeStamp, endTimeStamp: endTimeStamp, shard: shard)
                
                //移除低于100毫秒的数据
                //移除低于100毫秒的数据
                if recogSeg.data.count > 6400 {
                    id_mark += 1
                    ressss.append(recogSeg)
                }
                
                
            }
            


        }
        return ressss
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
    
    func recognize(vadBuffers: [VADBuffer]) -> [RecognizeSegment] {
        
//        var sasasasas = 100
//
//        vadBuffers.forEach { (vvvv: VADBuffer) in
//            print("buffer: \(vvvv.buffer.count)")
//            test_SaveToWav(data: vvvv.buffer, index: sasasasas)
//            sasasasas+=1
//            vvvv.rangeTimes.forEach { range in
//                print("range start: \(range.sampleRange.start) end:\(range.sampleRange.end) ")
//                let dadfdad = vvvv.buffer.subdata(in: Int(range.sampleRange.start << 2)..<Int(range.sampleRange.end << 2))
//                test_SaveToWav(data: dadfdad, index: sasasasas)
//                sasasasas+=1
//            }
//        }
        
        
        
        //1.找到range
//        sasasasas = 200
        let matchSegmentsList = vadBuffers.map { vadBuffer in
            var speechTranscripts:[TranscriptSegment] = whisper.transcriptSync(buffer: vadBuffer.buffer)
            speechTranscripts = fillterSpeechTranscript(&speechTranscripts)
            
//            speechTranscripts.forEach { elemet in
//                print("speechTranscripts start: \(elemet.start) end:\(elemet.end) speech:\(elemet.speech)")
//                let dadfdad = vadBuffer.buffer.subdata(in: (elemet.start<<2)..<(elemet.end<<2))
//                test_SaveToWav(data: dadfdad, index: sasasasas)
//                sasasasas+=1
//            }

            return matchTranscriptLocationInBufferByVAD(&speechTranscripts, vadBuffer.rangeTimes)
        }
        
        //第一次提取
        var recognizeSegments:[RecognizeSegment] = []
        for (index, matchSegments) in matchSegmentsList.enumerated() {
            let buffer = vadBuffers[index]
            
            let recognizeSegs = caculateSegment(matchSegments: matchSegments, vadBuffer: buffer)
            
            recognizeSegments.append(contentsOf: recognizeSegs)
        }
        
        //Test Raw
//        sasasasas = 300
//        recognizeSegments.forEach { segment in
//            print("recognizeSegments start: \(segment.startIndex) end:\(segment.endIndex) speech:\(segment.speech)")
//            test_SaveToWav(data: segment.data, index: sasasasas)
//            sasasasas+=1
//        }
        
        
        
        
        //增强识别
//        let mutiShardSegments = recognizeSegments.filter { seg in
//            seg.shard
//        }
//
//        var enhanceds: [[RecognizeSegment]] = []
//        var enhancedItems: [RecognizeSegment] = []
//        var tempDataCount = 0
//        let spaceBytes = 16000 * MemoryLayout<Float>.size
//        mutiShardSegments.forEach { segment in
//            if tempDataCount + spaceBytes + segment.data.count > clampFixBytes {
//                enhanceds.append(enhancedItems)
//                enhancedItems = []
//                tempDataCount = 0
//            }
//
//            enhancedItems.append(segment)
//            tempDataCount += segment.data.count
//            tempDataCount += spaceBytes
//        }
//        if !enhancedItems.isEmpty {
//            enhanceds.append(enhancedItems)
//        }
//
//        var sasasasas = 200
//        enhanceds.forEach { segments in
//            var recoginazeData = Data()
//            var rangeTime: [VADRange] = []
//            var tempIndex = 0
//
//            segments.forEach { segment in
//                test_SaveToWav(data: segment.data, index: sasasasas)
//                sasasasas+=1
//                recoginazeData.append(segment.data)
//                recoginazeData.append(Data(repeating: 0, count: spaceBytes))
//
//                let startIndexInrecoginazeData = tempIndex + segment.startIndex
//                let endIndexInrecoginazeData = startIndexInrecoginazeData + segment.endIndex
//                tempIndex += spaceBytes
//
//                rangeTime.append(VADRange(realTimeStamp: TimeRange(start: segment.startTimeStamp, end: segment.endTimeStamp), sampleRange: TimeRange(start: Int64(startIndexInrecoginazeData), end: Int64(endIndexInrecoginazeData))))
//            }
//            test_SaveToWav(data: recoginazeData, index: 400)
//            var speechTranscripts = whisper.transcriptSync(buffer: recoginazeData)
//            speechTranscripts = fillterSpeechTranscript(&speechTranscripts)
//            let matchSegments:[VADAndTranscriptMatchSegment] = matchTranscriptLocationInBufferByVAD(&speechTranscripts, rangeTime)
//            //TODO: 重写回去
//            matchSegments.forEach { segggg in
//                let origin = segments[segggg.vadIndex]
//                recognizeSegments[origin.id].speech = segggg.speechs.first ?? ""
//            }
//
//        }
        
        
        return recognizeSegments
        
        
        
    }
    
}
