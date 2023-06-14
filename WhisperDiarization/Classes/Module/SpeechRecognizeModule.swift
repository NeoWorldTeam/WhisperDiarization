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
    var sasasasas = 0
    init() {
        whisper = WhisperDiarization()
    }
    
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
            
        }
        return matchSegments
    }
    
    
    func fillterSpeechTranscript(_ transcripts: inout [TranscriptSegment]) -> [TranscriptSegment] {
        

        transcripts.enumerated().forEach { elem in
            let filterSpeech = transcripts[elem.offset].speech.trimmingCharacters(in: .whitespaces)
            transcripts[elem.offset].speech = filterSpeech
        }
        
        
        let speechTranscripts = transcripts.reduce(into: [TranscriptSegment]()) { (result, element) in
            guard !element.speech.hasPrefix("("),
                  !element.speech.hasPrefix("[") else {
                return
            }
            guard let lastItem = result.last else{
                result.append(element)
                return
            }
            
            if lastItem.speech != element.speech {
                result.append(element)
            }
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
    
//    func test_SaveToWav(data: Data, index: Int, prefix: String?) {
//        let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
//        let fileURL = documentsDirectory.appendingPathComponent("audio_" + (prefix ?? "") + String(index) + ".wav")
//
//        // 创建AVAudioFile
//        let audioFile = try! AVAudioFile(forWriting: fileURL, settings: [
//            AVFormatIDKey: Int(kAudioFormatLinearPCM),
//            AVSampleRateKey: 16000,
//            AVNumberOfChannelsKey: 1,
//            AVLinearPCMBitDepthKey: 32,
//            AVLinearPCMIsBigEndianKey: false,
//            AVLinearPCMIsFloatKey: true
//        ])
//
//        // 写入音频数据
//        let audioBuffer = AVAudioPCMBuffer(pcmFormat: audioFile.processingFormat, frameCapacity: UInt32(data.count) / audioFile.processingFormat.streamDescription.pointee.mBytesPerFrame)!
//        audioBuffer.frameLength = audioBuffer.frameCapacity
//        let audioBufferData = audioBuffer.floatChannelData![0]
//        audioBufferData.withMemoryRebound(to: UInt8.self, capacity: data.count) { pointer in
//            data.copyBytes(to: pointer, count: data.count)
//        }
//
//        try! audioFile.write(from: audioBuffer)
//
//        print("文件已经保存到：\(fileURL)")
//    }
    
    func recognize(vadBuffers: [VADBuffer]) -> [RecognizeSegment] {
        
        let matchSegmentsList = vadBuffers.map { vadBuffer in
            
//            test_SaveToWav(data: vadBuffer.buffer, index: sasasasas, prefix: "input_")
            
            var speechTranscripts:[TranscriptSegment] = whisper.transcriptSync(buffer: vadBuffer.buffer)
            
            
//            print("before filter: \(speechTranscripts)")
            speechTranscripts = fillterSpeechTranscript(&speechTranscripts)
//            print("after filter: \(speechTranscripts)")
            
//            for item in speechTranscripts {
//                let dadfdad = vadBuffer.buffer.subdata(in: (item.start<<2)..<(item.end<<2))
//                test_SaveToWav(data: dadfdad, index: sasasasas, prefix: "clip_")
//                sasasasas+=1
//            }

            return matchTranscriptLocationInBufferByVAD(&speechTranscripts, vadBuffer.rangeTimes)
        }
        
        var recognizeSegments:[RecognizeSegment] = []
        for (index, matchSegments) in matchSegmentsList.enumerated() {
            let buffer = vadBuffers[index]
            let recognizeSegs = caculateSegment(matchSegments: matchSegments, vadBuffer: buffer)
            recognizeSegments.append(contentsOf: recognizeSegs)
        }
        return recognizeSegments
        
        
        
    }
    
}
