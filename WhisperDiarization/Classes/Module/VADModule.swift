//
//  VADManager.swift
//  WhisperDiarization
//
//  Created by fuhao on 2023/5/8.
//

import Foundation
import SpeakerEmbeddingForiOS
import AVFoundation


struct TimeRange {
    var start: Int64
    var end: Int64
    
    func duration() -> Int64{
        return end - start
    }
}

struct VADRange {
    let realTimeStamp: TimeRange
    let sampleRange: TimeRange
    var data: Data?
}

struct VADBuffer {
    var buffer: Data
    var rangeTimes: [VADRange]
}

//struct VADResult {
//    var buffers: [VADBuffer]
//
//    init() {
//        let buffer = VADBuffer(buffer: Data(), rangeTimes: [])
//        buffers = [buffer]
//    }
//}

class VADModule {
    let sf: Int
    let limitInSec: Int
    let vadFrameFixByte: Int
    let windowSize = 512
    var currentResult: VADResult?
    
    let vad: VoiceActivityDetector = VoiceActivityDetector()
    let processFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
    
    
    var lastStartTimeStamp: Int64 = 0
    var lastEndTimeStamp: Int64 = 0
    
    var cacheAudioData = Data()
    var backupAudioData = Data()
    var segmentationThreshold = 1000

    var vadBuffersInQueue:[VADRange] = []
    
    init(sf: Int = 16000, limitInSec: Int = 30) {
        self.sf = sf
        self.limitInSec = limitInSec
        self.vadFrameFixByte = MemoryLayout<Float>.size * sf * limitInSec
    }

    func checkAudio(buffer: Data, timeStamp: Int64) -> [VADBuffer] {
        //上次时间超时，重置vad
        if lastEndTimeStamp > 0 && timeStamp - lastEndTimeStamp > segmentationThreshold {
//            doLastVadHandle()
            resetVAD()
        }
        storeInCache(buffer: buffer,timeStamp: timeStamp)
        while doThisVadHandle() == false {
            print("doThisVadHandle finish cacheAudioData count is \(cacheAudioData.count)")
        }
        return vadResultCheck()
    }
}

private extension VADModule {
    func resetVAD() {
        vad.resetState()
    }
    
    func storeInCache(buffer: Data,timeStamp: Int64) {
        if backupAudioData.count > 0 {
            cacheAudioData = backupAudioData + cacheAudioData
            backupAudioData = Data()
        }
        cacheAudioData.append(buffer)

        
        
        //更新时间戳
        lastEndTimeStamp = Int64((buffer.count / MemoryLayout<Float>.size) / (sf / 1000)) + timeStamp
        let dataDurationTimeStamp = cacheAudioData.count / MemoryLayout<Float>.size / (sf / 1000)
        lastStartTimeStamp = lastEndTimeStamp - Int64(dataDurationTimeStamp)
    }

    
    
    func createVadBuffer(vadRanges: [VADRange], rangeSpace: Int, startIndex: Int, endIndex: Int) -> VADBuffer {
        var buffer = Data()
        var resultRanges: [VADRange] = []
        for index in startIndex..<endIndex {
            let startSampleIndex = buffer.count / MemoryLayout<Float>.size
            
            let vadRange = vadRanges[index]
            buffer.append(vadRange.data!)
            
            //计算剩余空间
            let dataRestSpace = (30 * 16000 * MemoryLayout<Float>.size) - buffer.count
            let spaceSize = min( dataRestSpace , rangeSpace)
            buffer.append(Data(repeating: 0, count: spaceSize))
            
            let sampleCount = vadRange.sampleRange.duration()
            let newVadRange = VADRange(realTimeStamp: vadRange.realTimeStamp, sampleRange: TimeRange(start: Int64(startSampleIndex), end: Int64(startSampleIndex + Int(sampleCount))))
            resultRanges.append(newVadRange)
        }
        
//        //计算剩余空间
//        let dataRestSpace = (30 * 16000 * MemoryLayout<Float>.size) - buffer.count
//        if dataRestSpace > 0 {
//            buffer.append(Data(repeating: 0, count: dataRestSpace))
//        }
        
        
        return VADBuffer(buffer: buffer, rangeTimes: resultRanges)
    }
    
    //判定有合适的vad数据
    func vadResultCheck() -> [VADBuffer] {
        //1. 历史数据拼接
        var results:[VADBuffer] = []
        let rangeSpace:Int = Int(sf) * MemoryLayout<Float>.size
        
        
        guard !vadBuffersInQueue.isEmpty else {
            return results
        }
        var startIndex = 0
        var chunkDuration = 0
        
        for index in 0..<vadBuffersInQueue.count {
            let vadRange = vadBuffersInQueue[index]

            
            let duration = vadRange.realTimeStamp.duration()
            if chunkDuration + Int(duration) > (29 * 1000) {
                //超过限制,加入队列
                let vadBuffer = createVadBuffer(vadRanges: vadBuffersInQueue, rangeSpace: rangeSpace, startIndex: startIndex, endIndex: index)
                startIndex = index
                results.append(vadBuffer)
                chunkDuration = 0
            }
            
            
            chunkDuration += (Int(duration) + 1000)
        }
        
        vadBuffersInQueue.removeSubrange(0..<startIndex)
        
        
        return results

//        //当前buffers 长度
//        let currentTotalBytes = vadBuffersInQueue.reduce(0) { (result1, buffer: VADBuffer) -> Int in
//            return result1 + buffer.rangeTimes.reduce(0) { (result2, rangeTime: VADRange) -> Int in
//                return result2 + MemoryLayout<Float>.size * Int(rangeTime.sampleRange.end - rangeTime.sampleRange.start) + rangeSpace
//            }
//        }
//
//
//
//        //可能存在的buffer数量
//        var usedBufferNum:Int = currentTotalBytes/(30 * sf * MemoryLayout<Float>.size)
////        print("usedBufferNum:\(usedBufferNum),currentTotalBytes size:\(currentTotalBytes)")
//        guard usedBufferNum > 0 else {
//            return results
//        }
//        usedBufferNum += results.count
//
//
//        //2. 当前数据拼接,只加入必要的，
//        if vadBuffersInQueue.isEmpty == false {
//            var newData = Data()
//            var newRange:[VADRange] = []
//            while vadBuffersInQueue.count > 0 && results.count < usedBufferNum {
//                guard vadBuffersInQueue.isEmpty == false else {
//                    break
//                }
//
//                while vadBuffersInQueue[0].rangeTimes.count > 0 && results.count < usedBufferNum {
//                    guard vadBuffersInQueue[0].rangeTimes.isEmpty == false else {
//                        fatalError("buffer.rangeTimes.count should be more than 0")
//                    }
//
//                    let startIndex = Int(vadBuffersInQueue[0].rangeTimes[0].sampleRange.start) * MemoryLayout<Float>.size
//                    let endIndex = Int(vadBuffersInQueue[0].rangeTimes[0].sampleRange.end) * MemoryLayout<Float>.size
//                    let rangeData = vadBuffersInQueue[0].buffer.subdata(in: startIndex..<endIndex)
//
//                    //超过大小
//                    if newData.count + rangeData.count + rangeSpace >= (30 * sf * MemoryLayout<Float>.size) {
//                        let remainBytes = (30 * sf * MemoryLayout<Float>.size) - newData.count
//                        newData.append(Data(repeating: 0, count: remainBytes))
//                        results.append(VADBuffer(buffer: newData, rangeTimes: newRange))
//
//                        if results.count >= usedBufferNum {
//                            break
//                        }
//
//                        newData = Data()
//                        newRange = []
//                    }
//
//                    let startSample = Int64(newData.count / MemoryLayout<Float>.size)
//
//                    let endSample = Int64(startSample) + (vadBuffersInQueue[0].rangeTimes[0].sampleRange.end - vadBuffersInQueue[0].rangeTimes[0].sampleRange.start)
//
//                    let sampleRange = TimeRange(start: startSample, end: endSample)
//
//                    newRange.append(VADRange(realTimeStamp: vadBuffersInQueue[0].rangeTimes[0].realTimeStamp, sampleRange: sampleRange))
//
//                    newData.append(rangeData)
//
//
//                    if newData.count + rangeSpace <= (30 * sf * MemoryLayout<Float>.size) {
//                        newData.append(Data(repeating: 0, count: rangeSpace))
//                    }
//
//                    //推出使用后的range
//                    vadBuffersInQueue[0].rangeTimes.removeFirst()
//                }
//
//                //推出使用后的buffer
//                if vadBuffersInQueue[0].rangeTimes.isEmpty {
//                    vadBuffersInQueue.removeFirst()
//                }
//            }
//        }
//        return results
        
    }
    
    func popAvalibleData() -> Data? {
        let chunkCount = Int(cacheAudioData.count / (512 * MemoryLayout<Float>.size))
        let audioFrameCount = chunkCount * 512
        let audioFrameSize = Int(audioFrameCount) * MemoryLayout<Float>.size
        
        guard audioFrameSize > 0 else {
            return nil
        }

        
        let avalibleData = cacheAudioData.subdata(in: 0..<audioFrameSize)
        cacheAudioData.removeSubrange(0..<audioFrameSize)
        return avalibleData
    }
    
    func generateAudioBuffer(data: Data) -> AVAudioPCMBuffer {
        let audioFrameSize = data.count
        let audioFrameCount = AVAudioFrameCount(audioFrameSize / MemoryLayout<Float>.size)
        guard let pcmBuffer = AVAudioPCMBuffer(pcmFormat: processFormat, frameCapacity: audioFrameCount) else {
            fatalError("Unable to create PCM buffer")
        }
        pcmBuffer.frameLength = audioFrameCount

        let pcmFloatPointer: UnsafeMutablePointer<Float> = pcmBuffer.floatChannelData![0]
        let pcmRawPointer = pcmFloatPointer.withMemoryRebound(to: UInt8.self, capacity: audioFrameSize) {
            return UnsafeMutableRawPointer($0)
        }

        data.withUnsafeBytes { (bytes: UnsafeRawBufferPointer) in
            pcmRawPointer.copyMemory(from: bytes.baseAddress!, byteCount: data.count)
        }

        return pcmBuffer
    }
    
//    func doLastVadHandle() {
//        print("doLastVadHandle and vadBuffersInQueue:\(vadBuffersInQueue.count) vadBuffers:\(vadBuffers.count)")
//        //copy data
//        vadBuffers.append(contentsOf: vadBuffersInQueue)
//        vadBuffersInQueue.removeAll()
//
//
//        //1.check
//        //1.1 拼接备份
//        if !backupAudioData.isEmpty {
//            cacheAudioData = backupAudioData + cacheAudioData
//            backupAudioData = Data()
//        }
//        guard cacheAudioData.count >= (windowSize * MemoryLayout<Float>.size) else{
//            return
//        }
//
//
//        //7.release
//        defer {
//            cacheAudioData.removeAll()
//        }
//
//        //2.get data
//        guard let data = getAvalibleData() else {
//            return
//        }
////        let pcmBuffer = generateAudioBuffer(data: data)
//
//
//        //3.detect
//        guard let detectResult:[VADTimeResult] = vad.detectContinuouslyForTimeStemp(buffer: data) else {
//            return
//        }
//        guard detectResult.isEmpty == false else {
//            return
//        }
//
//
//        //4.convert data
//        var vadRanges: [VADRange] = detectResult.map { timeResult in
////            let startBytes = timeResult.start * MemoryLayout<Float>.size
////            let endBytes = timeResult.end * MemoryLayout<Float>.size
//            let startTimestemp = Int64(timeResult.start / (sf / 1000))  + lastStartTimeStamp
//            let endTimestemp = Int64(timeResult.end / (sf / 1000))  + lastStartTimeStamp
//
//            return VADRange(realTimeStamp: TimeRange(start: Int64(startTimestemp), end: Int64(endTimestemp)), sampleRange: TimeRange(start: Int64(timeResult.start), end: Int64(timeResult.end)))
//        }
//
//
//
//
//
//
//        //5.keep continuous
////        guard let lastVadRange:VADRange = vadRanges.last else {
////            return
////        }
////        let lastEndSampleIndex = Int(lastVadRange.sampleRange.end) * MemoryLayout<Float>.size
////        //最后如果有语音，那么保留最后一个在队列
////        if data.count - lastEndSampleIndex < 512 * MemoryLayout<Float>.size {
////            vadRanges.removeLast()
////            useDataIndex = Int(vadRanges.last?.sampleRange.end ?? 0)
////        }
//
//
//        //6. add buffer
//        guard vadRanges.isEmpty == false else {
//            return
//        }
//
//        let vadBuffer = VADBuffer(buffer: data, rangeTimes: vadRanges)
//        vadBuffers.append(vadBuffer)
//    }

    func doThisVadHandle() -> Bool {
//        print("doThisVadHandle and vadBuffersInQueue count is \(vadBuffersInQueue.count)")
        //1.check
        guard cacheAudioData.count >= (windowSize * MemoryLayout<Float>.size) else{
            return true
        }
        
        //2.get data
        guard let data = popAvalibleData() else {
            return true
        }
        //3.detect
        guard let detectResult:[VADTimeResult] = vad.detectContinuouslyForTimeStemp(buffer: data),
              !detectResult.isEmpty else {
            return false
        }

        //4.convert data
        var vadRanges: [VADRange] = detectResult.map { timeResult in
            let startTimestemp = Int64(timeResult.start / (sf / 1000)) + lastStartTimeStamp
            let endTimestemp = Int64(timeResult.end / (sf / 1000)) + lastStartTimeStamp
            
            return VADRange(realTimeStamp: TimeRange(start: startTimestemp, end: endTimestemp), sampleRange: TimeRange(start: Int64(timeResult.start), end: Int64(timeResult.end)))
        }
        
        //5.keep continuous
        guard let lastVadRange:VADRange = vadRanges.last else {
            return false
        }
        let lastEndSampleBytes = Int(lastVadRange.sampleRange.end) * MemoryLayout<Float>.size
        //最后如果有语音，那么保留最后备用队列
        if vadRanges.count > 1 && (data.count - lastEndSampleBytes) <= (512 * MemoryLayout<Float>.size) {
            let waitHandleData = data.subdata(in: Int(lastVadRange.sampleRange.start) * MemoryLayout<Float>.size..<Int(lastVadRange.sampleRange.end) * MemoryLayout<Float>.size)
            backupAudioData.append(waitHandleData)
            vadRanges.removeLast()
            print("移入备份队列, start:\(lastVadRange.sampleRange.start),end:\(lastVadRange.sampleRange.end), back count:\(backupAudioData.count)")
        }

        var rangeDuration = 0
        for rangeItem in vadRanges {
            let rangeData = data.subdata(in: (Int(rangeItem.sampleRange.start) * MemoryLayout<Float>.size)..<(Int(rangeItem.sampleRange.end) * MemoryLayout<Float>.size))
            rangeDuration += Int((rangeItem.realTimeStamp.end - rangeItem.realTimeStamp.start))
            
            let vadRange = VADRange(realTimeStamp: rangeItem.realTimeStamp, sampleRange: rangeItem.sampleRange, data: rangeData)
            vadBuffersInQueue.append(vadRange)
        }
        print("有效时长: \(Double(rangeDuration) * 0.001)")
        
        
        return false
    }
}
