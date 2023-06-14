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

class VADModule {
    let sf: Int
    let limitInSec: Int
    let vadFrameFixByte: Int
    let windowSize = 512
    var currentResult: VADResult?
    var test_n = 0
    var test_m = 0
    var test_o = 0
//    var xfsffdfdsf2 = 100
    
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
//        test_SaveToWav(data: buffer, index: test_o,prefix: "o_")
//        test_o += 1
        
        //上次时间超时，重置vad
        if lastEndTimeStamp > 0,
           timeStamp - lastEndTimeStamp > segmentationThreshold {
           resetVAD()
        }
        storeInCache(buffer: buffer,timeStamp: timeStamp)
        doThisVadHandle()
        return vadResultCheck()
    }
}

private extension VADModule {
    func resetVAD() {
        print("resetVAD !!!!!!!!!!!")
        vad.resetState()
    }
    
    func storeInCache(buffer: Data,timeStamp: Int64) {
        print("1 storeInCache buffer size: \(buffer.count)")
        if backupAudioData.count > 0 {
            print("2 add backupAudioData size: \(backupAudioData.count)")
            cacheAudioData = backupAudioData + cacheAudioData
            backupAudioData = Data()
            print("3 cacheAudioData buffer size: \(cacheAudioData.count)")
        }
        
        cacheAudioData.append(buffer)
        print("4 cacheAudioData buffer size: \(cacheAudioData.count)")
        
        
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
    }
    
    func popAvalibleData() -> Data? {
        let chunkCount = Int(cacheAudioData.count / (512 * MemoryLayout<Float>.size))
        let audioFrameCount = chunkCount * 512
        let audioFrameSize = Int(audioFrameCount) * MemoryLayout<Float>.size
        
        guard audioFrameSize > 0 else {
            return nil
        }

    
        let avalibleData = cacheAudioData.prefix(audioFrameSize)
        cacheAudioData.removeSubrange(0..<audioFrameSize)
        print("after popAvalibleData cacheAudioData size: \(cacheAudioData.count)")
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
    
//    func test_SaveToWav(data: Data, index: Int, prefix: String? = nil) {
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
        
//        test_SaveToWav(data: data, index: test_n)
//        test_n+=1
//
        //3.detect
        print("vad 检查数据采样数: \(data.count >> 2)")
        guard var detectResult:[VADResult] = vad.detectContinuouslyForTimeStemp(buffer: data),
              !detectResult.isEmpty else {
            print("vad 没有有效数据")
            return false
        }
        
        
        let vadvvvvvSampleNum = detectResult.reduce(into: 0) { partialResult, timeResult in
            partialResult = partialResult + (timeResult.end - timeResult.start)
        }
        print("vad 有效采样数: \(vadvvvvvSampleNum)")
        
        //5.keep continuous
        guard let lastVADResult:VADResult = detectResult.popLast() else {
            return false
        }
        if lastVADResult.start > 0 {
            //最后如果有语音，那么保留最后备用队列
            let dataStartIndex = lastVADResult.start * MemoryLayout<Float>.size
            let dataEndIndex = lastVADResult.end * MemoryLayout<Float>.size
            let waitHandleData = data.subdata(in: dataStartIndex..<dataEndIndex)
            
//            test_SaveToWav(data: waitHandleData, index: test_n, prefix: "x")

            backupAudioData.append(waitHandleData)
            print("移入备份队列, start:\(dataStartIndex),end:\(dataEndIndex), back count:\(backupAudioData.count)")
        }

        
        

        //4.convert data
        var vadRanges: [VADRange] = detectResult.map { timeResult in
            let startTimestemp = Int64(timeResult.start / (sf / 1000)) + lastStartTimeStamp
            let endTimestemp = Int64(timeResult.end / (sf / 1000)) + lastStartTimeStamp
            
            return VADRange(realTimeStamp: TimeRange(start: startTimestemp, end: endTimestemp), sampleRange: TimeRange(start: Int64(timeResult.start), end: Int64(timeResult.end)))
        }
        
        
        

        
        
        
        

        var rangeDuration = 0
        for rangeItem in vadRanges {
            let rangeData = data.subdata(in: (Int(rangeItem.sampleRange.start) * MemoryLayout<Float>.size)..<(Int(rangeItem.sampleRange.end) * MemoryLayout<Float>.size))
            rangeDuration += Int((rangeItem.realTimeStamp.end - rangeItem.realTimeStamp.start))
            
            let vadRange = VADRange(realTimeStamp: rangeItem.realTimeStamp, sampleRange: rangeItem.sampleRange, data: rangeData)
            vadBuffersInQueue.append(vadRange)
            
//            test_SaveToWav(data: rangeData, index: test_m, prefix: String(test_n-1)+"_i_")
//            test_m+=1
        }
        print("有效时长: \(Double(rangeDuration) * 0.001)")
        
        
        return false
    }
}
