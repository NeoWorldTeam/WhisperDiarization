//
//  AudioPreprocess.swift
//  WhisperDiarization
//
//  Created by fuhao on 2023/4/27.
//

import Foundation
import AVFAudio

struct CaptureAudioSegment {
    var timeStamp: Int64
    var buffer: Data
}


class AudioPreprocess {
//    private let _queue: DispatchQueue = DispatchQueue(label: "AudioPreprocess", attributes: .concurrent)
    private var _audioConverter: AVAudioConverter?
    
    private let semaphore = DispatchSemaphore(value: 0)
    private var bufferCaches = [CaptureAudioSegment]()
    
    private let processFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
    private var sourceFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
    private var bufferSize = 0
    
    private let maxItemCount: Int
    init(maxItemCount: Int) {
        self.maxItemCount = maxItemCount
    }
    
    
    private func caculeteFrameCapacity(audioFileA: AVAudioPCMBuffer, audioFormatB: AVAudioFormat) -> Int {
        let audioFormatA = audioFileA.format
        
        let sampleRateA = audioFormatA.sampleRate
        let sampleRateB = audioFormatB.sampleRate

        // 获取A和B音频每个采样点的位深度和声道数
        let channelCountA = audioFormatA.channelCount
        let channelCountB = audioFormatB.channelCount
        
        
        let bitsPerChannelA = audioFormatA.streamDescription.pointee.mBitsPerChannel
        let bitsPerChannelB = audioFormatB.streamDescription.pointee.mBitsPerChannel
        

        // 计算A和B音频每个采样点的字节数
        let bytesPerSampleA = bitsPerChannelA / 8 * channelCountA
        let bytesPerSampleB = bitsPerChannelB / 8 * channelCountB

        // 计算A和B音频每个帧的字节数
        let bytesPerFrameA = bytesPerSampleA * audioFormatA.streamDescription.pointee.mFramesPerPacket
        let bytesPerFrameB = bytesPerSampleB * audioFormatB.streamDescription.pointee.mFramesPerPacket

        // 计算将A音频转换为B音频所需的帧容量
        let frameCapacity = Int(ceil(Float(audioFileA.frameLength) / Float(bytesPerFrameA) * (Float(sampleRateB) / Float(sampleRateA)) * Float(bytesPerFrameB)))
        return frameCapacity
    }
    
    private func _checkAudioFormat(pcmFormatA: AVAudioFormat,pcmFormatB: AVAudioFormat) -> Bool {
        // 检查采样率是否匹配
        guard pcmFormatA.sampleRate == pcmFormatB.sampleRate else {
            return false
        }
        
        // 检查通道数是否匹配
        guard pcmFormatA.channelCount == pcmFormatB.channelCount else {
            return false
        }
        
        // 检查位深度是否匹配
        guard pcmFormatA.commonFormat == pcmFormatB.commonFormat else {
            return false
        }
        
        return true
    }
    
    private func resetSourceFormat(buffer: AVAudioPCMBuffer) {
        bufferSize = caculeteFrameCapacity(audioFileA: buffer, audioFormatB: processFormat)
        _audioConverter =  AVAudioConverter(from: buffer.format, to: processFormat)
        guard _audioConverter != nil else {
            return
        }
        sourceFormat = buffer.format
    }
    
    private func converAudio(_ buffer: AVAudioPCMBuffer) -> AVAudioPCMBuffer?{
        guard _checkAudioFormat(pcmFormatA: buffer.format, pcmFormatB: processFormat) == false else {
            return buffer
        }
        
        if _checkAudioFormat(pcmFormatA: buffer.format, pcmFormatB: sourceFormat) == false {
            resetSourceFormat(buffer: buffer)
        }
        
        guard let audioConverter = _audioConverter else {
            return nil
        }
        
        
        let tempAudioBuffer = AVAudioPCMBuffer(pcmFormat: processFormat, frameCapacity: AVAudioFrameCount(bufferSize))!
        let inputBlock: AVAudioConverterInputBlock = { inNumPackets, outStatus in
            outStatus.pointee = AVAudioConverterInputStatus.haveData
            return buffer
        }
        
        var error: NSError?
        let status = audioConverter.convert(to: tempAudioBuffer, error: &error, withInputFrom: inputBlock)
        guard status != .error && error == nil else {
            return nil
        }
        
        return tempAudioBuffer
    }
    
    func enqueues(_ buffer: AVAudioPCMBuffer,timeStamp: Int64) {
        guard self.bufferCaches.count < self.maxItemCount else {
            print("生产队列已满1")
            self.semaphore.signal()
            return
        }
        let data = Data(bytes: buffer.floatChannelData![0], count: Int(buffer.frameLength) * MemoryLayout<Float>.size)
        
        
        let audioSegment = CaptureAudioSegment(timeStamp: timeStamp, buffer: data)
        self.bufferCaches.append(audioSegment)
        self.semaphore.signal()
    }
    
    
    func dequeue() -> CaptureAudioSegment? {
        var item: CaptureAudioSegment?
        if self.bufferCaches.count > 0 {
            item = self.bufferCaches.removeFirst()
        }
        
        if item == nil {
//            print("消费队列等待")
            semaphore.wait()
//            print("消费队列结束等待")
        }
        
        if item == nil && self.bufferCaches.count > 0 {
            item = self.bufferCaches.removeFirst()
        }
//        print("消费队列返回结果")
        return item
    }
    
    
}
