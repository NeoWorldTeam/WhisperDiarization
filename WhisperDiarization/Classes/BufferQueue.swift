//
//  BufferQueue.swift
//  WhisperDiarization
//
//  Created by fuhao on 2023/4/27.
//
import Foundation
import AVFAudio

class BufferQueue<T> {
    private var items = [T]()
    private let semaphore = DispatchSemaphore(value: 0)
    private let queue = DispatchQueue(label: "BufferQueue", attributes: .concurrent)
    
    func enqueue(_ item: T) {
        queue.async(flags: .barrier) {
            self.items.append(item)
            self.semaphore.signal()
        }
    }
    
    func enqueues(_ items: [T]) {
        queue.async(flags: .barrier) {
            self.items.append(contentsOf: items)
            self.semaphore.signal()
        }
    }
    
    func dequeue() -> T? {
        var item: T?
        queue.async(flags: .barrier) {
            if !self.items.isEmpty {
                item = self.items.removeFirst()
            }
        }
        if item == nil {
            semaphore.wait()
            queue.async(flags: .barrier) {
                item = self.items.removeFirst()
            }
        }
        return item
    }
}


class AudioBufferQueue {
    var tempBufferCache = [AVAudioPCMBuffer]()
    var tempSampleNum = 0
    
    let fixBuffers: BufferQueue<Date> = BufferQueue<Date>()
    
    func dequeue(buffer: AVAudioPCMBuffer)  {
        let sampleNum = buffer.frameLength        
        tempSampleNum += Int(sampleNum)
        tempBufferCache.append(buffer)
        
        guard tempSampleNum >= 512 else {
            return
        }
        
        let appendData = Data()
        for item in tempBufferCache {
            
        }
        

    }
    
    
}


