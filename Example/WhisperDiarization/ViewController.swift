//
//  ViewController.swift
//  WhisperDiarization
//
//  Created by fuhao on 04/20/2023.
//  Copyright (c) 2023 fuhao. All rights reserved.
//

import UIKit
import WhisperDiarization
import AVFoundation
import Accelerate

class ViewController: UIViewController {
    var cacheBuffer: AVAudioPCMBuffer?
    let speechRecognition = CSSpeechRecognition()
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        self.view.addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(clickTest)))
    }
    
    
    func loadAudioFile(url: URL?) -> AVAudioPCMBuffer? {
        guard let url = url,
              let file = try? AVAudioFile(forReading: url) else {
            return nil
        }

        let format = file.processingFormat
        let frameCount = UInt32(file.length)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            return nil
        }

        do {
            try file.read(into: buffer)
            return buffer
        } catch {
            return nil
        }
    }
    
    @objc
    func clickTest(sender: Any) {
        
        
//        let X1:[Float] = [1, 2, 3, 4,
//                 5, 6, 7, 8,
//                 9, 10, 11, 12]
//        let M = 3  // X 的行数
//        let N = 4  // X 的列数
//        
//        var Y = [Float](repeating: 0.0, count: M*M) // 存储结果的矩阵
//            
//        var XT = [Float](repeating: 0.0, count: N*M)
//        vDSP_mtrans(X1, 1, &XT, 1, UInt(N), UInt(M))
//        vDSP_mmul(X1, 1, XT, 1, &Y, 1, UInt(M), UInt(N), UInt(N))
//        
//        var result = [Float](repeating: 0.0, count: M) // 存储每行和的数组
//        
//        // 对每行进行求和
//        for i in 0..<M {
//            vDSP_sve(&Y[i*M], 1, &result[i], UInt(M))
//        }
////        var norms = [Double](repeating: 0.0, count: n * m)
////        let strideX = vDSP_Stride(m)  // X 的列数的跨度
////
////        vDSP_vsqD(X1, strideX, &norms, 1, vDSP_Length(n * m))  // 平方
////        vDSP_sveD(norms, 1, &norms[0], vDSP_Length(n))  // 求和
//        
//        return
        
        let filePath = Bundle.main.url(forResource: "output29", withExtension: "wav")!
        guard let buffer = loadAudioFile(url: filePath) else {
            return
        }
        cacheBuffer = buffer
        let numSamples = buffer.frameLength
        let floatArray = buffer.floatChannelData![0]
        let floatPointer = UnsafePointer<Float>(floatArray)
        
        speechRecognition.pushAudioBuffer(buffer: buffer, timeStamp: Int64(Date().timeIntervalSince1970 * 1000))
        
        
//        whisper?.transcript(samples: floatPointer, numSamples: Int(numSamples)) { result, error in
//            guard let result:TranscriptResult = result else {
//                return
//            }
//
//            for item in result.speechs {
//                print("s:\(item.speech), t0:\(item.startTimeStamp), t1:\(item.endTimeStamp)")
//            }
//
//        }
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }

}

