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
import RosaKit
import SpeakerEmbeddingForiOS

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
    
    func convertSTFTtoArray(complexNumbers: [[(real: Double, imagine: Double)]]) -> [[[Float]]]{
        var convertedNumbers: [[[Float]]] = []
        
        complexNumbers.forEach { rows in
            var convertedRow: [[Float]] = []
            
            rows.forEach { complexNumber in
                let real = complexNumber.real
                let imaginary = complexNumber.imagine
                
                let convertedNumber: [Float] = [Float(real), Float(imaginary)]
                convertedRow.append(convertedNumber)
            }
            convertedNumbers.append(convertedRow)
        }
        
        return convertedNumbers
    }
    
    @objc
    func clickTest(sender: Any) {
        
        
//        guard speechRecognition.test() else {
//            return
//        }
        
        
        let filePath = Bundle.main.url(forResource: "ssss_16k", withExtension: "wav")!
        guard let buffer = loadAudioFile(url: filePath) else {
            return
        }
        cacheBuffer = buffer
        let numSamples = buffer.frameLength
        let floatArray:UnsafeMutablePointer<Float> = buffer.floatChannelData![0]
        let floatPointer = UnsafePointer<Float>(floatArray)
        
//        let samples:[Double] = (0..<numSamples).map { Double(floatArray[Int($0)]) }
//        let stft = samples.stft(nFFT: 400, hopLength: 160)
//        let STFT = convertSTFTtoArray(complexNumbers: stft)
//
//
//
//        let flatSTFT = STFT.flatMap { channel in
//            channel.flatMap { time in
//                time.flatMap { v in
//                    v
//                }
//            }
//        }
//
//        let data = flatSTFT.withUnsafeBytes { Data($0) }
//        let featureExtarer = SpeakerEmbedding()
//        let embeding = featureExtarer.extractFeature(data: data)
//        print(embeding)
        
        
//        //MARK: - 测试数据
//        let jsonData = try! JSONSerialization.data(withJSONObject: STFT, options: .prettyPrinted)
//        if let jsonString = String(data: jsonData, encoding: .utf8) {
//            let fileURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("stft.json")
//            do {
//                try jsonString.write(to: fileURL, atomically: true, encoding: .utf8)
//                print("JSON file saved successfully")
//            } catch {
//                print("Error: \(error.localizedDescription)")
//            }
//        }
//
//
        speechRecognition.pushAudioBuffer(buffer: buffer, timeStamp: Int64(Date().timeIntervalSince1970 * 1000))
        
        
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }

}

