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
        
        
//        guard speechRecognition.test() else {
//            return
//        }
        
        
        let filePath = Bundle.main.url(forResource: "output29", withExtension: "wav")!
        guard let buffer = loadAudioFile(url: filePath) else {
            return
        }
        cacheBuffer = buffer
        let numSamples = buffer.frameLength
        let floatArray = buffer.floatChannelData![0]
        let floatPointer = UnsafePointer<Float>(floatArray)
        
        speechRecognition.pushAudioBuffer(buffer: buffer, timeStamp: Int64(Date().timeIntervalSince1970 * 1000))
        
        
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }

}

