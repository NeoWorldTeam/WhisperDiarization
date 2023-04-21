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

class ViewController: UIViewController {
    var whisper: WhisperDiarization?
    var cacheBuffer: AVAudioPCMBuffer?
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        self.view.addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(clickTest)))
        whisper = WhisperDiarization()
    }
    
    @objc
    func clickTest(sender: Any) {
        let filePath = Bundle.main.path(forResource: "output29", ofType: "wav")!
        let file = try! AVAudioFile(forReading: URL(fileURLWithPath: filePath))
        let format = file.processingFormat
        print("format: \(format)")
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(file.length))!
        try! file.read(into: buffer)
        cacheBuffer = buffer
        let numSamples = buffer.frameLength
        let floatArray = buffer.floatChannelData![0]
        let floatPointer = UnsafePointer<Float>(floatArray)
        whisper?.transcript(samples: floatPointer, numSamples: Int(numSamples)) { result, error in
            guard let result:TranscriptResult = result else {
                return
            }
            
            for item in result.speechs {
                print("s:\(item.speech), t0:\(item.startTimeStamp), t1:\(item.endTimeStamp)")
            }
            
        }
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }

}

