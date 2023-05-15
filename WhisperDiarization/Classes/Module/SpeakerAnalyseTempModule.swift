//
//  SpeakerAnalyseTempModule.swift
//  WhisperDiarization
//
//  Created by fuhao on 2023/5/11.
//

import Foundation
import ObjectMapper

struct Speaker : Mappable {
    var index: Int = -1
    var features: [[Float]] = []

    init?(map: ObjectMapper.Map) {}
    
    init(index: Int, features:[[Float]]) {
        self.index = index
        self.features = features
    }
    
    

    mutating func mapping(map: Map) {
        index       <- map["index"]
        features    <- map["features"]
    }
}


class SpeakerAnalyseTempModule {
    let fixHostFeatureCount = 2
    var hostSpeaker: Speaker!
    
    //临时
    var speakers: [Speaker] = []

    func preload() {
        //1. 读用户特征
        if let speaker_host_str = UserDefaults.standard.string(forKey: "cs_speaker_host") {
            hostSpeaker = Speaker(JSONString: speaker_host_str)
        }else {
            hostSpeaker = Speaker(index: 0, features: [])
        }
        
        speakers.append(hostSpeaker)
    }
    
    func generateNewIndex() -> Int {
        return speakers.last!.index + 1
    }
    
    
    func saveToHost(_ hostFeatures: [[Float]]) {
        guard var hostSpeaker = speakers.first else {
            let appendFeatures:[[Float]] = Array(hostFeatures.prefix(fixHostFeatureCount))
            speakers.append(Speaker(index: 0, features: appendFeatures))
            return
        }
        
        let remianFixFeature = fixHostFeatureCount - speakers[0].features.count
        guard remianFixFeature > 0 else {
            return
        }
        let appendFeatures = hostFeatures.prefix(remianFixFeature)
        speakers[0].features.append(contentsOf: appendFeatures)
        
        if let ss = speakers[0].toJSONString() {
            UserDefaults.standard.set(ss, forKey: "cs_speaker_host")
        }
    }
    
    
    func getTopSpeakerFeature(num: Int) -> ([Int], [[[Float]]]) {
        guard speakers.count > 0 else {
            return ([],[])
        }
        let minNum = min(num, speakers.count)

        let speakersLimit = Array(speakers[0..<minNum])
        
        let speakerFeatures: [[[Float]]] = speakersLimit.map { score in
            return speakers[score.index].features
        }
        let speakerIndexs: [Int] = speakersLimit.map { score in
            return speakers[score.index].index
        }

        return (speakerIndexs,speakerFeatures)
    }

    
    func updateSpeaker(index: Int, feature: [Float]) {
        guard let speakerIndex = speakers.firstIndex(where: {$0.index == index}) else {
            var speaker = Speaker(index: index, features: [])
            speaker.features.append(feature)
            speakers.append(speaker)
            return
        }
        
        switch index {
        case 0:
            if speakers[speakerIndex].features.count > 8 {
                speakers[speakerIndex].features.remove(at: 5)
            }
            break
            
        default:
            if speakers[speakerIndex].features.count > 3 {
                speakers[speakerIndex].features.removeFirst()
            }
            break
        }

        speakers[speakerIndex].features.append(feature)
    }
    
    
}
