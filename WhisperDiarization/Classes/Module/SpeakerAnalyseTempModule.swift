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
    let fixHostFeatureCount = 5
    var hostSpeaker: Speaker!
    
    //临时
    var speakers: [Speaker] = []
    
    var recentlyIndex = 0
    var speakerRecentlyRecord:[Int:Int] = [Int:Int]()

    func preload() {
        //1. 读用户特征
        if let speaker_host_str = UserDefaults.standard.string(forKey: "cs_speaker_host") {
            hostSpeaker = Speaker(JSONString: speaker_host_str)
        }else {
            hostSpeaker = Speaker(index: 0, features: [])
        }
        
        speakers.append(hostSpeaker)
    }
    
//    func generateNewIndex() -> Int {
//        return speakers.last!.index + 1
//    }
    
    
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
        
        
        let lastetSpeakers:[Int] = Array<Int>(speakerRecentlyRecord.keys).sorted(by: >)
        let onlyLastetSpeakers = Set(lastetSpeakers)
        var recentSpeakerIndexs:[Int] = Array<Int>(onlyLastetSpeakers.prefix(num - 1))
        
        var speakersLimit = [Int]()
        speakersLimit.append(0)
        speakersLimit.append(contentsOf: recentSpeakerIndexs)

        let speakerFeatures: [[[Float]]] = speakersLimit.map { index in
            return speakers[index].features
        }

        return (speakersLimit,speakerFeatures)
    }

    //更新数据，且将最新的用户提到前面
    func updateSpeaker(index: Int, feature: [Float]) {
        guard let speakerIndex = speakers.firstIndex(where: {$0.index == index}) else {
            var speaker = Speaker(index: index, features: [])
            speaker.features.append(feature)
            speakers.append(speaker)
            return
        }
        
        switch index {
        case 0:
            if speakers[speakerIndex].features.count >= fixHostFeatureCount * 2 {
                speakers[speakerIndex].features.remove(at: fixHostFeatureCount)
            }
            break
            
        default:
            recentlyIndex += 1
            speakerRecentlyRecord[speakerIndex] = recentlyIndex
            if speakers[speakerIndex].features.count >= fixHostFeatureCount * 2{
                speakers[speakerIndex].features.removeFirst()
            }
            break
        }

        speakers[speakerIndex].features.append(feature)
    }
    

}
