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
    var words: Int = 0

    init?(map: ObjectMapper.Map) {}
    
    init(index: Int, feature:[Float]) {
        self.index = index
        self.features.append(feature)
    }

    mutating func mapping(map: Map) {
        index       <- map["index"]
        features    <- map["features"]
    }
}


class SpeakerAnalyseTempModule {
    var speakers: [Speaker] = []
    var isUpdated = false
    func preload() {
        //1. 读用户特征
        if let speakers_str = UserDefaults.standard.string(forKey: "cs_speakers_temp") {
            speakers = Array<Speaker>(JSONString: speakers_str) ?? []
        }
    }
    
    func generateNewIndex() -> Int {
        guard let lastOne = speakers.last else {
            return 0
        }
        return lastOne.index + 1
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
    

//    //更新多个用户信息
//    func updateSpeakers(indexs: [Int], features: [[Float]], words: [Int]) {
//        for (i, _) in indexs.enumerated() {
//            updateSpeaker(index: indexs[i], feature: features[i], word: words[i])
//        }
//
//        speakers.sort { s1, s2 in
//            s1.words > s2.words
//        }
//
//        if let ss = speakers.toJSONString() {
//            UserDefaults.standard.set(ss, forKey: "cs_speakers_temp")
//        }
//    }
    
    func updateSpeaker(index: Int, feature: [Float], word: Int) {
        guard let speakerIndex = speakers.firstIndex(where: {$0.index == index}) else {
            let speaker = Speaker(index: index, feature: feature)
            speakers.append(speaker)
            return
        }
        
        if speakers[speakerIndex].features.count > 3 {
            speakers[speakerIndex].features.removeFirst()
        }
        
        speakers[speakerIndex].features.append(feature)
        speakers[speakerIndex].words += word
        isUpdated = true
    }
    
    func store() {
        guard isUpdated else {
            return
        }
        
        if let ss = speakers.toJSONString() {
            UserDefaults.standard.set(ss, forKey: "cs_speakers_temp")
        }
        
    }
    
}
