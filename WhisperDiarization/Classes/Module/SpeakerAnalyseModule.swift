////
////  SpeakerAnalyseModule.swift
////  WhisperDiarization
////
////  Created by fuhao on 2023/5/11.
////
//
//import Foundation
//import ObjectMapper
//
//
//struct Speaker : Mappable {
//    var index: Int = -1
//    var features: [[Float]] = []
//
//    init?(map: ObjectMapper.Map) {}
//
//    mutating func mapping(map: Map) {
//        index       <- map["index"]
//        features    <- map["features"]
//    }
//}
//
//struct SpeakerScore : Mappable {
//    mutating func mapping(map: ObjectMapper.Map) {
//        index       <- map["index"]
//        score       <- map["score"]
//    }
//    
//    init(){}
//    init?(map: ObjectMapper.Map) {}
//    var index: Int = -1
//    var score: Float = 0
//}
//
//struct RecordData  : Mappable{
//    var records: [DayRecord] = []
//    
//    
//    init(){}
//    init?(map: ObjectMapper.Map) {}
//    mutating func mapping(map: Map) {
//        records       <- map["records"]
//    }
//}
//
//struct DayRecord  : Mappable{
//    var hours: [HourRecord] = []
//    
//    init(){}
//    init?(map: ObjectMapper.Map) {}
//    mutating func mapping(map: Map) {
//        hours       <- map["hours"]
//    }
//}
//
//struct HourRecord  : Mappable{
//    var timeStampRange: Range<Int64> = 0..<1
//    var speakerRecords: [SpeakerRecord] = []
//    
//    init(){
//        let currentTimeMillis = Int64(Date().timeIntervalSince1970 * 1000)
//        timeStampRange = currentTimeMillis..<(currentTimeMillis + 60 * 60 * 1000)
//    }
//    init?(map: ObjectMapper.Map) {}
//    mutating func mapping(map: Map) {
//        timeStampRange       <- map["timeStampRange"]
//        speakerRecords       <- map["speakerRecords"]
//    }
//}
//
//struct SpeakerRecord  : Mappable{
//    var index: Int = -1
//    var wordsLength: Int = 0
//    var score: Float = 0
//    
//    init(){}
//    init?(map: ObjectMapper.Map) {}
//    mutating func mapping(map: Map) {
//        index             <- map["index"]
//        wordsLength       <- map["wordsLength"]
//        score             <- map["score"]
//    }
//}
//
//
//
//
//class SpeakerAnalyseModule {
//    var speakers: [Speaker] = []
//    var recordData: RecordData = RecordData()
//    var currentHourRecord: HourRecord = HourRecord()
//    var speakerScoreList: [SpeakerScore] = []
//    
//    func preload() {
//        //1. 读用户特征
//        if let speakers_str = UserDefaults.standard.string(forKey: "cs_speakers") {
//            speakers = Array<Speaker>(JSONString: speakers_str) ?? []
//        }
//        
//        //2. 读取用户记录
//        if let records_str = UserDefaults.standard.string(forKey: "cs_speaker_records") {
//            if let records = RecordData(JSONString: records_str) {
//                recordData = records
//            }
//        }
//        
//        //3. 读当前用户权重
//        if let speakers_rank_str = UserDefaults.standard.string(forKey: "cs_speakers_rank") {
//            speakerScoreList = Array<SpeakerScore>(JSONString: speakers_rank_str) ?? []
//        }
//        
//        //4. 读最新的小时的数据
//        if let hour_data_str = UserDefaults.standard.string(forKey: "cs_hour_data") {
//            if let hourRecord = HourRecord(JSONString: hour_data_str) {
//                currentHourRecord = hourRecord
//            }
//        }
//        
//        //TODO: 检查小时数据的更新
//        
//    }
//    
//    
//    func getTopSpeakerFeature(num: Int) -> [[[Float]]] {
//        guard speakerScoreList.count > 0 else {
//            return []
//        }
//        let minNum = min(num, speakerScoreList.count)
//    
//        let speakerScoreRangeList = Array(speakerScoreList[0..<minNum])
//        let speakerFeatures: [[[Float]]] = speakerScoreRangeList.map { score in
//            return speakers[score.index].features
//        }
//        return speakerFeatures
//    }
//    
//    func addCurrentHourToRecord() {
//        
//    }
//    
//    //更新多个用户信息
//    func updateSpeaker(indexs: [Int], features: [[Float]], words: [Int], timeStamps: [Int64]) {
//        guard indexs.count > 0,
//              indexs.count == features.count,
//              indexs.count == words.count,
//              indexs.count == timeStamps.count else {
//            return
//        }
//        
//        //判断是否在当前工作小时
//        guard let speechTimeStamp = timeStamps.first else {
//            return
//        }
//        if !currentHourRecord.timeStampRange.contains(speechTimeStamp) {
//            //更新当前工作小时
//            addCurrentHourToRecord()
//        }
//        
////        indexs.forEach { index in
////            guard let speakerRecord = currentHourRecord.speakerRecords.first(where: {$0.index == index}) else {
////                let sss = SpeakerRecord()
////                sss.index = index
////
////                currentHourRecord.speakerRecords.append(<#T##newElement: SpeakerRecord##SpeakerRecord#>)
////
////                continue
////            }
////        }
//    }
//    
//
//    
//}
