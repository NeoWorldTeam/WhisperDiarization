//
//  agglomerativeClustering.swift
//  SpeakerEmbeddingForiOS
//
//  Created by fuhao on 2023/4/28.
//

import Foundation
import Accelerate


typealias Cluster = [[Float]]
typealias ClusterSize = Int

internal class MLTools {
    static func transposeMatrix(_ matrix: [[Float]]) -> [[Float]] {
        let rowCount = matrix.count
        let columnCount = matrix[0].count
        
        let inputMatrix = matrix.flatMap { $0 }
        var outputMatrix = [Float](repeating: 0, count: rowCount * columnCount)
        
        // 使用Accelerate中的转置函数进行计算
        vDSP_mtrans(inputMatrix, 1, &outputMatrix, 1, UInt(columnCount), UInt(rowCount))
        
        // 将结果向量转换回矩阵
        var result = [[Float]]()
        for i in 0..<columnCount {
            let row = Array(outputMatrix[i*rowCount..<i*rowCount+rowCount])
            result.append(row)
        }
        
        return result
    }
    
    static func transposeMatrix(_ inputMatrix:inout [Float], n: Int, m: Int) -> [Float] {
        let rowCount = n
        let columnCount = m
        
        var outputMatrix = [Float](repeating: 0, count: rowCount * columnCount)
        
        // 使用Accelerate中的转置函数进行计算
        vDSP_mtrans(inputMatrix, 1, &outputMatrix, 1, UInt(columnCount), UInt(rowCount))

        return outputMatrix
    }
    
    static func matrixMultiply(a: [[Float]], b: [[Float]]) -> [[Float]] {
        let rowCount = a.count
        let columnCount = b[0].count
        let innerDimension = b.count
        
        var result = [[Float]](repeating: [Float](repeating: 0, count: columnCount), count: rowCount)
        
        // 将输入矩阵转换为行优先存储的向量
        var vectorA = a.flatMap { $0 }
        var vectorB = b.flatMap { $0 }
        var vectorResult = result.flatMap { $0 }
        
        // 使用Accelerate中的矩阵乘法函数进行计算
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(rowCount), Int32(columnCount), Int32(innerDimension), 1.0, &vectorA, Int32(innerDimension), &vectorB, Int32(columnCount), 0.0, &vectorResult, Int32(columnCount))
        
        // 将结果向量转换回矩阵
        for i in 0..<rowCount {
            result[i] = Array(vectorResult[i*columnCount..<i*columnCount+columnCount])
        }
        
        return result
    }
    
    static func row_norms(_ X: inout [[Float]], n: Int, m: Int) -> [Float] {
        var sumOfSquaresL2 = [Float](repeating: 0.0, count: n)
        for i in 0..<n {
            vDSP_svesq(X[i], 1, &sumOfSquaresL2[i], vDSP_Length(m))
        }
        
        var count = Int32(n)
        var result = [Float](repeating: 0.0, count: n)
        vvsqrtf(&result, &sumOfSquaresL2, &count)
        return result
    }

    static func _handle_zeros_in_scale(_ scale: inout [Float]) {
        let eps = 10 * Float.ulpOfOne
        for i in 0..<scale.count {
            if scale[i] < eps {
                scale[i] = 1.0
            }
        }
    }

    static func normalize(_ X: inout [[Float]], n: Int, m: Int) -> [[Float]]{
        
        var norms = row_norms(&X, n: n, m: m)
        _handle_zeros_in_scale(&norms)
        
        
        var X_normalized:[[Float]] = [[Float]](repeating: Array(repeating: 0, count: m), count: n)
        
        for i in 0..<X.count {
            let normV = norms[i]
            for j in 0..<X[0].count {
                X_normalized[i][j] = X[i][j] / normV
            }
        }

        return X_normalized
    }

    static func cosine_similarity(_ X: inout [[Float]], _ n: Int, _ d: Int) -> [[Float]] {
        let X_normalized = normalize(&X, n: n, m: d)
        let Y_normalized = transposeMatrix(X_normalized)
        let ret = matrixMultiply(a: X_normalized, b: Y_normalized)
        return ret
    }

    static func cosine_distances(_ X: inout [[Float]], _ n: Int, _ d: Int) -> [[Float]] {
        let S = cosine_similarity(&X, n, d)
        var S_flat = S.flatMap { $0 }
        
        var scale: Float = -1.0
        var oneVec: Float = 1.0
        vDSP_vsmul(S_flat, 1, &scale, &S_flat, 1, vDSP_Length(S_flat.count))
        vDSP_vsadd(S_flat, 1, &oneVec, &S_flat, 1, vDSP_Length(S_flat.count))
        
        var zero: Float = 0.0
        var two: Float = 2.0
        vDSP_vclip(S_flat, 1, &zero, &two, &S_flat, 1, vDSP_Length(n * n))
        
        var S_modified = stride(from: 0, to: S_flat.count, by: n).map {
            Array(S_flat[$0..<Swift.min($0+n, S_flat.count)])
        }
        
        for i in 0..<n {
            S_modified[i][i] = 0
        }
        
//        var diagonal = [Float](repeating: 0.0, count: Swift.min(S.count, S[0].count))
//        var diagonalValue:Float = 0
//        vDSP_vfill(&diagonal, &diagonalValue, 1, vDSP_Length(diagonal.count))
//        for i in stride(from: 0, to: Swift.min(S.count, S[0].count), by: 1) {
//            S_modified[i][i] = diagonal[i]
//        }
//        
        return S_modified
    }

    static func pairwise_distances(_ X: [[Float]]) -> [[Float]] {
        let N = X.count
        let M = X[0].count
        var copyX = X
        let distances = cosine_distances(&copyX, N, M)
        return distances
    }
    
    
    
    
    static func agglomerativeClustering(_ X: [[Float]], _ minK: Int, _ maxK: Int) -> [Int] {
        
        var dist = [Float](repeating: 0.0, count: (X.count * (X.count - 1)) >> 1 )
        var distNum = 0
        for row in 0..<X.count {
            for row2 in 0..<X.count {
                
                guard row < row2 else{
                    continue
                }
                
                let X_x:[Float] = X[row]
                let X_y:[Float] = X[row2]
                var distance: Float = 0
                vDSP_distancesq(X_x, 1, X_y, 1, &distance, vDSP_Length(X.count));
                distance = sqrtf(distance)
                dist[distNum] = distance
                distNum+=1
            }
        }
        
        

        var labels = [Int32](repeating: 1000, count: X.count)
        let xxx = AggClusteringWrapper2()

        dist.withUnsafeMutableBufferPointer({ (cccc:inout UnsafeMutableBufferPointer<Float>) in
            let dataPtr: UnsafeMutablePointer<Float> = cccc.baseAddress!
            labels.withUnsafeMutableBufferPointer { (dddd:inout UnsafeMutableBufferPointer<Int32>) in
               let labelsPtr = dddd.baseAddress
               xxx.agglomerativeClustering(dataPtr, row: Int32(X.count),minClusterNum: Int32(minK), maxClusterNum: Int32(maxK), labels: labelsPtr)
            }
        })
        return labels.compactMap { Int($0) }
    }
    
    static func convertToUnsafeMutablePointer(_ array: [[Float]]) -> UnsafeMutablePointer<UnsafeMutablePointer<Float>?>! {
        let numColumns = array[0].count
        let numRows = array.count
        
        // 创建 UnsafeMutablePointer<UnsafeMutablePointer<Float>?>，大小为 numRows * sizeof(UnsafeMutablePointer<Float>?)
        let pointer = UnsafeMutablePointer<UnsafeMutablePointer<Float>?>.allocate(capacity: numRows)
        // 为每个 UnsafeMutablePointer<Float> 分配空间，大小为 numColumns * sizeof(Float)
        for i in 0..<numRows {
            pointer[i] = UnsafeMutablePointer<Float>.allocate(capacity: numColumns)
            // 将数据从 array 复制到 UnsafeMutablePointer<Float>
            for j in 0..<numColumns {
                pointer[i]![j] = array[i][j]
            }
        }
        return pointer
    }
    
    static func convertToIntPointer(_ array: [Int]) -> UnsafeMutablePointer<Int32> {
        let pointer = UnsafeMutablePointer<Int32>.allocate(capacity: array.count)
        for i in 0..<array.count {
            pointer[i] = Int32(array[i])
        }
        return pointer
    }

    
    // 计算 Silhouette score
//    static func silhouetteScore(_ samples: [[Float]], _ labels: [Int], _ k: Int) -> Float {
//        let count = samples.count
//
//        let samplesPtr: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>! = convertToUnsafeMutablePointer(samples)
//        let labelsPtr: UnsafeMutablePointer<Int32> = convertToIntPointer(labels)
//        
//        let mSilhouetteScoreWrapper = SilhouetteScoreWrapper()
//        let socre = mSilhouetteScoreWrapper.score(samplesPtr, labels: labelsPtr, itemNum: Int32(count), cluster: Int32(k))
//        for index in 0..<samples.count {
//            samplesPtr[index]?.deallocate()
//        }
//        samplesPtr.deallocate()
//        labelsPtr.deallocate()
//        return socre
//        
////        precondition(samples.count == labels.count, "The number of samples and labels must be the same")
////        let uniqueLabels = Set(labels)
////        var clusters = [[[Float]]](repeating: [], count: uniqueLabels.count)
////        for i in 0..<labels.count {
////            let sss:[Float] = samples[i]
////            clusters[labels[i]].append(sss)
////        }
////        var score:Float = 0.0
////        for i in 0..<samples.count {
////            let sample = samples[i]
////            let label = labels[i]
////            let a = meanIntraClusterDistance(sample, clusters[label])
////            let b = meanInterClusterDistance(sample, clusters.filter { $0 != clusters[label] })
////            score += (b - a) / max(a, b)
////        }
////        return score / Float(samples.count)
//        
//        
////        samples.withUnsafeMutableBufferPointer { (aaa: inout UnsafeMutableBufferPointer<[Float]>) in
////
////        }
//        
////        let distancePtr = UnsafeMutablePointer<Float>(samples)
//        
////        SilhouetteScoreWrapper.score(UnsafeMutablePointer<UnsafeMutablePointer<Float>?>!, labels: <#T##UnsafeMutablePointer<Int32>!#>, itemNum: <#T##Int32#>)
//    }
    
}

