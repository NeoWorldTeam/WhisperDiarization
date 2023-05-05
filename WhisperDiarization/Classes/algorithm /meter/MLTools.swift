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
    
    
    
    
    static func euclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count, "Input vectors must have the same length")
        var sum: Float = 0
        for i in 0..<a.count {
            let diff = a[i] - b[i]
            sum += diff * diff
        }
        return sqrt(sum)
    }

    static func clusterDistance(_ a: Cluster, _ aSize: ClusterSize, _ b: Cluster, _ bSize: ClusterSize) -> Float {
        var sum: Float = 0
        for i in 0..<aSize {
            for j in 0..<bSize {
                sum += euclideanDistance(a[i], b[j])
            }
        }
        return sum / Float(aSize * bSize)
    }

    static func mergeClusters(_ clusters: inout [Cluster], _ clusterSizes: inout [ClusterSize], _ i: Int, _ j: Int) {
        let newClusterSize = clusterSizes[i] + clusterSizes[j]
        var newCluster: Cluster = []
        newCluster.reserveCapacity(newClusterSize)
        newCluster.append(contentsOf: clusters[i])
        newCluster.append(contentsOf: clusters[j])
        clusters.remove(at: j)
        clusters[i] = newCluster
        clusterSizes.remove(at: j)
        clusterSizes[i] = newClusterSize
    }

    static func agglomerativeClustering(_ X: [[Float]], _ k: Int) -> [Int] {
        
//        let row = X.count
//        let column  = X[0].count
//        let flattenedArray = X.flatMap { $0 }
//        let unsafePointer = UnsafePointer<Float>(flattenedArray)
//        let testModule = TestMoudule()
//        
//        var labels = [Int] (repeating: 0, count: row)
//        labels.withUnsafeMutableBytes { (ptr: UnsafeMutableRawBufferPointer) -> Void in
//            let int32Ptr = ptr.baseAddress!.assumingMemoryBound(to: Int32.self)
//            testModule.fit(unsafePointer, row: Int32(row), column: Int32(column), minNumClusters: Int32(k), labels: int32Ptr)
//        }
//        
//        return labels
        
        
        let row = X.count
        let column  = X[0].count
        let itemsPtr: UnsafeMutablePointer<item_t> = UnsafeMutablePointer<item_t>.allocate(capacity: row)
        
        defer {
            itemsPtr.deallocate()
        }
        
        for i in 0..<row {
            let rowArray:[Float] = X[i]
            let rowPointer = UnsafeMutablePointer<Float>(mutating: rowArray)
            var item = item_t()
            item.coord = coord_s(items: rowPointer, dim: Int32(column))
            itemsPtr[i] = item
        }
        
        
        let clusters:UnsafeMutablePointer<cluster_t> =  agglomerate(Int32(row), itemsPtr)
        
        
        

        // 使用内存区域
//        for i in 0..<count {
//            ptr[i] = /* 初始化 item_t 元素 */
//        }
//        let clusters :UnsafeMutablePointer<cluster_t> = agglomerate(<#T##num_items: Int32##Int32#>, UnsafeMutablePointer<item_t>!)
        
        
//        let n = X.count
//        precondition(k <= n, "Number of clusters k must be less than or equal to the number of data points n")
//        var clusters: [Cluster] = []
//        var clusterSizes: [ClusterSize] = []
//        for i in 0..<n {
//            clusters.append([X[i]])
//            clusterSizes.append(1)
//        }
//        while clusters.count > k {
//            var minI = 0
//            var minJ = 1
//            var minDist = clusterDistance(clusters[minI], clusterSizes[minI], clusters[minJ], clusterSizes[minJ])
//            for i in 0..<clusters.count {
//                for j in (i+1)..<clusters.count {
//                    let dist = clusterDistance(clusters[i], clusterSizes[i], clusters[j], clusterSizes[j])
//                    if dist < minDist {
//                        minI = i
//                        minJ = j
//                        minDist = dist
//                    }
//                }
//            }
//            mergeClusters(&clusters, &clusterSizes, minI, minJ)
//        }
//        var labels = Array(repeating: -1, count: n)
//        for i in 0..<k {
//            var start = 0
//            for j in 0..<i {
//                start += clusterSizes[j]
//            }
//            for j in 0..<clusterSizes[i] {
//                labels[start+j] = i
//            }
//        }
//        return labels
        
        return []
        
        
        
    }
    
    
    

    // 计算样本到其所属簇的平均距离
    static func meanIntraClusterDistance(_ sample: [Float], _ cluster: [[Float]]) -> Float {
        var totalDistance:Float = 0.0
        for point in cluster {
            totalDistance += euclideanDistance(sample, point)
        }
        return totalDistance / Float(cluster.count)
    }

    // 计算样本到其他簇的平均距离
    static func meanInterClusterDistance(_ sample: [Float], _ clusters: [[[Float]]]) -> Float {
        var totalDistance:Float = 0.0
        for cluster in clusters {
            var clusterDistance:Float = 0.0
            for point in cluster {
                clusterDistance += euclideanDistance(sample, point)
            }
            totalDistance += clusterDistance / Float(cluster.count)
        }
        return totalDistance / Float(clusters.count)
    }

    // 计算 Silhouette score
    static func silhouetteScore(_ samples: [[Float]], _ labels: [Int]) -> Float {
        precondition(samples.count == labels.count, "The number of samples and labels must be the same")
        let uniqueLabels = Set(labels)
        var clusters = [[[Float]]](repeating: [], count: uniqueLabels.count)
        for i in 0..<labels.count {
            var xxx:[[Float]] = clusters[labels[i]]
            let sss:[Float] = samples[i]
            xxx.append(sss)
        }
        var score:Float = 0.0
        for i in 0..<samples.count {
            let sample = samples[i]
            let label = labels[i]
            let a = meanIntraClusterDistance(sample, clusters[label])
            let b = meanInterClusterDistance(sample, clusters.filter { $0 != clusters[label] })
            score += (b - a) / max(a, b)
        }
        return score / Float(samples.count)
    }
    
    
//    func hierarchicalClustering(_ data: UnsafeMutablePointer<Float>, _ n: Int, _ dim: Int, _ method: Int) -> [Int] {
//        var distances = [Float](repeating: 0.0, count: n * n)
//        var clusters = [Int](0..<n)
//
//        // 计算距离矩阵
//        vDSP_distances(data, 1, data, 1, &distances, vDSP_Length(n), vDSP_Length(dim))
//
//        // 进行层次聚类
//        var linkage = [Int](repeating: 0, count: 2 * (n - 1))
//        let status = vDSP_hierarchical_cluster(&distances, vDSP_Length(n), &linkage, vDSP_Length(n - 1), vDSP_Length(method))
//        guard status == vDSP_Length(n - 1) else { return clusters }
//
//        // 从聚类结果中提取簇的标签
//        var results = [Int](repeating: 0, count: n)
//        for i in 0..<n {
//            results[clusters[i]] = i
//        }
//        for i in stride(from: n - 2, to: -1, by: -1) {
//            let left = linkage[2 * i]
//            let right = linkage[2 * i + 1]
//            let newCluster = min(clusters[left], clusters[right])
//            clusters[left] = newCluster
//            clusters[right] = newCluster
//        }
//
//        return results
//    }

}

