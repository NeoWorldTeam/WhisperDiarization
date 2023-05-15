//
//  AgglHierClusterer.hpp
//  WhisperDiarization
//
//  Created by fuhao on 2023/5/6.
//

#ifndef AgglHierClusterer_hpp
#define AgglHierClusterer_hpp


#include <map>
#include <string>
#include <vector>
#include <cstdio>
#include <assert.h>

#include "ClusterNode.hpp"
#include "DistanceCalculator.hpp"

namespace CSAlgorithm  {
class AggClustering {
private:
    int node_num_;  // util now how many cluster node have been created

    int base_node_num_;  // num of nodes who has no child

    float *distance_matrix_;

    ClusterNode *cluster_node_array_;

    // The cluster name to cluster label_ mapping
    std::map<std::string, int> node_name_map_;

    const DistanceCalculator * distance_calculator_;

private:
    std::vector<std::string> splitString(std::string input,
            std::string seg_pattern);

    int findNearestNeighbor(const ClusterNode &node, float *distance);

    inline size_t getDistanceMatrixIndex(const ClusterNode &left_node,
            const ClusterNode &right_node) {
        size_t left_dis_label = left_node.getDistanceMatrixLabel();
        size_t right_dis_label = right_node.getDistanceMatrixLabel();
        return getDistanceMatrixIndex(left_dis_label, right_dis_label);
    }

    inline size_t getDistanceMatrixIndex(size_t left_dis_label,
            size_t right_dis_label) {
        assert(left_dis_label != right_dis_label);
        if (left_dis_label > right_dis_label) {
            return ((left_dis_label * (left_dis_label - 1)) >> 1)
                    + right_dis_label;
        } else {
            return ((right_dis_label * (right_dis_label - 1)) >> 1)
                    + left_dis_label;
        }
    }

    //  aggregate two nodes; return the label of new node
    int aggregate(ClusterNode* left_node,
            ClusterNode* right_node,
            float distance);

public:
    int getBaseNodeNum() const {
        return base_node_num_;
    }

    void setBaseNodeNum(int base_node_num) {
        base_node_num_ = base_node_num;
    }

    float *getDistanceMatrix() const {
        return distance_matrix_;
    }

    void setDistanceMatrix(float *distanceMatrix) {
        distanceMatrix = distanceMatrix;
    }

    ClusterNode *getClusterNodeArray() const {
        return cluster_node_array_;
    }

    void setClusterNodeArray(ClusterNode *clusterNodeArray) {
        clusterNodeArray = clusterNodeArray;
    }

    std::map<std::string, int> const &getNodeNameMap() const {
        return node_name_map_;
    }

    void setNodeNameMap(std::map<std::string, int> const &node_name_map) {
        node_name_map_ = node_name_map;
    }

public:
    AggClustering(DistanceCalculatorType::Type type = DistanceCalculatorType::WARD):  // NOLINT
            node_num_(0),
            base_node_num_(0),
            distance_matrix_(NULL),
            cluster_node_array_(NULL) {
        distance_calculator_ =
                DistanceCalculatorFactory::createCalculator(type);
    }

    ~AggClustering();

    
    bool init(float* distance_matrix, int node_num);


    bool loadDistanceMatrix(float *distance_matrix,
        int node_num);

    bool doCluster();


    bool output(float distance_threshold, int* labels);
    bool cutTree(int minK, int maxK, int* labels) ;
};
}  // namespace CSAlgorithm 



#endif /* AgglHierClusterer_hpp */
