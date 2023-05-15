//
//  AgglHierClusterer.cpp
//  WhisperDiarization
//
//  Created by fuhao on 2023/5/6.
//
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <stack>
#include <queue>
#include "ClusterNode.hpp"
#include "AggClustering.hpp"

namespace CSAlgorithm  {

static const int maxCharBufferSize = 1 << 11;  // input buffer size 8k
static const std::string seg = ",";

std::vector<std::string> AggClustering::splitString(
        std::string input,
        std::string seg_pattern) {
    std::string::size_type pos;
    std::vector<std::string> result;
    input += seg_pattern;
    int size = input.size();
    for (int i = 0; i < size; i++) {
        pos = input.find_first_of(seg_pattern, i);
        if (pos < size) {
            std::string s = input.substr(i, pos - i);
            result.push_back(s);
            i = pos + seg_pattern.size() - 1;
        }
    }
    return result;
}

int AggClustering::findNearestNeighbor(
        const ClusterNode &node,
        float *distance) {
    int distance_matrix_label = node.getDistanceMatrixLabel();
    assert(distance_matrix_label >= 0);
    float min_distance = -1.0;  // it is not setted
    int min_distance_node_label = -1;
    for (int i = 0; i < node_num_; ++i) {
        if (cluster_node_array_[i].getDistanceMatrixLabel() < 0 ||
                cluster_node_array_[i].getLabel() == node.getLabel()) {
            continue;
        }

        
        int distance_index = getDistanceMatrixIndex(
                distance_matrix_label,
                cluster_node_array_[i].getDistanceMatrixLabel());
        if (min_distance < 0.0
                || distance_matrix_[distance_index] < min_distance) {
            min_distance = distance_matrix_[distance_index];
            min_distance_node_label = i;
        }
    }
    
    
    *distance = min_distance;
    return min_distance_node_label;
}


bool AggClustering::init(float* dists, int node_num) {
    assert(node_num > 1);
    //iterate print the dists
//    for (int i = 0; i < node_num; ++i) {
//        for (int j = 0; j < node_num; ++j) {
//            printf("dists[%d][%d] = %f\n", i, j, dists[i * node_num + j]);
//        }
//    }

    base_node_num_ = node_num;
    cluster_node_array_ =
            new(std::nothrow) ClusterNode[(base_node_num_ << 1) -1];
    if (cluster_node_array_ == NULL) {
        return false;
    }

    // then create the matrix of the distance
    size_t distanceEdgeNum =
            (size_t(base_node_num_) * size_t(base_node_num_ - 1)) >> 1;

    distance_matrix_ = new float[distanceEdgeNum];

    if (distance_matrix_ == NULL) {
        return false;
    }

    memset(distance_matrix_, 0, sizeof(float) * distanceEdgeNum);
    // load distance matrix from file
    if (!loadDistanceMatrix(dists, node_num)) {
        fprintf(stderr, "Init of clusterer failed,"
                " cannot load distance matrix correctly\n");
        
        
        
        return false;
    } else {
        fprintf(stderr, "Init success!\n");
    }
    node_num_ = base_node_num_;
    
//    //iterate print the cluster_node_array_
//    for (int i = 0; i < node_num_; ++i) {
//        printf("cluster_node_array_[%d].getLabel() = %d\n",
//                i, cluster_node_array_[i].getLabel());
//    }
//    //print the distance_matrix_
//    for(int i = 0; i < distanceEdgeNum; ++i) {
//        printf("distance_matrix_[%d] = %f\n", i, distance_matrix_[i]);
//    }
    return true;
}


AggClustering::~AggClustering() {
    delete distance_calculator_;
    delete[] cluster_node_array_;
    delete[] distance_matrix_;
}


bool AggClustering::loadDistanceMatrix(float *dists,
        int node_num) {
//    size_t expected_pair_num =
//            ((size_t)base_node_num_ * (size_t)(base_node_num_ -1)) >> 1;

//    char input_buffer[maxCharBufferSize] = {0};
    int loaded_node_num = 0;
    size_t loaded_pair_num = 0;

    
    int index_record = 0;
    for (int r = 0; r < node_num; r++) {
        for (int c = 0; c < node_num; c++) {
            if (c <= r)
                continue;
            
            //print value
//            printf("dists[%d][%d] = %f\n", r, c, dists[r * node_num + c]);

            
            
            int left_label = r;
            int right_label = c;

            float distance = dists[index_record++];


            std::string left_label_str = "label_"  + std::to_string(left_label);
            std::string right_label_str = "label_"  + std::to_string(right_label);
            

            
            if (cluster_node_array_[left_label].getLabel() < 0) {
                cluster_node_array_[left_label].init(
                        left_label, left_label, left_label_str);
            }

            if (cluster_node_array_[right_label].getLabel() < 0) {
                cluster_node_array_[right_label].init(
                        right_label, right_label, right_label_str);
            }
            size_t index = getDistanceMatrixIndex(left_label, right_label);

            if (distance_matrix_[index] <= 0.0) {
                distance_matrix_[index] = distance;
                //print distance_matrix_ item
//                printf("distance_matrix_[%d] = %f\n", index, distance_matrix_[index]);
                
                
                loaded_pair_num++;
                if (loaded_pair_num % 1000000 == 0) {
                    fprintf(stderr, "%lu pairs loaded\n", loaded_pair_num);
                }
            }
        }
    }


//    if (loaded_node_num != base_node_num_ ||
//            loaded_pair_num != expected_pair_num) {
//        fprintf(stderr, "Load %d nodes and %lu pairs, load error!\n",
//                loaded_node_num, loaded_pair_num);
//        return false;
//    }
    fprintf(stderr, "Load file success! %d nodes and %lu pairs loaded\n",
            loaded_node_num, loaded_pair_num);
    return true;
}

/*
* Aggregate all the node's into one
*/
bool AggClustering::doCluster() {
    std::stack<int> nearest_neighbor_chain;
    nearest_neighbor_chain.push(0);

    // Current end of the nearest neighbor chain
    ClusterNode* top_node = &cluster_node_array_[nearest_neighbor_chain.top()];

    float nearest_distance = 0.0;
    // The node which ready to add to the nearest neighbor chain
    int nearest_neighbor_label =
            findNearestNeighbor(*top_node, &nearest_distance);
    ClusterNode* nearest_neighbor = &cluster_node_array_[nearest_neighbor_label];

    while (!nearest_neighbor_chain.empty()) {
        if (node_num_ == (base_node_num_ << 1) - 1) {
            fprintf(stderr, "Agglometive hierical cluster success!\n");
            break;
        }
        float next_nearest_distance = 0.0;
        // The next node of the nearest neighbor chain
        int next_node_label = findNearestNeighbor(*nearest_neighbor,
                &next_nearest_distance);

        if (next_node_label == top_node->getLabel()) {
            // aggregate the top node and it's nearest neighbor
            nearest_neighbor_chain.pop();
            int new_node_label = aggregate(top_node,
                    nearest_neighbor,
                    nearest_distance);

            if (nearest_neighbor_chain.empty()) {
                nearest_neighbor_chain.push(new_node_label);
            }
            // then find the nearest neighbor of the top;
            // update the loop's condition
            if (nearest_neighbor_chain.size() == 1) {
                top_node = &cluster_node_array_[nearest_neighbor_chain.top()];
                nearest_neighbor_label = findNearestNeighbor(*top_node,
                                                             &nearest_distance);
                
            } else {
                // length of current nearest neighbor chain is bigger than 2
                nearest_neighbor_label = nearest_neighbor_chain.top();
                nearest_neighbor_chain.pop();
                top_node = &cluster_node_array_[nearest_neighbor_chain.top()];
                
                
                size_t xxxxIndex = getDistanceMatrixIndex(
                                                       top_node->getDistanceMatrixLabel(),
                                                          cluster_node_array_[nearest_neighbor_label].getDistanceMatrixLabel()
                                                       );
                nearest_distance = distance_matrix_[xxxxIndex];
            }
        } else {  // push the next node in to stack and goto next
            nearest_neighbor_chain.push(nearest_neighbor_label);
            top_node = nearest_neighbor;
            nearest_neighbor_label = next_node_label;
            nearest_distance = next_nearest_distance;
        }
        nearest_neighbor = &cluster_node_array_[nearest_neighbor_label];
    }
    return true;
}

/*
* Aggregate two cluster nodes to a new node.
* The new node will be pushed into the cluster node array.
* And then the distance matrix will be updated
*/
int AggClustering::aggregate(
        ClusterNode *left_node,
        ClusterNode *right_node,
        float distance) {
    assert(left_node != NULL && right_node != NULL);
    // new node
    int left_node_dis_label = left_node->getDistanceMatrixLabel();
    int right_node_dis_label = right_node->getDistanceMatrixLabel();
    left_node->setDistanceMatrixLabel(-1);
    right_node->setDistanceMatrixLabel(-1);
    int new_node_dis_label =
            left_node_dis_label < right_node_dis_label ?
                    left_node_dis_label : right_node_dis_label;
    cluster_node_array_[node_num_].setLabel(node_num_);
    cluster_node_array_[node_num_].setBasicNodeNum(
            left_node->getBasicNodeNum() + right_node->getBasicNodeNum());
    cluster_node_array_[node_num_].setLeftChildLabel(left_node->getLabel());
    cluster_node_array_[node_num_].setRightChildLabel(right_node->getLabel());  // NOLINT
    cluster_node_array_[node_num_].setDistanceMatrixLabel(new_node_dis_label);  // NOLINT
    bool xxx = abs(distance) < 0.00001;
    cluster_node_array_[node_num_].setDistance(distance);
    
    
    
    
    // then update the distance matrix
    int cur_dis_label = -1;
    float cur_left_dis = 0.0;
    float cur_right_dis = 0.0;
    float cur_new_dis = 0.0;
    for (int i = 0; i < node_num_; ++ i) {
        cur_dis_label = cluster_node_array_[i].getDistanceMatrixLabel();
        if (cur_dis_label < 0) {  // current node has been aggregated;
            continue;
        }
        cur_left_dis = distance_matrix_[getDistanceMatrixIndex(
                left_node_dis_label, cur_dis_label)];
        cur_right_dis = distance_matrix_[getDistanceMatrixIndex(
                right_node_dis_label, cur_dis_label)];
        cur_new_dis = (*distance_calculator_)(left_node->getBasicNodeNum(),
                right_node->getBasicNodeNum(),
                cluster_node_array_[i].getBasicNodeNum(),
                cur_left_dis,
                cur_right_dis,
                distance
        );
        distance_matrix_[getDistanceMatrixIndex(cur_dis_label, new_node_dis_label)] = cur_new_dis;  // NOLINT
    }

    // update distance matrix end; merge complete
    node_num_ ++;
    return cluster_node_array_[node_num_ - 1].getLabel();
}


bool AggClustering::output(float distance_threshold,int* labels) {
    
//    //print cluster_node_array_ children
//    for (int i = node_num_ - 1, j = 0; i >= 0; -- i, ++ j) {
//        ClusterNode & cur_node = cluster_node_array_[i];
//        printf("node %d: label: %d, left: %d, right: %d, distance: %f\n",
//                j, cur_node.getLabel(), cur_node.getLeftChildLabel(),
//                cur_node.getRightChildLabel(), cur_node.getDistance());
//    }
    
    
    int clusterLabel = 0;
    
    std::queue<int> cluster_nodes_queue;
    cluster_nodes_queue.push(node_num_ - 1);
    int clusterNum = 1;
    
    int cur_label = -1;
    while (! cluster_nodes_queue.empty()) {
        cur_label = cluster_nodes_queue.front();
        cluster_nodes_queue.pop();

        assert(cur_label >= 0 && cur_label < node_num_);

        ClusterNode& cur_cluster_node = cluster_node_array_[cur_label];
        if (cur_cluster_node.getLeftChildLabel() < 0 &&
                cur_cluster_node.getRightChildLabel() < 0) {
            // leaf nodes
            labels[cur_cluster_node.getLabel()] = clusterLabel;
            
            if (clusterLabel < 1) {
                clusterLabel++;
            }
        } else if (clusterNum < 2 && cur_cluster_node.getDistance() > 0.3) {
            cluster_nodes_queue.push(cur_cluster_node.getLeftChildLabel());
            cluster_nodes_queue.push(cur_cluster_node.getRightChildLabel());
            clusterNum++;
        } else {
            // output all the children of the nodes
            std::queue<int> out_queue;
            std::vector<int> out_labels;
            out_queue.push(cur_label);
            int cur_out_label = -1;
            while (!out_queue.empty()) {
                cur_out_label = out_queue.front();
                out_queue.pop();
                ClusterNode & cur_out_node = cluster_node_array_[cur_out_label];
                if (cur_out_node.getLeftChildLabel() < 0 &&
                        cur_out_node.getRightChildLabel() < 0) {
                    out_labels.push_back(cur_out_label);
                } else {
                    // not a leaf node
                    out_queue.push(cur_out_node.getLeftChildLabel());
                    out_queue.push(cur_out_node.getRightChildLabel());
                }
            }
//            fprintf(out_file, "%lu", out_labels.size());
            for (std::vector<int>::const_iterator it = out_labels.begin();
                    it != out_labels.end(); ++ it) {
                labels[cluster_node_array_[(*it)].getLabel()] = clusterLabel;
            }
            if (!out_labels.empty()){
                if (clusterLabel < 1) {
                    clusterLabel++;
                }
            }
        }
    }
    return true;
}


bool AggClustering::cutTree(int minK, int maxK, int* labels) {
    
    float distance_n = -1;
    for (int i = node_num_ - 1, j = 0; i >= 0; -- i, ++ j) {
        ClusterNode & cur_node = cluster_node_array_[i];
        if (distance_n < 0) {
            distance_n = cur_node.getDistance();
        }
        
        printf("node %d: label: %d, left: %d, right: %d, distance: %f\n",
                j, cur_node.getLabel(), cur_node.getLeftChildLabel(),
                cur_node.getRightChildLabel(), cur_node.getDistance());
    }
    
    float threshold_dis = distance_n * 0.5;

    
    int clusterLabel = 0;
    
    std::queue<int> cluster_nodes_queue;
    cluster_nodes_queue.push(node_num_ - 1);
    int clusterNum = 1;
    
    int cur_label = -1;
    while (! cluster_nodes_queue.empty()) {
        cur_label = cluster_nodes_queue.front();
        cluster_nodes_queue.pop();

        assert(cur_label >= 0 && cur_label < node_num_);

        ClusterNode& cur_cluster_node = cluster_node_array_[cur_label];
        if (cur_cluster_node.getLeftChildLabel() < 0 &&
                cur_cluster_node.getRightChildLabel() < 0) {
            // leaf nodes
            labels[cur_cluster_node.getLabel()] = clusterLabel;
            
            if (clusterLabel < maxK) {
                clusterLabel++;
            }
        } else if (clusterNum < maxK && (cur_cluster_node.getDistance() > threshold_dis || clusterNum < minK)) {
            cluster_nodes_queue.push(cur_cluster_node.getLeftChildLabel());
            cluster_nodes_queue.push(cur_cluster_node.getRightChildLabel());
            clusterNum++;
        } else {
            // output all the children of the nodes
            std::queue<int> out_queue;
            std::vector<int> out_labels;
            out_queue.push(cur_label);
            int cur_out_label = -1;
            while (!out_queue.empty()) {
                cur_out_label = out_queue.front();
                out_queue.pop();
                ClusterNode & cur_out_node = cluster_node_array_[cur_out_label];
                if (cur_out_node.getLeftChildLabel() < 0 &&
                        cur_out_node.getRightChildLabel() < 0) {
                    out_labels.push_back(cur_out_label);
                } else {
                    // not a leaf node
                    out_queue.push(cur_out_node.getLeftChildLabel());
                    out_queue.push(cur_out_node.getRightChildLabel());
                }
            }
            for (std::vector<int>::const_iterator it = out_labels.begin();
                    it != out_labels.end(); ++ it) {
                labels[cluster_node_array_[(*it)].getLabel()] = clusterLabel;
            }
            if (!out_labels.empty()){
                if (clusterLabel < maxK) {
                    clusterLabel++;
                }
            }
        }
    }
    return true;
}

}  // namespace CSAlgorithm 
