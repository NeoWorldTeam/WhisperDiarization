//
//  silhouette_score.hpp
//  WhisperDiarization
//
//  Created by fuhao on 2023/5/8.
//

#ifndef silhouette_score_h
#define silhouette_score_h
#include <vector>


float silhouetteScore(std::vector<std::vector<float>> distances, std::vector<int> y,int numClusters);


#endif /* silhouette_score_h */
