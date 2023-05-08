//
//  silhouette_score.cpp
//  WhisperDiarization
//
//  Created by fuhao on 2023/5/8.
//


#include <stdio.h>
#include <string>
#include "silhouette_score.hpp"
#include <math.h>

float silhouetteScore(std::vector<std::vector<float>> distances, std::vector<int> y,int numClusters)
{
    std::vector<float> intraClusters(distances.size(), 0.0); // Computing intraclusters distances of each point
    std::vector<float> crossClusters(distances.size()); // Minimum Distance of each point to other clusters
    for (int i = 0; i < distances.size(); i++)
    {
        int sumNum = 0;
        std::vector<float> interClusters(numClusters,0.0); // values of point to each cluster points
        std::vector<int> sumsOfParticular(numClusters, 0);
        for (int j = 0; j < distances.size(); j++)
        {
            if (y[j] == y[i])
            {
                // Sum of distance of point to each
                 // other point in same cluster
                intraClusters[i] += distances[i][j];
                sumNum++;
            }
            else
            {
                // Sum of distance of point to
                // points in different clusters
                interClusters[y[j]] += distances[i][j];
                sumsOfParticular[y[j]]++; // computes points in that cluster
            }
        }
        // Mean of sum values of distances b/w
        // points of same cluster
        intraClusters[i] /= sumNum;
        float minimumOfall = std::numeric_limits<float>::max();
        
        for (int j = 0; j < numClusters; j++)
        {
            if (j != y[i])
            {
                interClusters[j] /=
                    sumsOfParticular[j]; // Mean of values of interclusters
                                         // distances
                if (interClusters[j] < minimumOfall)
                { // computing minimum value of means of intercluster distances
                    minimumOfall = interClusters[j];
                }
            }
        }
        crossClusters[i] = minimumOfall;
    }
    float si = 0.0;
    for (int i = 0; i < distances.size(); i++)
    {
        si += ((crossClusters[i] - intraClusters[i]) / std::max(intraClusters[i], crossClusters[i])); // s = b[i]-a[i] / max(b[i],a[i])
    }
    return si / float(distances.size());
};
