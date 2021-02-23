/*
* All Rights Reserved
* Author: yanglei
* Date: 上午11:45
*/
#ifndef VINS_GMS_REJECTOR_H
#define VINS_GMS_REJECTOR_H

#include "base/data_typedef.h"
#include <array>
#include <map>
#include <utility>
#include <vector>
#include "opencv2/core.hpp"

namespace hdmap {
namespace ddi {

class GMSRejector {
public:
    // OpenCV Keypoints & Correspond Image Size & Nearest Neighbor Matches
    GMSRejector(const std::vector<Feature::Ptr> &vkp1, const cv::Size &size1,
                const std::vector<Feature::Ptr> &vkp2, const cv::Size &size2,
                const std::vector<IndMatch> &vMatches, const double thresholdFactor)
            : mThresholdFactor(thresholdFactor)
    {
        // Input initialize
        normalizePoints(vkp1, size1, mvP1);
        normalizePoints(vkp2, size2, mvP2);
        mNumberMatches = vMatches.size();
        convertMatches(vMatches, mvMatches);

        // Grid initialize
        mGridSizeLeft = cv::Size(20, 20);
        mGridNumberLeft = mGridSizeLeft.width * mGridSizeLeft.height;

        // Initialize the neighbor of left grid
        mGridNeighborLeft = cv::Mat::zeros(mGridNumberLeft, 9, CV_32SC1);
        initalizeNeighbors(mGridNeighborLeft, mGridSizeLeft);
    }

    ~GMSRejector()
    {}

    // Get Inlier Mask
    // Return number of inliers
    int getInlierMask(std::vector<bool> &vbInliers, const bool withRotation = false, const bool withScale = false);


private:
    // Normalized Points
    std::vector<cv::Point2f> mvP1, mvP2;

    // Matches
    std::vector<std::pair<int, int>> mvMatches;

    // Number of Matches
    size_t mNumberMatches;

    // Grid Size
    cv::Size mGridSizeLeft, mGridSizeRight;
    int mGridNumberLeft;
    int mGridNumberRight;

    // x      : left grid idx
    // y      : right grid idx
    // value  : how many matches from idx_left to idx_right
    cv::Mat mMotionStatistics;

    //
    std::vector<int> mNumberPointsInPerCellLeft;

    // Inldex  : grid_idx_left
    // Value   : grid_idx_right
    std::vector<int> mCellPairs;

    // Every Matches has a cell-pair
    // first  : grid_idx_left
    // second : grid_idx_right
    std::vector<std::pair<int, int>> mvMatchPairs;

    // Inlier Mask for output
    std::vector<bool> mvbInlierMask;

    //
    cv::Mat mGridNeighborLeft;
    cv::Mat mGridNeighborRight;

    double mThresholdFactor = 6.0;

    // Assign Matches to Cell Pairs
    void assignMatchPairs(const int GridType);

    void convertMatches(const std::vector<IndMatch> &vMatches, std::vector<std::pair<int, int>> &matches);

    int getGridIndexLeft(const cv::Point2f &pt, const int type);

    int getGridIndexRight(const cv::Point2f &pt);

    std::vector<int> getNB9(const int idx, const cv::Size &GridSize);

    void initalizeNeighbors(cv::Mat &neighbor, const cv::Size &GridSize);

    void normalizePoints(const std::vector<Feature::Ptr> &kp, const cv::Size &size, std::vector<cv::Point2f> &npts);

    // Run
    int run(const int rotationType);

    void setScale(const int scale);

    // Verify Cell Pairs
    void verifyCellPairs(const int rotationType);
};

}
}

#endif //VINS_GMS_REJECTOR_H
