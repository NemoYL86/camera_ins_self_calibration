/*
* All Rights Reserved
* Author: yanglei
* Date: 下午5:51
*/
#ifndef FEATURES_FEATURE_TRACKER_H
#define FEATURES_FEATURE_TRACKER_H

#include "base/data_typedef.h"
#include <cstdio>
#include <iostream>
#include <queue>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

namespace hdmap {
namespace ddi {


class FeatureTracker {
public:
    static constexpr double FOCAL_LENGTH = 400.0L;
    static constexpr size_t MAX_CNT = 2000;
    static constexpr size_t MIN_DIST = 25;

    FeatureTracker();

    ~FeatureTracker()
    {};

    std::vector<Feature::Ptr> TrackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat(), bool isShow = false);

    void SetIntrinsicParam(cv::Mat kMat);

    std::vector<IndMatch> GetMatches()
    { return matches_; }

    cv::Mat getTrackImage();


private:
    void ExtractMatches();

    void setMask();

    void rejectWithF();

    std::vector<cv::Point2f> undistortedPts(std::vector<cv::Point2f> &pts);

    void showTwoImage(const cv::Mat &img1, const cv::Mat &img2,
                      std::vector<cv::Point2f> pts1, std::vector<cv::Point2f> pts2);

    void drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight,
                   std::vector<size_t> &curLeftIds,
                   std::vector<cv::Point2f> &curLeftPts,
                   std::vector<cv::Point2f> &curRightPts,
                   std::map<size_t, cv::Point2f> &prevLeftPtsMap);

    double distance(cv::Point2f &pt1, cv::Point2f &pt2);

    void removeOutliers(std::set<size_t> &removePtsIds);

    void setPrediction(std::map<size_t, Eigen::Vector3d> &predictPts);


private:
    std::vector<Feature::Ptr> curFeatures_;
    std::vector<Feature::Ptr> prevFeatures_;
    std::vector<IndMatch> matches_;

    size_t row_, col_;
    cv::Mat imTrack;
    cv::Mat mask_;

    cv::Mat prevImg_, curImg_;
    std::vector<cv::Point2f> newPts_;
    std::vector<cv::Point2f> predictPts_;
    std::vector<cv::Point2f> prevPts_, curPts_, curRightPts_;
    std::vector<cv::Point2f> curUnPts_, curUnRightPts_;
    std::vector<size_t> ids_, idsRight_;
    std::vector<size_t> trackedCnt_;
    std::map<size_t, cv::Point2f> curUnRightPtsMap_;
    std::map<size_t, cv::Point2f> prevLeftPtsMap;

    Eigen::Matrix3d kMat_, kMatInv_;

    double curTime_ = 0;
    double prevTime_ = 0;
    size_t landmarkNum_ = 0;
    size_t frameId_ = 0;
    bool hasPrediction_ = false;
    bool useStereoMatch_ = false;
};

}
}

#endif //VINS_FEATURE_TRACKER_H
