/*
* All Rights Reserved
* Author: yanglei
* Date: 下午5:51
*/
#include "feature_tracker.h"
#include "opencv2/core/eigen.hpp"

namespace hdmap {
namespace ddi {

using namespace std;


static bool inBorder(const cv::Point2f &pt, int row, int col)
{
    const size_t BORDER_SIZE = 1;
    size_t img_x = cvRound(pt.x);
    size_t img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE;
}

static double distance(cv::Point2f pt1, cv::Point2f pt2)
{
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

template<typename T>
static void reduceVector(vector<T> &v, const vector<uint8_t> &status)
{
    size_t j = 0;
    for (size_t i = 0; i < size_t(v.size()); i++) {
        if (status[i])
            v[j++] = v[i];
    }
    v.resize(j);
}

FeatureTracker::FeatureTracker()
{
    useStereoMatch_ = 0;
    landmarkNum_ = 0;
    frameId_ = 0;
    hasPrediction_ = false;
}


void FeatureTracker::SetIntrinsicParam(cv::Mat kMat)
{
    cv::cv2eigen(kMat, kMat_);
    kMatInv_ = kMat_.inverse();
}


void FeatureTracker::setMask()
{
    mask_ = cv::Mat::zeros(row_, col_, CV_8UC1);
    mask_.rowRange(0, row_ * 0.4).setTo(255);

    // prefer to keep features that are tracked for long time
    vector<pair<size_t, pair<cv::Point2f, size_t>>> cnt_pts_id;

    for (size_t i = 0; i < curPts_.size(); i++) {
        cnt_pts_id.push_back(make_pair(trackedCnt_[i], make_pair(curPts_[i], ids_[i])));
    }

    sort(cnt_pts_id.begin(), cnt_pts_id.end(),
         [](const pair<size_t, pair<cv::Point2f, size_t>> &a, const pair<size_t, pair<cv::Point2f, size_t>> &b)
         {
             return a.first > b.first;
         });

    curPts_.clear();
    ids_.clear();
    trackedCnt_.clear();

    for (auto &it : cnt_pts_id) {
        if (mask_.at<uchar>(it.second.first) == 255) {
            curPts_.push_back(it.second.first);
            ids_.push_back(it.second.second);
            trackedCnt_.push_back(it.first);
            cv::circle(mask_, it.second.first, MIN_DIST, 0, -1);
        }
    }
}


double FeatureTracker::distance(cv::Point2f &pt1, cv::Point2f &pt2)
{
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}


std::vector<Feature::Ptr> FeatureTracker::
TrackImage(double timestamp, const cv::Mat &imLeft, const cv::Mat &imRight, bool isShow)
{
    static cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 40, 0.01);

    curTime_ = timestamp;
    curImg_ = imLeft;
    row_ = curImg_.rows;
    col_ = curImg_.cols;
    cv::Mat rightImg = imRight;

    curPts_.clear();

    if (prevPts_.size() > 0) {
        vector<uchar> status;
        vector<float> err;
        if (hasPrediction_) {
            curPts_ = predictPts_;
            cv::calcOpticalFlowPyrLK(prevImg_, curImg_, prevPts_, curPts_, status, err, cv::Size(41, 41), 1,
                                     cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 40, 0.01),
                                     cv::OPTFLOW_USE_INITIAL_FLOW);

            size_t succ_num = 0;
            for (size_t i = 0; i < status.size(); i++) {
                if (status[i])
                    succ_num++;
            }
            if (succ_num < 10) {
                cv::calcOpticalFlowPyrLK(prevImg_, curImg_, prevPts_, curPts_, status, err, cv::Size(41, 41), 3);
            }
        } else {
            cv::calcOpticalFlowPyrLK(prevImg_, curImg_, prevPts_, curPts_, status, err, cv::Size(41, 41), 3);
        }
        cv::cornerSubPix(curImg_, curPts_, cv::Size(3, 3), cv::Size(-1, -1), criteria);

        // reverse check
        if (1) {
            vector<uchar> reverse_status;
            vector<cv::Point2f> reverse_pts = prevPts_;
            cv::calcOpticalFlowPyrLK(curImg_, prevImg_, curPts_, reverse_pts, reverse_status, err, cv::Size(41, 41), 1,
                                     cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 40, 0.01),
                                     cv::OPTFLOW_USE_INITIAL_FLOW);

            cv::cornerSubPix(prevImg_, reverse_pts, cv::Size(3, 3), cv::Size(-1, -1), criteria);

            //cv::calcOpticalFlowPyrLK(curImg_, prevImg_, curPts_, reverse_pts, reverse_status, err, cv::Size(41, 41), 3);
            for (size_t i = 0; i < status.size(); i++) {
                if (status[i] && reverse_status[i] && distance(prevPts_[i], reverse_pts[i]) <= 0.5) {
                    status[i] = 1;
                } else
                    status[i] = 0;
            }
        }

        for (size_t i = 0; i < size_t(curPts_.size()); i++) {
            if (status[i] && !inBorder(curPts_[i], row_, col_))
                status[i] = 0;
        }
        reduceVector(prevPts_, status);
        reduceVector(curPts_, status);
        reduceVector(ids_, status);
        reduceVector(trackedCnt_, status);
    }

    for (auto &n : trackedCnt_) {
        n++;
    }


    if (1) {
        //rejectWithF();
        setMask();
        size_t n_max_cnt = MAX_CNT - curPts_.size();
        if (n_max_cnt > 0) {
            cv::goodFeaturesToTrack(curImg_, newPts_, MAX_CNT - curPts_.size(), 0.01, MIN_DIST, mask_);
        } else {
            newPts_.clear();
        }

        cv::cornerSubPix(curImg_, newPts_, cv::Size(3, 3), cv::Size(-1, -1), criteria);

        for (auto &p : newPts_) {
            curPts_.push_back(p);
            ids_.push_back(landmarkNum_++);
            trackedCnt_.push_back(1);
        }
    }

    curUnPts_ = undistortedPts(curPts_);

    cout << "cur-prev pts: " << curPts_.size() << " / " << prevLeftPtsMap.size() << endl;


    if (!imRight.empty() && useStereoMatch_) {
        idsRight_.clear();
        curRightPts_.clear();
        curUnRightPts_.clear();
        curUnRightPtsMap_.clear();
        if (!curPts_.empty()) {
            vector<cv::Point2f> reverseLeftPts;
            vector<uchar> status, statusRightLeft;
            vector<float> err;
            // cur left ---- cur right
            cv::calcOpticalFlowPyrLK(curImg_, rightImg, curPts_, curRightPts_, status, err, cv::Size(41, 41), 3);

            cv::cornerSubPix(rightImg, curRightPts_, cv::Size(3, 3), cv::Size(-1, -1), criteria);
            // reverse check cur right ---- cur left
            if (1) {
                cv::calcOpticalFlowPyrLK(rightImg, curImg_, curRightPts_, reverseLeftPts, statusRightLeft, err,
                                         cv::Size(41, 41), 3);
                for (size_t i = 0; i < status.size(); i++) {
                    if (status[i] && statusRightLeft[i] && inBorder(curRightPts_[i], row_, col_) &&
                        distance(curPts_[i], reverseLeftPts[i]) <= 0.5)
                        status[i] = 1;
                    else
                        status[i] = 0;
                }
            }

            idsRight_ = ids_;
            reduceVector(curRightPts_, status);
            reduceVector(idsRight_, status);
            // only keep left-right pts
            /*
            reduceVector(curPts_, status);
            reduceVector(ids, status);
            reduceVector(trackedCnt_, status);
            reduceVector(curUnPts_, status);
            reduceVector(pts_velocity, status);
            */
            curUnRightPts_ = undistortedPts(curRightPts_);
        }
    }


    if (isShow) {
        drawTrack(curImg_, rightImg, ids_, curPts_, curRightPts_, prevLeftPtsMap);
    }


    prevLeftPtsMap.clear();
    for (size_t i = 0; i < curPts_.size(); i++) {
        prevLeftPtsMap[ids_[i]] = curPts_[i];
    }

    curFeatures_.clear();
    for (size_t i = 0; i < ids_.size(); i++) {
        Feature::Ptr feature(new Feature);
        feature->landmarkId_ = ids_[i];
        feature->cameraId_ = 0;
        feature->frameId_ = frameId_;
        feature->uv_ << curPts_[i].x, curPts_[i].y;
        feature->xyz_ << curUnPts_[i].x, curUnPts_[i].y, 1.0;
        curFeatures_.push_back(feature);
    }

    if (!imRight.empty() && useStereoMatch_) {
        for (size_t i = 0; i < idsRight_.size(); i++) {
            Feature::Ptr feature(new Feature);
            feature->landmarkId_ = idsRight_[i];
            feature->cameraId_ = 1;
            feature->frameId_ = frameId_;
            feature->uv_ << curRightPts_[i].x, curRightPts_[i].y;
            feature->xyz_ << curUnRightPts_[i].x, curUnRightPts_[i].y, 1.0;
            curFeatures_.push_back(feature);
        }
    }

    //
    ExtractMatches();

    //
    prevImg_ = curImg_;
    prevPts_ = curPts_;
    prevTime_ = curTime_;
    hasPrediction_ = false;

    return curFeatures_;
}


void FeatureTracker::
ExtractMatches()
{
    matches_.clear();

    if(prevFeatures_.empty()){
        prevFeatures_.assign(curFeatures_.begin(), curFeatures_.end());
        return;
    }

    std::unordered_map<size_t, size_t> featId2LandmarkIdPrev;
    for (size_t i = 0; i < prevFeatures_.size(); i++) {
        featId2LandmarkIdPrev.insert({i, prevFeatures_[i]->landmarkId_});
    }

    std::unordered_map<size_t, size_t> landmarkId2FeatIdCurr;
    for (size_t i = 0; i < curFeatures_.size(); i++) {
        landmarkId2FeatIdCurr.insert({curFeatures_[i]->landmarkId_, i});
    }

    for(auto &prevId : featId2LandmarkIdPrev){
        auto &featI = prevId.first;
        auto it = landmarkId2FeatIdCurr.find(prevId.second);
        if(it != landmarkId2FeatIdCurr.end()){
            auto &featJ = it->second;
            matches_.push_back({featI, featJ});
        }
    }

    prevFeatures_.assign(curFeatures_.begin(), curFeatures_.end());
}


void FeatureTracker::
rejectWithF()
{
    if (curPts_.size() >= 8) {
        vector<cv::Point2f> un_cur_pts(curPts_.size()), un_prev_pts(prevPts_.size());
        for (size_t i = 0; i < curPts_.size(); i++) {
            Eigen::Vector3d tmp_p = kMatInv_ * Eigen::Vector3d(curPts_[i].x, curPts_[i].y, 1.0);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col_ / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row_ / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            tmp_p = kMatInv_ * Eigen::Vector3d(prevPts_[i].x, prevPts_[i].y, 1.0);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col_ / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row_ / 2.0;
            un_prev_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_prev_pts, cv::FM_RANSAC, 3.0, 0.99, status);

        reduceVector(prevPts_, status);
        reduceVector(curPts_, status);
        reduceVector(curUnPts_, status);
        reduceVector(ids_, status);
        reduceVector(trackedCnt_, status);
    }
}


vector<cv::Point2f> FeatureTracker::
undistortedPts(vector<cv::Point2f> &pts)
{
    vector<cv::Point2f> un_pts;
    for (size_t i = 0; i < pts.size(); i++) {
        Eigen::Vector3d a(pts[i].x, pts[i].y, 1.0);
        Eigen::Vector3d b = kMatInv_ * a;
        un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
    }
    return un_pts;
}


void FeatureTracker::
drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight,
          vector<size_t> &curLeftIds,
          vector<cv::Point2f> &curLeftPts,
          vector<cv::Point2f> &curRightPts,
          map<size_t, cv::Point2f> &prevLeftPtsMap)
{
    //size_t rows = imLeft.rows;
    size_t cols = imLeft.cols;
    if (!imRight.empty() && useStereoMatch_)
        cv::hconcat(imLeft, imRight, imTrack);
    else
        imTrack = imLeft.clone();
    cv::cvtColor(imTrack, imTrack, CV_GRAY2RGB);

    for (size_t j = 0; j < curLeftPts.size(); j++) {
        double len = std::min(1.0, 1.0 * trackedCnt_[j] / 20);
        cv::circle(imTrack, curLeftPts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    }
    if (!imRight.empty() && useStereoMatch_) {
        for (size_t i = 0; i < curRightPts.size(); i++) {
            cv::Point2f rightPt = curRightPts[i];
            rightPt.x += cols;
            cv::circle(imTrack, rightPt, 2, cv::Scalar(0, 255, 0), 2);
            //cv::Point2f leftPt = curLeftPtsTrackRight[i];
            //cv::line(imTrack, leftPt, rightPt, cv::Scalar(0, 255, 0), 1, 8, 0);
        }
    }

    map<size_t, cv::Point2f>::iterator mapIt;
    for (size_t i = 0; i < curLeftIds.size(); i++) {
        size_t id = curLeftIds[i];
        mapIt = prevLeftPtsMap.find(id);
        if (mapIt != prevLeftPtsMap.end()) {
            cv::arrowedLine(imTrack, curLeftPts[i], mapIt->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
        }
    }
}


void FeatureTracker::
setPrediction(std::map<size_t, Eigen::Vector3d> &predictPts)
{
    hasPrediction_ = true;
    predictPts_.clear();
    for (size_t i = 0; i < ids_.size(); i++) {
        size_t id = ids_[i];
        auto itPredict = predictPts.find(id);
        if (itPredict != predictPts.end()) {
            Eigen::Vector3d tmp_uv = kMat_ * itPredict->second;
            tmp_uv /= tmp_uv[2];
            predictPts_.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
        } else
            predictPts_.push_back(prevPts_[i]);
    }
}


void FeatureTracker::
removeOutliers(set<size_t> &removePtsIds)
{
    std::set<size_t>::iterator itSet;
    vector<uchar> status;
    for (size_t i = 0; i < ids_.size(); i++) {
        itSet = removePtsIds.find(ids_[i]);
        if (itSet != removePtsIds.end())
            status.push_back(0);
        else
            status.push_back(1);
    }

    reduceVector(prevPts_, status);
    reduceVector(ids_, status);
    reduceVector(trackedCnt_, status);
}


cv::Mat FeatureTracker::getTrackImage()
{
    return imTrack;
}

}
}
