#include <iostream>
#include <stdio.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <cmath>
#include <string>
#include "glog/logging.h"
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include "features/feature_tracker.h"
#include "features/gms_rejector.h"
#include "base/tracklets.h"
#include "base/kitti_parser.h"

using namespace std;
using namespace Eigen;
using namespace hdmap;
using namespace hdmap::ddi;
using kitti::CameraCalibration;


template<typename T>
static void ReduceVector(std::vector<T> &v, const vector<uint8_t> &status)
{
    size_t j = 0;
    for (size_t i = 0; i < size_t(v.size()); i++) {
        if (status[i])
            v[j++] = v[i];
    }
    v.resize(j);
}

static bool RejectByGMS(const std::vector<Feature::Ptr> &features1, cv::Size size1,
                        const std::vector<Feature::Ptr> &features2, cv::Size size2,
                        std::vector<IndMatch> &matches)
{
    GMSRejector gms(features1, size1, features2, size2, matches, 4);
    std::vector<bool> flags;
    int numInliers = gms.getInlierMask(flags);
    if (flags.size() != matches.size()) {
        LOG(FATAL) << "--> ERROR , num_flag != num_matches !!!\n";
        return false;
    }
    std::cout << "flags / matches : " << flags.size() << " / " << matches.size() << " / " << numInliers << std::endl;
    std::vector<IndMatch> filterMatches;
    for (size_t i = 0; i < flags.size(); i++) {
        if (flags[i]) {
            filterMatches.push_back(matches[i]);
        }
    }
    matches.swap(filterMatches);
    return true;
}

static bool RejectByF(const std::vector<Feature::Ptr> &features1,
                      const std::vector<Feature::Ptr> &features2,
                      std::vector<IndMatch> &matches)
{
    std::vector<cv::Point2f> points1, points2;
    for (size_t i = 0; i < matches.size(); i++) {
        size_t ki = matches[i].i_;
        size_t kj = matches[i].j_;
        points1.push_back(cv::Point2f(features1[ki]->uv_.x(), features1[ki]->uv_.y()));
        points2.push_back(cv::Point2f(features2[kj]->uv_.x(), features2[kj]->uv_.y()));
    }
    std::vector<uchar> mask;
    cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, 1.0, 0.99, mask);
    ReduceVector(matches, mask);
    if (matches.empty()) {
        return false;
    }
    return true;
}

static inline void AbsoluteToRelative(const Eigen::Matrix3d &R0,
                                      const Eigen::Vector3d &t0,
                                      const Eigen::Matrix3d &R1,
                                      const Eigen::Vector3d &t1,
                                      const Eigen::Vector3d &x0,
                                      Eigen::Matrix3d &R,
                                      Eigen::Vector3d &t,
                                      Eigen::Vector3d &Rx0)
{
    R = R1 * R0.transpose();
    t = t1 - R * t0;
    Rx0 = R * x0;
}

static bool TriangulateIDWMidpoint(const Eigen::Matrix3d &R0,
                                   const Eigen::Vector3d &t0,
                                   const Eigen::Vector3d &x0,
                                   const Eigen::Matrix3d &R1,
                                   const Eigen::Vector3d &t1,
                                   const Eigen::Vector3d &x1,
                                   Eigen::Vector3d &x_euclidean)
{
    // absolute to relative
    Eigen::Matrix3d R;
    Eigen::Vector3d t, Rx0;
    AbsoluteToRelative(R0, t0, R1, t1, x0, R, t, Rx0);

    const double p_norm = Rx0.cross(x1).norm();
    const double q_norm = Rx0.cross(t).norm();
    const double r_norm = x1.cross(t).norm();

    // Eq. (10)
    const auto xprime1 = (q_norm / (q_norm + r_norm))
                         * (t + (r_norm / p_norm) * (Rx0 + x1));

    // relative to absolute
    x_euclidean = R1.transpose() * (xprime1 - t1);

    // Eq. (7)
    const Eigen::Vector3d lambda0_Rx0 = (r_norm / p_norm) * Rx0;
    const Eigen::Vector3d lambda1_x1 = (q_norm / p_norm) * x1;

    // Eq. (9) - test adequation
    return (t + lambda0_Rx0 - lambda1_x1).squaredNorm() <
           std::min(std::min((t + lambda0_Rx0 + lambda1_x1).squaredNorm(),
                             (t - lambda0_Rx0 - lambda1_x1).squaredNorm()),
                    (t - lambda0_Rx0 + lambda1_x1).squaredNorm());
}


int main(int argc, char **argv)
{
    cv::setUseOptimized(1);


    string dataPath = "/home/yanglei/Dataset/2011_10_03_drive_0047_sync/2011_10_03/2011_10_03_drive_0047_sync";
    std::string calibPath = "/home/yanglei/Dataset/2011_10_03_drive_0047_sync/2011_10_03_calib";

    kitti::KittiParser parser(calibPath, dataPath, true);
    parser.loadCalibration();
    parser.loadTimestampMaps();
    CameraCalibration cam;
    parser.getCameraCalibration(0, cam);
    cv::Mat kMat;
    cv::eigen2cv(cam.K, kMat);

    Eigen::Matrix4d imu2Cam = parser.T_camN_imu(0).inverse();


    ///
    FeatureTracker featureTracker;
    featureTracker.SetIntrinsicParam(kMat);

    cv::Mat imgPrev;

    std::vector<Feature::Ptr> prevFeatures;
    PairWiseMatches allMatches;
    HashMap<IndexT, std::vector<Feature::Ptr>> allFeatures;
    HashMap<IndexT, Pose3d> allPoses;

    for (size_t i = 600,j = 0; i < parser.timestamps_cam_ns_[0].size(); i++, j++) {
        uint64_t ts;
        cv::Mat img;
        parser.getImageAtEntry(i, 0, ts, img);

        Eigen::Matrix4d pose;
        if (!parser.interpolatePoseAtTimestamp(ts, pose)) {
            continue;
        }

        uint64_t frameId = ts;
        static uint64_t prevFrameId = frameId;
        static uint64_t tsPrev = ts;
        static Eigen::Matrix4d posePrev = pose;
        if (ts != tsPrev) {
            double dt = ((double) ts - (double) tsPrev) * 1e-9;
            double v = (pose.topRightCorner(3, 1) - posePrev.topRightCorner(3, 1)).norm() / dt;
            if (v < 1.0) {
                continue;
            }
        }
        tsPrev = ts;
        posePrev = pose;

        Pose3d pose3 = {frameId, pose};
        allPoses.insert({frameId, pose3});

        std::vector<Feature::Ptr> &&curFeatures = featureTracker.TrackImage(ts, img);
        allFeatures.insert({ts, curFeatures});

        std::vector<IndMatch> &&matches = featureTracker.GetMatches();

        if (!matches.empty()) {
            static cv::Size gmsSize = {img.cols, (int) std::ceil(img.rows * 0.4)};
            if (!RejectByGMS(prevFeatures, gmsSize, curFeatures, gmsSize, matches)) {
                continue;
            }
            if (!RejectByF(prevFeatures, curFeatures, matches)) {
                continue;
            }

            Pair ij = {prevFrameId, frameId};
            allMatches.insert(std::make_pair(ij, matches));
            cout << "--> features / matches: " << curFeatures.size() << " / " << matches.size() << endl;
#if 1
            // show
            cv::Mat imMatch;
            cv::vconcat(imgPrev, img, imMatch);
            cv::cvtColor(imMatch, imMatch, cv::COLOR_GRAY2BGR);
            for (auto &match: matches) {
                size_t i = match.i_;
                size_t j = match.j_;
                auto &pt1 = prevFeatures[i]->uv_;
                auto &pt2 = curFeatures[j]->uv_;
                cv::circle(imMatch, cv::Point2f(pt1.x(), pt1.y()), 2, cv::Scalar(255, 0, 0));
                cv::circle(imMatch, cv::Point2f(pt2.x(), imgPrev.rows + pt2.y()), 2, cv::Scalar(255, 0, 0));
                cv::line(imMatch, cv::Point2f(pt1.x(), pt1.y()), cv::Point2f(pt2.x(), imgPrev.rows + pt2.y()),
                         cv::Scalar(255, 180, 0));
            }
            cv::imshow("match image", imMatch);
            cv::waitKey(10);
#endif
        }

        prevFrameId = frameId;
        prevFeatures.swap(curFeatures);
        imgPrev = img;

        if (0) {
            cv::Mat &&imTrack = featureTracker.getTrackImage();
            cv::imshow("track image", imTrack);
            cv::waitKey(10);
        }
    }

    Tracklets mapTracks;
    TracksBuilder tracksBuilder;
    tracksBuilder.Build(allMatches);
    tracksBuilder.Filter(4);
    tracksBuilder.ExportToSTL(mapTracks);

    ::ofstream ofs("out.txt");
    for (auto it = mapTracks.begin(); it != mapTracks.end(); ++it) {
        IndexT kX3 = it->first;
        bool isOK = false;
        for (auto it1 = it->second.begin(); isOK != true && it1 != it->second.end(); ++it1) {
            IndexT iFrameId = it1->first;
            IndexT iFeat = it1->second;
            auto x1 = allFeatures[iFrameId][iFeat]->xyz_;
            auto pose1 = allPoses[iFrameId].ToPoseMat()*imu2Cam ;

            for (auto it2 = it->second.rbegin(); isOK != true && it2 != it->second.rend(); ++it2) {
                IndexT jFrameId = it2->first;
                IndexT jFeat = it2->second;
                auto x2 = allFeatures[jFrameId][jFeat]->xyz_;
                auto pose2 = allPoses[jFrameId].ToPoseMat()*imu2Cam ;

                if (iFrameId >= jFrameId) {
                    continue;
                }

                Eigen::Matrix3d R1 = pose1.topLeftCorner(3, 3);
                Eigen::Vector3d p1 = pose1.topRightCorner(3, 1);
                Eigen::Matrix3d R2 = pose2.topLeftCorner(3, 3);
                Eigen::Vector3d p2 = pose2.topRightCorner(3, 1);

                Eigen::Vector3d x3w;
                if(TriangulateIDWMidpoint(R1, p1, x1, R2, p2, x2, x3w)){
                    cout<<"id : "<<kX3<<"  , x3 : "<< x3w.transpose()<<endl;
                    ofs<<x3w.transpose()<<endl;
                    isOK = true;
                }
//                cout<<"frame : "<<iFrameId<<" - "<<jFrameId<<endl;
            }
        }
    }

    cout << endl << "--> Tracks: " << mapTracks.size() << endl;

    return 0;
}



