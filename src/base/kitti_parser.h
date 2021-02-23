/*
* All Rights Reserved
* Author: yanglei
* Date: 下午8:25
*/
#ifndef VISUAL_MAPPING_KITTI_PARSER_H
#define VISUAL_MAPPING_KITTI_PARSER_H

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <opencv2/core.hpp>
#include <memory>
#include <map>

namespace kitti {

struct CameraCalibration {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Intrinsics.
    Eigen::Vector2d image_size = Eigen::Vector2d::Zero();  // S_xx in calibration.
    Eigen::Matrix3d rect_mat =
            Eigen::Matrix3d::Identity();  // R_rect_xx in calibration.
    Eigen::Matrix<double, 3, 4> projection_mat =
            Eigen::Matrix<double, 3, 4>::Identity();  // P_xx in calibration.

    // Unrectified (raw) intrinsics. Should only be used if rectified set to
    // false.
    Eigen::Matrix3d K =
            Eigen::Matrix3d::Zero();  // Camera intrinsics, K_xx in calibration.
    Eigen::Matrix<double, 1, 5> D =
            Eigen::Matrix<double, 1,
                    5>::Zero();  // Distortion parameters, radtan model.

    // Extrinsics.
    Eigen::Matrix4d T_cam0_cam = Eigen::Matrix4d::Identity();

    bool distorted = false;
};

typedef std::vector<CameraCalibration,
        Eigen::aligned_allocator<CameraCalibration> >
        CameraCalibrationVector;

// Where t = 0 means 100% left transformation,
// and t = 1 means 100% right transformation.
Eigen::Matrix4d interpolateTransformations(const Eigen::Matrix4d &left,
                                           const Eigen::Matrix4d &right,
                                           double t);


class KittiParser {
public:
    // Constants for filenames for calibration files.
    static const std::string kVelToCamCalibrationFilename;
    static const std::string kCamToCamCalibrationFilename;
    static const std::string kImuToVelCalibrationFilename;
    static const std::string kVelodyneFolder;
    static const std::string kCameraFolder;
    static const std::string kPoseFolder;
    static const std::string kTimestampFilename;
    static const std::string kDataFolder;

    KittiParser(const std::string& calibration_path,
                const std::string& dataset_path, bool rectified);

    // MAIN API: all you should need to use!
    // Loading calibration files.
    bool loadCalibration();
    void loadTimestampMaps();

    // Load specific entries (indexed by filename).
    bool getPoseAtEntry(uint64_t entry, uint64_t& timestamp,
                        Eigen::Matrix4d& pose);
    uint64_t getPoseTimestampAtEntry(uint64_t entry);

    bool interpolatePoseAtTimestamp(uint64_t timestamp, Eigen::Matrix4d& pose);

    bool getGpsAtEntry() { /* TODO! */
        return false;
    }
    bool getImuAtEntry() { /* TODO! */
        return false;
    }
    bool getImageAtEntry(uint64_t entry, uint64_t cam_id, uint64_t& timestamp,
                         cv::Mat& image);

    bool getCameraCalibration(uint64_t cam_id, CameraCalibration& cam) const;

    Eigen::Matrix4d T_camN_vel(int cam_number) const;
    Eigen::Matrix4d T_camN_imu(int cam_number) const;

    // Returns the nanosecond timestamp since epoch for a particular entry.
    // Returns -1 if no valid timestamp is found.
    // int64_t getTimestampNsAtEntry(int64_t entry) const;

    // Basic accessors.
    Eigen::Matrix4d T_cam0_vel() const;
    Eigen::Matrix4d T_vel_imu() const;

    size_t getNumCameras() const;


    bool loadCamToCamCalibration();
    bool loadVelToCamCalibration();
    bool loadImuToVelCalibration();

    bool convertGpsToPose(const std::vector<double>& oxts, Eigen::Matrix4d& pose);
    double latToScale(double lat) const;
    void latlonToMercator(double lat, double lon, double scale,
                          Eigen::Vector2d& mercator) const;
    bool loadTimestampsIntoVector(const std::string& filename,
                                  std::vector<uint64_t>& timestamp_vec) const;

    bool parseVectorOfDoubles(const std::string& input,
                              std::vector<double>& output) const;

    std::string getFolderNameForCamera(int cam_number) const;
    std::string getFilenameForEntry(uint64_t entry) const;

    // Base paths.
    std::string calibration_path_;
    std::string dataset_path_;
    // Whether this dataset contains raw or rectified images. This determines
    // which calibration is read.
    bool rectified_;

    // Cached calibration parameters -- std::vector of camera calibrations.
    CameraCalibrationVector camera_calibrations_;

    // Transformation chain (cam-to-cam extrinsics stored above in cam calib
    // struct).
    Eigen::Matrix4d T_cam0_vel_ = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d T_vel_imu_ = Eigen::Matrix4d::Identity();

    // Timestamp map from index to nanoseconds.
    std::vector<uint64_t> timestamps_vel_ns_;
    std::vector<uint64_t> timestamps_pose_ns_;
    // Vector of camera timestamp vectors.
    std::vector<std::vector<uint64_t> > timestamps_cam_ns_;

    // Cached pose information, to correct to odometry frame (instead of absolute
    // world coordinates).
    bool initial_pose_set_;
    Eigen::Matrix4d T_initial_pose_ = Eigen::Matrix4d::Identity();
    double mercator_scale_;
};

}  // namespace kitti


#endif //VISUAL_MAPPING_KITTI_PARSER_H
