/*
* All Rights Reserved
* Author: yanglei
* Date: 下午7:51
*/
#ifndef VISUAL_MAPPING_REPROJECTIVE_FACTOR_H
#define VISUAL_MAPPING_REPROJECTIVE_FACTOR_H

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace hdmap {
namespace ddi {

struct ReprojectiveFactor {
    ReprojectiveFactor(double u, double v, double fx, double fy, double ppx, double ppy)
            : u_(u), v_(v), fx_(fx), fy_(fy), ppx_(ppx), ppy_(ppy)
    {}

    template<typename T>
    bool operator()(const T *const ip_wb,
                    const T *const iq_wb,
                    const T *const ip_bc,
                    const T *const iq_bc,
                    const T *ix3w,
                    T *iresiduals) const
    {
        Eigen::Matrix<T, 3, 1> p_wb;
        p_wb << ip_wb[0], ip_wb[1], ip_wb[2];
        Eigen::Quaternion<T> q_wb;
        q_wb.coeffs() << iq_wb[0], iq_wb[1], iq_wb[2], iq_wb[3];
        Eigen::Matrix<T, 3, 1> p_bc;
        p_bc << ip_bc[0], ip_bc[1], ip_bc[2];
        Eigen::Quaternion<T> q_bc;
        q_bc.coeffs() << iq_bc[0], iq_bc[1], iq_bc[2], iq_bc[3];
        Eigen::Matrix<T, 3, 1> x3w;
        x3w << ix3w[0], ix3w[1], ix3w[2];

        Eigen::Matrix<T, 3, 1> x3b = q_wb.inverse() * (x3w - p_wb);
        Eigen::Matrix<T, 3, 1> x3c = q_bc.inverse() * (x3b - p_bc);

        T hx = x3c(0) / x3c(2);
        T hy = x3c(1) / x3c(2);

        T uhat = T(fx_) * hx + T(ppx_);
        T vhat = T(fy_) * hy + T(ppy_);

        iresiduals[0] = uhat - T(u_);
        iresiduals[1] = vhat - T(v_);

        return true;
    }

    static ceres::CostFunction *Create(double u, double v, double fx, double fy, double ppx, double ppy)
    {
        return (new ceres::AutoDiffCostFunction<ReprojectiveFactor, 2, 4, 3, 4, 3, 3>(
                new ReprojectiveFactor(u, v, fx, fy, ppx, ppy)));
    }

    double u_;
    double v_;
    double fx_;
    double fy_;
    double ppx_;
    double ppy_;
};


}
}
#endif //VISUAL_MAPPING_REPROJECTIVE_FACTOR_H
