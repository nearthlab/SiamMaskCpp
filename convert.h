#ifndef SIAMMASK_CONVERT_Hh
#define SIAMMASK_CONVERT_Hh

#include "common.h"

inline Tensor toTensor(const Mat& img) {
    Mat imgf;
    img.convertTo(imgf, CV_32FC3);
    return torch::tensor(
           torch::ArrayRef<float>(
                imgf.ptr<float>(),
                imgf.rows * imgf.cols * imgf.channels()
           )
    ).reshape({1, img.rows, img.cols, 3}).permute({0, 3, 1, 2});
}

inline Tensor toTensor(const GpuMat& gpuimg) {
    Mat img;
    gpuimg.download(img);
    return toTensor(img);
}

inline void toMat(const Tensor& tensor, Mat& mat) {
    CV_Assert(tensor.dim() == 2, tensor.scalar_type() == torch::ScalarType::Float);
    Mat(tensor.size(0), tensor.size(1), CV_32FC1, tensor.cpu().data_ptr<float>()).copyTo(mat);
}

inline void toGpuMat(const Tensor& tensor, GpuMat& gmat) {
    Mat mat;
    toMat(tensor, mat);
    gmat.upload(mat);
}

#endif // SIAMMASK_CONVERT_Hh
