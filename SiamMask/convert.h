#ifndef SIAMMASK_CONVERT_Hh
#define SIAMMASK_CONVERT_Hh

#include <torch/script.h>
#include <opencv2/cudaarithm.hpp>

inline torch::Tensor toTensor(const cv::Mat& img) {
    cv::Mat imgf;
    img.convertTo(imgf, CV_32FC3);
    return torch::tensor(
           torch::ArrayRef<float>(
                imgf.ptr<float>(),
                imgf.rows * imgf.cols * imgf.channels()
           )
    ).reshape({1, img.rows, img.cols, 3}).permute({0, 3, 1, 2});
}

inline torch::Tensor toTensor(const cv::cuda::GpuMat& gpuimg) {
    cv::Mat img;
    gpuimg.download(img);
    return toTensor(img);
}

inline void toMat(const torch::Tensor& tensor, cv::Mat& mat) {
    CV_Assert(tensor.dim() == 2, tensor.scalar_type() == torch::ScalarType::Float);
    cv::Mat(tensor.size(0), tensor.size(1), CV_32FC1, tensor.cpu().data_ptr<float>()).copyTo(mat);
}

inline void toGpuMat(const torch::Tensor& tensor, cv::cuda::GpuMat& gmat) {
    cv::Mat mat;
    toMat(tensor, mat);
    gmat.upload(mat);
}

#endif // SIAMMASK_CONVERT_Hh
