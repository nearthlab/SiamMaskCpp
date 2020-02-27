#ifndef SIAMMASK_CONVERT_Hh
#define SIAMMASK_CONVERT_Hh

#include <sstream>
#include <opencv2/core.hpp>
#include <torch/torch.h>

inline torch::Tensor toTensor(const cv::Mat& arr) {
    switch (arr.type()) {
        case CV_8UC3: {
            cv::Mat floatArr;
            arr.convertTo(floatArr, CV_32FC3);
            return torch::tensor(
                torch::ArrayRef<float>(
                    floatArr.ptr<float>(),
                    floatArr.rows * floatArr.cols * floatArr.channels()
                )
            ).reshape({1, arr.rows, arr.cols, 3}).permute({0, 3, 1, 2});
        }
        case CV_32FC1: {
            return torch::tensor(
                torch::ArrayRef<float>(
                    arr.ptr<float>(),
                    arr.rows * arr.cols * arr.channels()
                )
            ).reshape({arr.rows, arr.cols});
        }
        case CV_32FC3: {
            return torch::tensor(
                torch::ArrayRef<float>(
                    arr.ptr<float>(),
                    arr.rows * arr.cols * arr.channels()
                )
            ).reshape({1, arr.rows, arr.cols, 3}).permute({0, 3, 1, 2});
        }
        default: {
            std::ostringstream sout;
            sout << "toTensor(): unrecognized cv::Mat::type() " << arr.type();
            throw std::runtime_error(sout.str());
        }
    }
}

inline void toMat(const torch::Tensor& tensor, cv::Mat& mat) {
    TORCH_CHECK(tensor.dim() == 2, "toMat(): expected tensor.dim() == 2 but got", tensor.dim());
    TORCH_CHECK(tensor.scalar_type() == torch::ScalarType::Float, "toMat(): expected tensor.scalar_type() == torch::ScalarType::Float but got ", tensor.scalar_type());
    cv::Mat(tensor.size(0), tensor.size(1), CV_32FC1, tensor.cpu().data_ptr<float>()).copyTo(mat);
}

#endif // SIAMMASK_CONVERT_Hh
