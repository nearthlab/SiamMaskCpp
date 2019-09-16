#ifndef SIAMMASK_COMMON_Hh
#define SIAMMASK_COMMON_Hh

#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>

using Tensor = torch::Tensor;
using IValue = torch::IValue;
using Device = torch::Device;

using GpuMat = cv::cuda::GpuMat;
using Mat = cv::Mat;
using Point = cv::Point;
using Rect = cv::Rect;
using RotatedRect = cv::RotatedRect;
using Scalar = cv::Scalar;

template<typename T, typename U>
using pair = std::pair<T, U>;
template<typename T>
using vector = std::vector<T>;
template<typename T, typename U>
using map = std::map<T, U>;
using string = std::string;
using std::cout;
using std::endl;

#ifndef shapeof
#define shapeof(m) vector<int>({m.rows, m.cols, m.channels()})
#endif // shapeof

#ifndef print
#define print(x) cout << (#x) << ": " << x << endl
#endif // print

#endif // SIAMMASK_COMMON_Hh
