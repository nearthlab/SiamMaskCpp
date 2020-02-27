#ifndef SIAMMASK_NUMPY_Hh
#define SIAMMASK_NUMPY_Hh

#include "geometry.h"

namespace numpy {
    const double pi = 3.1415926535897932385;

    inline cv::Mat tile(const cv::Mat& x, int64_t reps, bool along_row = true) {
        cv::Mat concat;
        if(along_row)
            cv::hconcat(std::vector<cv::Mat>(reps, x), concat);
        else
            cv::vconcat(std::vector<cv::Mat>(reps, x), concat);
        return concat;
    }

    inline std::pair<cv::Mat, cv::Mat> meshgrid(const cv::Mat& x, const cv::Mat& y) {
        CV_Assert(x.rows == 1 && y.rows == 1);
        return std::make_pair(
            tile(x, y.cols, false),
            tile(y.t(), x.cols)
        );
    }

    inline cv::Mat hanning(int64_t M) {
        cv::Mat w(1, M, CV_32FC1);
        for(int64_t n = 0; n < M; ++n)
            w.at<float>(0, n) = 0.5 - 0.5*cos(2*pi*n / (M - 1));
        return w;
    }

    inline cv::Mat outer(const cv::Mat& x, const cv::Mat& y) {
        CV_Assert(x.rows == 1 && y.rows == 1);
        return x.t() * y;
    }

    inline std::vector<long> unravel_index(long index, const std::vector<long>& dims) {
        std::vector<long> indices;
        for(unsigned long i = dims.size() - 1; i >= 1; --i) {
            long ax_idx = index % dims[i];
            indices.push_back(ax_idx);
            index = (index - ax_idx) / dims[i];
        }
        indices.push_back(index);
        std::reverse(indices.begin(), indices.end());
        return indices;
    }

    inline cv::Mat arange(long begins, long end, long step = 1) {
        cv::Mat range(1, int((end - begins) / step), CV_32FC1);
        for(int r = 0; r < range.cols; ++r)
            range.at<float>(0, r) = begins + r * step;
        return range;
    }

    inline cv::Mat arange(unsigned long n) {
        return arange(0, n);
    }
}

#endif // SIAMMASK_NUMPY_Hh
