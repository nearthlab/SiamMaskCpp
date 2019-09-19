#ifndef SIAMMASK_NUMPY_Hh
#define SIAMMASK_NUMPY_Hh

#include <dlib/numeric_constants.h>
#include "geometry.h"

namespace numpy {
    inline cv::Mat tile(const cv::Mat& x, uint64_t reps, bool along_row = true) {
        cv::Mat concat;
        if(along_row)
            cv::hconcat(std::vector<cv::Mat>(reps, x), concat);
        else
            cv::vconcat(std::vector<cv::Mat>(reps, x), concat);
        return concat;
    }

    inline cv::cuda::GpuMat concat(const cv::cuda::GpuMat& x, const cv::cuda::GpuMat& y, bool along_row = true) {
        if(x.empty())
            return y.clone();
        else if (y.empty())
            return x.clone();
        CV_Assert(x.channels() == y.channels(), x.type() == y.type());
        if(along_row) {
            CV_Assert(x.rows == y.rows);
            cv::cuda::GpuMat res = cv::cuda::createContinuous(x.rows, x.cols + y.cols, x.type());
            x.copyTo(res(getRect(x)));
            y.copyTo(res(translateRect(getRect(y), cv::Point(x.cols, 0))));
            return res;
        } else {
            CV_Assert(x.cols == y.cols);
            cv::cuda::GpuMat res = cv::cuda::createContinuous(x.rows + y.rows, x.cols, x.type());
            x.copyTo(res(getRect(x)));
            y.copyTo(res(translateRect(getRect(y), cv::Point(0, x.rows))));
            return res;
        }
    }

    inline cv::cuda::GpuMat tile(const cv::cuda::GpuMat& x, uint64_t reps, bool along_row = true) {
        uint8_t digits = log2(reps) + 1;
        std::vector<cv::cuda::GpuMat> concats;
        concats.push_back(x);

        for(uint8_t digit = 0; digit + 1 < digits; ++digit)
            concats.push_back(concat(concats.back(), concats.back(), along_row));

        cv::cuda::GpuMat tiled;
        for(uint8_t digit = 0; digit < digits; ++digit) {
            if((reps >> digit) & 1)
                tiled = concat(tiled, concats[digit], along_row);
        }

        return tiled;
    }

    inline std::pair<cv::Mat, cv::Mat> meshgrid(const cv::Mat& x, const cv::Mat& y) {
        CV_Assert(x.rows == 1 && y.rows == 1);
        return std::make_pair(
            tile(x, y.cols, false),
            tile(y.t(), x.cols)
        );
    }

    inline std::pair<cv::cuda::GpuMat, cv::cuda::GpuMat> meshgrid(const cv::cuda::GpuMat& x, const cv::cuda::GpuMat& y) {
        CV_Assert(x.rows == 1 && y.rows == 1);
        cv::cuda::GpuMat yt;
        cv::cuda::transpose(y, yt);
        return std::make_pair(
            tile(x, y.cols, false),
            tile(yt, x.cols)
        );
    }

    inline cv::Mat hanning(uint64_t M) {
        cv::Mat w(1, M, CV_32FC1);
        for(uint64_t n = 0; n < M; ++n)
            w.at<float>(0, n) = 0.5 - 0.5*cos(2*dlib::pi*n / (M - 1));
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
