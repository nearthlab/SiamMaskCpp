#ifndef SIAMMASK_NUMPY_Hh
#define SIAMMASK_NUMPY_Hh

#include "geometry.h"

namespace numpy {
    inline Mat tile(const Mat& x, uint64_t reps, bool along_row = true) {
        Mat concat;
        if(along_row)
            cv::hconcat(vector<Mat>(reps, x), concat);
        else
            cv::vconcat(vector<Mat>(reps, x), concat);
        return concat;
    }

    inline GpuMat concat(const GpuMat& x, const GpuMat& y, bool along_row = true) {
        if(x.empty())
            return y.clone();
        else if (y.empty())
            return x.clone();
        CV_Assert(x.channels() == y.channels(), x.type() == y.type());
        if(along_row) {
            CV_Assert(x.rows == y.rows);
            GpuMat res = cv::cuda::createContinuous(x.rows, x.cols + y.cols, x.type());
            x.copyTo(res(getRect(x)));
            y.copyTo(res(translateRect(getRect(y), Point(x.cols, 0))));
            return res;
        } else {
            CV_Assert(x.cols == y.cols);
            GpuMat res = cv::cuda::createContinuous(x.rows + y.rows, x.cols, x.type());
            x.copyTo(res(getRect(x)));
            y.copyTo(res(translateRect(getRect(y), Point(0, x.rows))));
            return res;
        }
    }

    inline GpuMat tile(const GpuMat& x, uint64_t reps, bool along_row = true) {
        uint8_t digits = log2(reps) + 1;
        vector<GpuMat> concats;
        concats.push_back(x);

        for(uint8_t digit = 0; digit + 1 < digits; ++digit)
            concats.push_back(concat(concats.back(), concats.back(), along_row));

        GpuMat tiled;
        for(uint8_t digit = 0; digit < digits; ++digit) {
            if((reps >> digit) & 1)
                tiled = concat(tiled, concats[digit], along_row);
        }

        return tiled;
    }

    inline pair<Mat, Mat> meshgrid(const Mat& x, const Mat& y) {
        CV_Assert(x.rows == 1 && y.rows == 1);
        return std::make_pair(
            tile(x, y.cols, false),
            tile(y.t(), x.cols)
        );
    }

    inline pair<GpuMat, GpuMat> meshgrid(const GpuMat& x, const GpuMat& y) {
        CV_Assert(x.rows == 1 && y.rows == 1);
        GpuMat yt;
        cv::cuda::transpose(y, yt);
        return std::make_pair(
            tile(x, y.cols, false),
            tile(yt, x.cols)
        );
    }

    inline Mat hanning(uint64_t M) {
        Mat w(1, M, CV_32FC1);
        for(uint64_t n = 0; n < M; ++n)
            w.at<float>(0, n) = 0.5 - 0.5*cos(2*dlib::pi*n / (M - 1));
        return w;
    }

    inline Mat outer(const Mat& x, const Mat& y) {
        CV_Assert(x.rows == 1 && y.rows == 1);
        return x.t() * y;
    }

    inline vector<long> unravel_index(long index, const vector<long>& dims) {
        vector<long> indices;
        for(unsigned long i = dims.size() - 1; i >= 1; --i) {
            long ax_idx = index % dims[i];
            indices.push_back(ax_idx);
            index = (index - ax_idx) / dims[i];
        }
        indices.push_back(index);
        std::reverse(indices.begin(), indices.end());
        return indices;
    }

    inline Mat arange(long begins, long end, long step = 1) {
        Mat range(1, int((end - begins) / step), CV_32FC1);
        for(int r = 0; r < range.cols; ++r)
            range.at<float>(0, r) = begins + r * step;
        return range;
    }

    inline Mat arange(unsigned long n) {
        return arange(0, n);
    }
}

#endif // SIAMMASK_NUMPY_Hh
