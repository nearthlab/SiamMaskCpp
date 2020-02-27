#ifndef SIAMMASK_GEOMETRY_Hh
#define SIAMMASK_GEOMETRY_Hh

#include <opencv2/core.hpp>

inline cv::Point center(const cv::Rect& rect) {
    return (rect.tl() + rect.br()) / 2;
}

inline cv::Rect centeredRect(const cv::Point& center, int width, int height) {
    return cv::Rect(center.x - width / 2, center.y - height / 2, width, height);
}

cv::Rect getRect(const cv::Mat& img) {
    return cv::Rect(0, 0, img.cols, img.rows);
}

inline cv::Rect translateRect(const cv::Rect& rect, const cv::Point& p) {
    return cv::Rect(rect.x + p.x, rect.y + p.y, rect.width, rect.height);
}

#endif // SIAMMASK_GEOMETRY_Hh
