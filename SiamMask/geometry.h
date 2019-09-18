#ifndef SIAMMASK_GEOMETRY_Hh
#define SIAMMASK_GEOMETRY_Hh

#include <opencv2/core.hpp>

inline cv::Point center(const cv::Rect& rect) {
    return (rect.tl() + rect.br()) / 2;
}

inline cv::Rect centeredRect(const cv::Point& center, int width, int height) {
    return cv::Rect(center.x - width / 2, center.y - height / 2, width, height);
}

template<typename MatType>
cv::Rect getRect(const MatType& img) {
    return cv::Rect(0, 0, img.cols, img.rows);
}

inline cv::Rect uniteRects(const cv::Rect& r1, const cv::Rect& r2) {
    const int left = std::min(r1.x, r2.x);
    const int top = std::min(r1.y, r2.y);
    const int right = std::max(r1.x + r1.width, r2.x + r2.width);
    const int bottom = std::max(r1.y + r1.height, r2.y + r2.height);
    return cv::Rect(left, top, right - left, bottom - top);
}

inline cv::Rect intersectRects(const cv::Rect& r1, const cv::Rect& r2) {
    const int left = std::max(r1.x, r2.x);
    const int top = std::max(r1.y, r2.y);
    const int right = std::min(r1.x + r1.width, r2.x + r2.width);
    const int bottom = std::min(r1.y + r1.height, r2.y + r2.height);
    return cv::Rect(left, top, right - left, bottom - top);
}

inline cv::Rect translateRect(const cv::Rect& rect, const cv::Point& p) {
    return cv::Rect(rect.x + p.x, rect.y + p.y, rect.width, rect.height);
}

#endif // SIAMMASK_GEOMETRY_Hh
