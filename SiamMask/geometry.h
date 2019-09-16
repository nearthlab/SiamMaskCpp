#ifndef SIAMMASK_GEOMETRY_Hh
#define SIAMMASK_GEOMETRY_Hh

#include "common.h"

inline Point center(const Rect& rect) {
    return (rect.tl() + rect.br()) / 2;
}

inline Rect centeredRect(const Point& center, int width, int height) {
    return Rect(center.x - width / 2, center.y - height / 2, width, height);
}

template<typename MatType>
Rect getRect(const MatType& img) {
    return Rect(0, 0, img.cols, img.rows);
}

inline Rect uniteRects(const Rect& r1, const Rect& r2) {
    const int left = std::min(r1.x, r2.x);
    const int top = std::min(r1.y, r2.y);
    const int right = std::max(r1.x + r1.width, r2.x + r2.width);
    const int bottom = std::max(r1.y + r1.height, r2.y + r2.height);
    return Rect(left, top, right - left, bottom - top);
}

inline Rect intersectRects(const Rect& r1, const Rect& r2) {
    const int left = std::max(r1.x, r2.x);
    const int top = std::max(r1.y, r2.y);
    const int right = std::min(r1.x + r1.width, r2.x + r2.width);
    const int bottom = std::min(r1.y + r1.height, r2.y + r2.height);
    return Rect(left, top, right - left, bottom - top);
}

inline Rect translateRect(const Rect& rect, const Point& p) {
    return Rect(rect.x + p.x, rect.y + p.y, rect.width, rect.height);
}

#endif // SIAMMASK_GEOMETRY_Hh
