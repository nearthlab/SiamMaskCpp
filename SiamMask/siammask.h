#ifndef SIAMMASK_SIAMMASK_Hh
#define SIAMMASK_SIAMMASK_Hh

#include <tuple>
#include <opencv2/cudawarping.hpp>
#include "convert.h"
#include "numpy.h"
#include "state.h"

namespace np = numpy;

class SiamMask {
public:
    typedef std::tuple<
        std::vector<torch::IValue>,
        std::vector<torch::IValue>,
        torch::IValue
    > FeatureType;

    SiamMask(const std::string& model_dir, const torch::Device& device) {

        std::vector<std::string> module_names = {
            "feature_extractor", "feature_downsampler", "rpn_model",
            "mask_conv_kernel", "mask_conv_search", "mask_depthwise_conv",
            "refine_model"
        };

        for(const std::string& module_name : module_names) {
            std::cout << "Loading SiamMask module: " << module_name << " ..."<< std::endl;
            module[module_name] = torch::jit::load(
                model_dir + "/" + module_name + ".pt"
            );
            module[module_name].to(device);
        }

        // this is to warm up GPU operations
        extractTemplateFeature(torch::zeros({1, 3, 32, 32}).to(device));
    }

    void extractTemplateFeature(const torch::IValue& template_image) {
        torch::IValue resnet_outputs = module["feature_extractor"].forward({template_image});
        std::vector<torch::IValue> resnet_features = resnet_outputs.toTuple().get()->elements();
        template_feature = module["feature_downsampler"].forward({resnet_features.back()});
    }

    FeatureType extractSearchFeatures(const torch::IValue& search_image) {
        torch::IValue resnet_outputs = module["feature_extractor"].forward({search_image});
        std::vector<torch::IValue> resnet_features = resnet_outputs.toTuple().get()->elements();
        torch::IValue search_feature = module["feature_downsampler"].forward({resnet_features.back()});
        resnet_features.pop_back();

        torch::IValue rpn_pred = module["rpn_model"].forward({template_feature, search_feature});
        std::vector<torch::IValue> rpn_features = rpn_pred.toTuple().get()->elements();

        torch::IValue kernel = module["mask_conv_kernel"].forward({template_feature});
        torch::IValue search = module["mask_conv_search"].forward({search_feature});
        torch::IValue corr_feature = module["mask_depthwise_conv"].forward({search, kernel});

        return make_tuple(
            resnet_features,
            rpn_features,
            corr_feature
        );
    }

    torch::IValue refineMask(
        const std::vector<torch::IValue>& resnet_features,
        const torch::IValue& corr_feature,
        const torch::IValue& pos
    ) {
        return module["refine_model"].forward({
            resnet_features[0], resnet_features[1],
            resnet_features[2], corr_feature,
            pos
        });
    }

private:
    std::map<std::string, torch::jit::script::Module> module;
    torch::IValue template_feature;
};

inline cv::cuda::GpuMat generateAnchorBase(const State& state) {
    cv::Mat anchors = cv::Mat::zeros(state.anchor_num(), 4, CV_32FC1);

    uint64_t size = state.stride * state.stride;
    uint64_t count = 0;

    cv::Mat anchors_offset = np::arange(state.anchor_density);

    anchors_offset *= state.stride / state.anchor_density;
    anchors_offset -= cv::mean(anchors_offset);

    for(uint8_t x = 0; x < anchors_offset.cols; ++x)
        for(uint8_t y = 0; y < anchors_offset.cols; ++y) {
            const float x_offset = anchors_offset.at<float>(0, x);
            const float y_offset = anchors_offset.at<float>(0, y);

            for(float r : state.ratios) {
                uint64_t ws = sqrt(size / r);
                uint64_t hs = ws * r;

                for(float s : state.scales) {
                    float w = ws * s;
                    float h = hs * s;
                    float a[] = {-w*0.5f+x_offset, -h*0.5f+y_offset, w*0.5f+x_offset, h*0.5f+y_offset};
                    cv::Mat(1, 4, CV_32FC1, a).copyTo(anchors.row(count));
                    count += 1;
                }
            }
        }

    cv::cuda::GpuMat ganchors;
    ganchors.upload(anchors);

    return ganchors;
}

inline cv::cuda::GpuMat generateAnchors(const State& state) {
    cv::cuda::GpuMat anchor = generateAnchorBase(state);

    cv::cuda::GpuMat x1 = anchor.col(0).clone();
    cv::cuda::GpuMat y1 = anchor.col(1).clone();
    cv::cuda::GpuMat x2 = anchor.col(2).clone();
    cv::cuda::GpuMat y2 = anchor.col(3).clone();

    cv::cuda::addWeighted(x1, 0.5, x2, 0.5, 0.0, anchor.col(0));
    cv::cuda::addWeighted(x1, 0.5, x2, 0.5, 0.0, anchor.col(1));
    cv::cuda::subtract(x2, x1, anchor.col(2));
    cv::cuda::subtract(y2, y1, anchor.col(3));

    uint64_t total_stride = state.stride;
    uint64_t anchor_num = anchor.rows;

    const uint64_t score_size = state.score_size();
    anchor = np::tile(anchor, score_size*score_size);
    anchor = anchor.reshape(1, anchor.rows * anchor.cols / 4);

    long ori = -(score_size / 2) * total_stride;

    cv::cuda::GpuMat gridseed;
    gridseed.upload(np::arange(ori, ori + total_stride * score_size, total_stride));
    std::pair<cv::cuda::GpuMat, cv::cuda::GpuMat> grid = np::meshgrid(gridseed, gridseed);
    cv::cuda::GpuMat& xx = grid.first;
    cv::cuda::GpuMat& yy = grid.second;

    xx = np::tile(
        xx.reshape(1, 1),
        anchor_num, false
    );

    yy = np::tile(
        yy.reshape(1, 1),
        anchor_num, false
    );

    xx = xx.reshape(1, xx.rows * xx.cols);
    yy = yy.reshape(1, yy.rows * yy.cols);

    xx.copyTo(anchor.col(0));
    yy.copyTo(anchor.col(1));

    return anchor;
}

inline torch::Tensor getSubwindowTensor(
    const cv::cuda::GpuMat& img,
    const cv::Rect& target,
    uint64_t model_sz,
    uint64_t original_sz,
    cv::Scalar avg_chans
) {
    cv::Rect context = centeredRect(center(target), original_sz, original_sz);

    cv::Rect img_rect = getRect(img);
    cv::Rect padded_rect = img_rect | context;
    cv::Point dp = -padded_rect.tl();

    context = translateRect(context, dp);
    img_rect = translateRect(img_rect, dp);
    padded_rect = translateRect(padded_rect, dp);

    cv::cuda::GpuMat padded_img(padded_rect.size(), CV_8UC3, avg_chans);
    img.copyTo(padded_img(img_rect));

    cv::cuda::GpuMat patch;
    cv::cuda::resize(padded_img(context), patch, cv::Size(model_sz, model_sz));

    return toTensor(patch);
}

inline void siameseInit(
    State& state,
    SiamMask& model,
    const cv::cuda::GpuMat& img,
    const cv::Rect& roi,
    const torch::Device& device
) {
    state.target = roi;
    state.anchors = generateAnchors(state);
    state.avg_chans = cv::cuda::sum(img) / (img.rows * img.cols);

    const unsigned long wc_z = state.target.width + state.context_amount * (state.target.width + state.target.height);
    const unsigned long hc_z = state.target.height + state.context_amount * (state.target.width + state.target.height);
    const unsigned long s_z = round(sqrt(wc_z * hc_z));

    // initialize the exemplar
    torch::IValue z_crop = getSubwindowTensor(
        img, state.target,
        state.exemplar_size, s_z,
        state.avg_chans
    ).to(device);

    model.extractTemplateFeature(z_crop);

    cv::Mat window;
    const uint64_t score_size = state.score_size();
    if(state.windowing == "cosine") {
        cv::Mat hanning = np::hanning(score_size);
        window = np::outer(hanning, hanning);
    }
    else
        window = cv::Mat::ones(score_size, score_size, CV_32FC1);

    state.window.upload(np::tile(window.reshape(1, 1), state.anchor_num()));
}

inline void siameseTrack(
    State& state,
    SiamMask& model,
    const cv::cuda::GpuMat& img,
    const torch::Device& device
) {
    // Prevent segmentation fault from degenerate box
    if(state.target.width == 0) { state.target.width = 1; }
    if(state.target.height == 0) { state.target.height = 1; }

    const unsigned long wc_x = state.target.width + state.context_amount * (state.target.width + state.target.height);
    const unsigned long hc_x = state.target.height + state.context_amount * (state.target.width + state.target.height);
    float s_x = round(sqrt(wc_x * hc_x));

    const float scale_x = state.exemplar_size / s_x;
    const float pad = (state.instance_size - state.exemplar_size) / (2 * scale_x);
    s_x += 2 * pad;
    s_x = round(s_x);

    cv::Rect crop_box =centeredRect(
        center(state.target), s_x, s_x
    );

    torch::IValue x_crop = getSubwindowTensor(
        img, state.target,
        state.instance_size, s_x,
        state.avg_chans
    ).to(device);

    SiamMask::FeatureType features = model.extractSearchFeatures(x_crop);
    const std::vector<torch::IValue>& resnet_features = std::get<0>(features);
    const std::vector<torch::IValue>& rpn_features = std::get<1>(features);
    const torch::IValue& rpn_pred_cls = rpn_features.front();
    const torch::IValue& rpn_pred_loc = rpn_features.back();
    const torch::IValue& corr_feature = std::get<2>(features);

    torch::Tensor pred_loc_tensor = rpn_pred_loc.toTensor().permute({1, 2, 3, 0}).contiguous().view({4, -1});
    torch::Tensor pred_cls_tensor = rpn_pred_cls.toTensor().permute({1, 2, 3, 0}).contiguous().view({2, -1}).permute({1, 0}).softmax(1);

    cv::cuda::GpuMat delta, score;
    toGpuMat(pred_loc_tensor, delta);
    toGpuMat(pred_cls_tensor, score);
    score = score.col(1);
    cv::cuda::transpose(score, score);

    cv::cuda::GpuMat anchor_xy, anchor_wh;
    cv::cuda::transpose(state.anchors.colRange(0, 2), anchor_xy);
    cv::cuda::transpose(state.anchors.colRange(2, 4), anchor_wh);

    cv::cuda::GpuMat delta_xy, delta_wh;
    cv::cuda::multiply(delta.rowRange(0, 2), anchor_wh, delta_xy);
    cv::cuda::add(delta_xy, anchor_xy, delta_xy);
    delta_xy.copyTo(delta.rowRange(0, 2));

    cv::cuda::exp(delta.rowRange(2, 4), delta_wh);
    cv::cuda::multiply(delta_wh, anchor_wh, delta_wh);
    delta_wh.copyTo(delta.rowRange(2, 4));

    static const auto change = [](const cv::cuda::GpuMat& r) -> cv::cuda::GpuMat {
        cv::cuda::GpuMat m, rinv;
        cv::cuda::divide(1, r, rinv);
        cv::cuda::max(r, rinv, m);
        return m;
    };

    static const auto szm = [](const cv::cuda::GpuMat& w, const cv::cuda::GpuMat& h) -> cv::cuda::GpuMat {
        cv::cuda::GpuMat pad;
        cv::cuda::addWeighted(w, 0.5, h, 0.5, 0.0, pad);
        cv::cuda::GpuMat padded_w, padded_h;
        cv::cuda::add(w, pad, padded_w);
        cv::cuda::add(h, pad, padded_h);
        cv::cuda::GpuMat prod;
        cv::cuda::multiply(padded_w, padded_h, prod);
        cv::cuda::GpuMat size;
        cv::cuda::sqrt(prod, size);
        return size;
    };

    static const auto sz = [](const float& w, const float& h) -> float {
        const float pad = (w + h) * 0.5;
        return std::sqrt((w + pad) * (h + pad));
    };

    const float target_w_in_crop = state.target.width * scale_x;
    const float target_h_in_crop = state.target.height * scale_x;

    cv::cuda::GpuMat size = szm(delta.row(2), delta.row(3));
    cv::cuda::GpuMat norm_size;
    cv::cuda::divide(size, sz(target_w_in_crop, target_h_in_crop), norm_size);
    const cv::cuda::GpuMat scale_penalty = change(norm_size);

    cv::cuda::GpuMat delta_ratio;
    cv::cuda::divide(delta.row(2), delta.row(3), delta_ratio);
    cv::cuda::GpuMat norm_ratio;
    cv::cuda::divide((target_w_in_crop / target_h_in_crop), delta_ratio, norm_ratio);
    const cv::cuda::GpuMat ratio_penalty = change(norm_ratio);

    cv::cuda::GpuMat ratio_penalty_1;
    cv::cuda::subtract(ratio_penalty, 1, ratio_penalty_1);
    cv::cuda::GpuMat penalty;
    cv::cuda::multiply(scale_penalty, ratio_penalty_1, penalty, -state.penalty_k);
    cv::cuda::exp(penalty, penalty);

    cv::cuda::GpuMat pscore;
    cv::cuda::multiply(penalty, score, penalty);
    cv::cuda::addWeighted(penalty, 1 - state.window_influence, state.window, state.window_influence, 0.0, pscore);

    double minValue, maxValue;
    cv::Point minLoc, maxLoc;
    cv::cuda::minMaxLoc(pscore, &minValue, &maxValue, &minLoc, &maxLoc);
    int best_pscore_id = maxLoc.x;
    cv::Mat score_mat;
    score.download(score_mat);
    state.score = score_mat.at<float>(maxLoc);

    cv::cuda::GpuMat pred_in_crop = delta.col(best_pscore_id);
    cv::cuda::divide(pred_in_crop, scale_x, pred_in_crop);

    cv::Mat penalty_mat;
    penalty.download(penalty_mat);
    const float lr = penalty_mat.at<float>(maxLoc) * state.score * state.lr;

    cv::Mat pred_in_crop_mat;
    pred_in_crop.download(pred_in_crop_mat);
    cv::Rect target(
        pred_in_crop_mat.at<float>(0, 0) + state.target.x,
        pred_in_crop_mat.at<float>(1, 0) + state.target.y,
        state.target.width * (1 - lr) + pred_in_crop_mat.at<float>(2, 0) * lr,
        state.target.height * (1 - lr) + pred_in_crop_mat.at<float>(3, 0) * lr
    );

    const long score_size = state.score_size();
    std::vector<long> best_pscore_id_mask = np::unravel_index(best_pscore_id, {5L, score_size, score_size});

    const long delta_x = best_pscore_id_mask[2];
    const long delta_y = best_pscore_id_mask[1];
    torch::Tensor mask_tensor = model.refineMask(
        resnet_features, corr_feature,
        torch::tensor({delta_y, delta_x})
    ).toTensor().to(device).sigmoid().squeeze().view({state.out_size, state.out_size});

    const float scale = crop_box.width / (float)state.instance_size;
    cv::Rect mask_pos(
        crop_box.x + (delta_x - state.base_size / 2) * state.total_stride * scale,
        crop_box.y + (delta_y - state.base_size / 2) * state.total_stride * scale,
        scale * state.exemplar_size,
        scale * state.exemplar_size
    );

    const cv::Rect imgrect = getRect(img);

    cv::cuda::GpuMat raw_mask, mask_chip;
    toGpuMat(mask_tensor, raw_mask);
    cv::cuda::resize(raw_mask, mask_chip, mask_pos.size());

    cv::cuda::GpuMat mask_in_img;
    mask_in_img.upload(cv::Mat::zeros(imgrect.size(), mask_chip.type()));
    cv::Rect mask_subpos = mask_pos & imgrect;
    cv::Rect mask_roi = translateRect(mask_subpos, -mask_pos.tl());
    mask_chip(mask_roi).copyTo(mask_in_img(mask_subpos));

    cv::cuda::GpuMat mask;
    cv::cuda::threshold(mask_in_img, mask, state.seg_thr, 255, CV_THRESH_BINARY);

    mask.download(state.mask);
    state.mask.convertTo(state.mask, CV_8UC1);

    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(state.mask, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    std::vector<double> areas;
    for(const auto& contour : contours)
        areas.push_back(cv::contourArea(contour));

    cv::Rect next_target = target;
    if(contours.size() > 0) {
        const auto max_idx = std::distance(areas.begin(), std::max_element(areas.begin(), areas.end()));
        const auto& max_area = areas[max_idx];
        if(max_area > 100) {
            next_target = cv::boundingRect(contours[max_idx]);
            state.rotated_rect = cv::minAreaRect(contours[max_idx]);
        }
    }

    state.target = next_target;
}

#endif // SIAMMASK_SIAMMASK_Hh
