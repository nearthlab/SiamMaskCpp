#ifndef SIAMMASK_SIAMMASK_Hh
#define SIAMMASK_SIAMMASK_Hh

#include <tuple>
#include <opencv2/imgproc.hpp>
#include <torch/script.h>

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

inline cv::Mat generateAnchorBase(const State& state) {
    cv::Mat anchors = cv::Mat::zeros(state.num_anchors(), 4, CV_32FC1);

    int64_t size = state.stride * state.stride;
    int64_t count = 0;

    cv::Mat anchors_offset = np::arange(state.anchor_density);

    anchors_offset *= state.stride / state.anchor_density;
    anchors_offset -= cv::mean(anchors_offset);

    for(uint8_t x = 0; x < anchors_offset.cols; ++x)
        for(uint8_t y = 0; y < anchors_offset.cols; ++y) {
            const float x_offset = anchors_offset.at<float>(0, x);
            const float y_offset = anchors_offset.at<float>(0, y);

            for(float r : state.ratios) {
                int64_t ws = sqrt(size / r);
                int64_t hs = ws * r;

                for(float s : state.scales) {
                    float w = ws * s;
                    float h = hs * s;
                    float a[] = {-w*0.5f+x_offset, -h*0.5f+y_offset, w*0.5f+x_offset, h*0.5f+y_offset};
                    cv::Mat(1, 4, CV_32FC1, a).copyTo(anchors.row(count));
                    count += 1;
                }
            }
        }

    return anchors;
}

inline cv::Mat generateAnchors(const State& state) {
    cv::Mat anchor = generateAnchorBase(state);

    cv::Mat x1 = anchor.col(0).clone();
    cv::Mat y1 = anchor.col(1).clone();
    cv::Mat x2 = anchor.col(2).clone();
    cv::Mat y2 = anchor.col(3).clone();

    cv::addWeighted(x1, 0.5, x2, 0.5, 0.0, anchor.col(0));
    cv::addWeighted(x1, 0.5, x2, 0.5, 0.0, anchor.col(1));
    cv::subtract(x2, x1, anchor.col(2));
    cv::subtract(y2, y1, anchor.col(3));

    uint64_t total_stride = state.stride;
    uint64_t anchor_num = anchor.rows;

    uint64_t score_size = state.score_size();
    anchor = np::tile(anchor, score_size * score_size);
    anchor = anchor.reshape(1, anchor.rows * anchor.cols / 4);

    long ori = -(score_size / 2) * total_stride;

    cv::Mat gridseed = np::arange(ori, ori + total_stride * score_size, total_stride);
    std::pair<cv::Mat, cv::Mat> grid = np::meshgrid(gridseed, gridseed);
    cv::Mat& xx = grid.first;
    cv::Mat& yy = grid.second;

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
    const cv::Mat& img,
    const cv::Rect& target,
    int64_t model_sz,
    int64_t original_sz,
    cv::Scalar avg_chans
) {
    cv::Rect context = centeredRect(center(target), original_sz, original_sz);

    cv::Rect img_rect = getRect(img);
    cv::Rect padded_rect = img_rect | context;
    cv::Point dp = -padded_rect.tl();

    context = translateRect(context, dp);
    img_rect = translateRect(img_rect, dp);
    padded_rect = translateRect(padded_rect, dp);

    cv::Mat padded_img(padded_rect.size(), CV_8UC3, avg_chans);
    img.copyTo(padded_img(img_rect));

    cv::Mat patch;
    cv::resize(padded_img(context), patch, cv::Size(model_sz, model_sz));

    return toTensor(patch);
}

inline void siameseInit(
    State& state,
    SiamMask& model,
    const cv::Mat& img,
    const cv::Rect& roi,
    const torch::Device& device
) {
    state.target = roi;
    state.anchors = toTensor(generateAnchors(state)).to(device);
    state.avg_chans = cv::sum(img) / (img.rows * img.cols);

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
    int64_t score_size = state.score_size();
    if(state.windowing == "cosine") {
        cv::Mat hanning = np::hanning(score_size);
        window = np::outer(hanning, hanning);
    }
    else
        window = cv::Mat::ones(score_size, score_size, CV_32FC1);

    state.window = toTensor(np::tile(window.reshape(1, 1), state.num_anchors())).to(device);
}

namespace util {
    torch::Tensor reciprocalMax (const torch::Tensor& r) {
        return torch::max(r, torch::reciprocal(r));
    };

    torch::Tensor getSize(const torch::Tensor& w, const torch::Tensor& h) {
        torch::Tensor pad = (w + h) * 0.5;
        return torch::sqrt((w + pad) * (h + pad));
    };

    float getSize(const float& w, const float& h) {
        float pad = (w + h) * 0.5;
        return std::sqrt((w + pad) * (h + pad));
    };
}

inline void siameseTrack(
    State& state,
    SiamMask& model,
    const cv::Mat& img,
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

    std::vector<torch::IValue> resnet_features, rpn_features;
    torch::IValue corr_feature;
    std::tie(
        resnet_features,
        rpn_features,
        corr_feature
    ) = model.extractSearchFeatures(x_crop);

    const torch::IValue& rpn_pred_cls = rpn_features.front();
    const torch::IValue& rpn_pred_loc = rpn_features.back();

    torch::Tensor delta = rpn_pred_loc.toTensor().permute({1, 2, 3, 0}).contiguous().view({4, -1}).transpose(0, 1);
    torch::Tensor score = rpn_pred_cls.toTensor().permute({1, 2, 3, 0}).contiguous().view({2, -1}).softmax(1).slice(0, 1, 2).flatten();

    torch::Tensor anchor_wh = state.anchors.slice(1, 2, 4);

    torch::Tensor delta_xy = delta.slice(1, 0, 2) * anchor_wh;
    torch::Tensor delta_wh = torch::exp(delta.slice(1, 2, 4));
    delta = torch::cat({delta_xy, delta_wh}, 1);

    float target_crop_w = state.target.width * scale_x;
    float target_crop_h = state.target.height * scale_x;

    torch::Tensor delta_w = delta.slice(1, 2, 3);
    torch::Tensor delta_h = delta.slice(1, 3, 4);
    torch::Tensor s_c = util::reciprocalMax(util::getSize(delta_w, delta_h) / util::getSize(target_crop_w, target_crop_h));
    torch::Tensor r_c = util::reciprocalMax((target_crop_w / target_crop_h) / (delta_w / delta_h));

    torch::Tensor penalty = torch::exp((1 - r_c * s_c) * state.penalty_k).squeeze();
    torch::Tensor pscore = (penalty * score) * (1 - state.window_influence) + state.window * state.window_influence;

    int64_t best_pscore_id = pscore.argmax().cpu().item<int64_t>();
    state.score = score.cpu()[best_pscore_id].item<float>();

    torch::Tensor target_delta = (delta.slice(0, best_pscore_id, best_pscore_id + 1).squeeze() / scale_x).cpu();
    float lr = penalty.cpu()[best_pscore_id].item<float>() * state.score * state.lr;
    cv::Rect target(
        target_delta[0].item<float>() + state.target.x,
        target_delta[1].item<float>() + state.target.y,
        state.target.width * (1 - lr) + target_delta[2].item<float>() * lr,
        state.target.height * (1 - lr) + target_delta[3].item<float>() * lr
    );

    const int64_t score_size = state.score_size();
    std::vector<long> best_pscore_id_mask = np::unravel_index(best_pscore_id, {5L, score_size, score_size});

    const long pos_x = best_pscore_id_mask[2];
    const long pos_y = best_pscore_id_mask[1];
    torch::Tensor mask_tensor = model.refineMask(
        resnet_features, corr_feature,
        torch::tensor({pos_y, pos_x})
    ).toTensor().to(device).sigmoid().squeeze().view({state.out_size, state.out_size});

    const float scale = crop_box.width / (float)state.instance_size;
    cv::Rect mask_pos(
        crop_box.x + (pos_x - state.base_size / 2) * state.total_stride * scale,
        crop_box.y + (pos_y - state.base_size / 2) * state.total_stride * scale,
        scale * state.exemplar_size,
        scale * state.exemplar_size
    );

    const cv::Rect imgrect = getRect(img);

    cv::Mat raw_mask, mask_chip;
    toMat(mask_tensor, raw_mask);
    cv::resize(raw_mask, mask_chip, mask_pos.size());

    cv::Mat mask_in_img = cv::Mat::zeros(imgrect.size(), mask_chip.type());
    cv::Rect mask_subpos = mask_pos & imgrect;
    cv::Rect mask_roi = translateRect(mask_subpos, -mask_pos.tl());
    mask_chip(mask_roi).copyTo(mask_in_img(mask_subpos));

    cv::threshold(mask_in_img, state.mask, state.seg_thr, 255, cv::THRESH_BINARY);
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
