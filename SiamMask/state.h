#ifndef SIAMMASK_STATE_Hh
#define SIAMMASK_STATE_Hh

#include <opencv2/core.hpp>
#include <torch/types.h>
#include <nlohmann/json.hpp>

struct State {

    // Default hyper-parameters
    float penalty_k = 0.09;
    float window_influence = 0.39;
    float lr = 0.38;
    float seg_thr = 0.3; // for mask
    std::string windowing = "cosine"; // to penalize large displacements [cosine/uniform]

    // Params from the network architecture, have to be consistent with the training
    long exemplar_size = 127; // input z size
    long instance_size = 255; // input x size (search region)
    long total_stride = 8;
    long out_size = 63; // for mask
    long base_size = 8;
    float context_amount = 0.5; // context amount for the exemplar

    // Params for anchor generation
    int64_t stride = 8;
    std::vector<float> ratios = { 0.33, 0.5, 1, 2, 3 };
    std::vector<float> scales = { 8 };
    int64_t image_center = 0;
    int64_t anchor_density = 1;
    torch::Tensor anchors;

    // Tracking params
    cv::Scalar avg_chans = cv::Scalar(0, 0, 0);
    cv::Rect target;
    float score = 0.f;
    torch::Tensor window;
    cv::Mat mask;
    cv::RotatedRect rotated_rect;

    void load_config(const std::string& config_path) {
        std::ifstream fin(config_path);
        if (fin.is_open()) {
            nlohmann::json json;
            fin >> json;
            const auto& hp = json.at("hp");

            penalty_k = hp.value("penalty_k", penalty_k);
            window_influence = hp.value("window_influence", window_influence);
            lr = hp.value("lr", lr);
            seg_thr = hp.value("seg_thr", seg_thr);
            windowing = hp.value("windowing", windowing);
            base_size = hp.value("base_size", base_size);
            instance_size = hp.value("instance_size", instance_size);
            out_size = hp.value("out_size", out_size);

            const auto& anchors = json.at("anchors");

            ratios = anchors.value("ratios", ratios);
            scales = anchors.value("scales", scales);
            stride = anchors.value("stride", stride);
            image_center = anchors.value("image_center", image_center);
            anchor_density = anchors.value("anchor_density", anchor_density);
        } else {
            std::ostringstream sout;
            sout << "Failed to open config file " << config_path;
            throw std::runtime_error(sout.str());
        }
    }

    int64_t num_anchors() const {
        return scales.size() * ratios.size() * anchor_density * anchor_density;
    }

    int64_t score_size() const {
        return int64_t((instance_size - exemplar_size) / total_stride) + 1 + base_size;
    }
};

std::ostream& operator << (std::ostream& out, const State& state) {
    using std::endl;
    out << "window_influence: " << state.window_influence << endl
        << "seg_thr: " << state.seg_thr << endl
        << "windowing: " << state.windowing << endl
        << "exemplar_size: " << state.exemplar_size << endl
        << "instance_size: " << state.instance_size << endl
        << "total_stride: " << state.total_stride << endl
        << "out_size: " << state.out_size << endl
        << "base_size: " << state.base_size << endl
        << "context_amount: " << state.context_amount << endl
        << "stride: " << state.stride << endl
        << "ratios: " << state.ratios << endl
        << "scales: " << state.scales << endl
        << "image_center: " << state.image_center << endl
        << "anchor_density: " << state.anchor_density << endl
        << "anchors(rows cols channels): " << state.anchors.sizes() << endl
        << "avg_chans: " << state.avg_chans << endl
        << "target: " << state.target << endl
        << "window(rows cols channels): " << state.window.sizes();
    return out;
}

#endif // SIAMMASK_STATE_Hh
