#ifndef SIAMMASK_STATE_Hh
#define SIAMMASK_STATE_Hh

#include "common.h"
#include "config_reader.h"

struct State {

    // Default hyper-parameters
    float penalty_k = 0.09;
    float window_influence = 0.39;
    float lr = 0.38;
    float seg_thr = 0.3; // for mask
    string windowing = "cosine"; // to penalize large displacements [cosine/uniform]

    // Params from the network architecture, have to be consistent with the training
    long exemplar_size = 127; // input z size
    long instance_size = 255; // input x size (search region)
    long total_stride = 8;
    long out_size = 63; // for mask
    long base_size = 8;
    float context_amount = 0.5; // context amount for the exemplar

    // Params for anchor generation
    uint64_t stride = 8;
    vector<float> ratios = { 0.33, 0.5, 1, 2, 3 };
    vector<float> scales = { 8 };
    uint64_t image_center = 0;
    uint64_t anchor_density = 1;
    GpuMat anchors;

    // Tracking params
    cv::Scalar avg_chans = cv::Scalar(0, 0, 0);
    Rect target;
    float score = 0.f;
    GpuMat window;
    Mat mask;

    void load_config(const string& config_path) {
        dlib::config_reader cr(config_path);
        cout << "Loading SiamMask config: " << config_path << " ..." << endl;
        penalty_k = dlib::get_option(cr, "hp.penalty_k", penalty_k);
        window_influence = dlib::get_option(cr, "hp.window_influence", window_influence);
        lr = dlib::get_option(cr, "hp.lr", lr);
        seg_thr = dlib::get_option(cr, "hp.seg_thr", seg_thr);
        windowing = dlib::get_option(cr, "hp.windowing", windowing);
        base_size = dlib::get_option(cr, "hp.base_size", base_size);
        instance_size = dlib::get_option(cr, "hp.instance_size", instance_size);
        out_size = dlib::get_option(cr, "hp.out_size", out_size);

        ratios = dlib::get_option(cr, "anchors.ratios", ratios);
        scales = dlib::get_option(cr, "anchors.scales", scales);
        stride = dlib::get_option(cr, "anchors.stride", stride);
        image_center = dlib::get_option(cr, "anchors.image_center", image_center);
        anchor_density = dlib::get_option(cr, "anchors.anchor_density", anchor_density);
    }

    uint64_t anchor_num() const {
        return scales.size() * ratios.size() * anchor_density * anchor_density;
    }

    uint64_t score_size() const {
        return uint64_t((instance_size - exemplar_size) / total_stride) + 1 + base_size;
    }
};

std::ostream& operator << (std::ostream& out, const State& state) {
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
        << "anchors(shape): " << shapeof(state.anchors) << endl
        << "avg_chans: " << state.avg_chans << endl
        << "target: " << state.target << endl
        << "window(shape): " << shapeof(state.window);
    return out;
}

#endif // SIAMMASK_STATE_Hh
