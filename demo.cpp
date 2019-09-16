#include <argparse/argparse.hpp>
#include <dlib/dir_nav.h>
#include <SiamMask/siammask.h>

void overlayMask(const Mat& src, const Mat& mask, Mat& dst) {
    vector<Mat> chans;
    cv::split(src, chans);
    cv::max(chans[2], mask, chans[2]);
    cv::merge(chans, dst);
}

void drawBox(
    Mat& img, const RotatedRect& box, const Scalar& color,
    int thickness = 1, int lineType = cv::LINE_8, int shift = 0
) {
    cv::Point2f corners[4];
    box.points(corners);
    for(int i = 0; i < 4; ++i) {
        cv::line(img, corners[i], corners[(i + 1) % 4], color, thickness, lineType, shift);
    }
}

int main(int argc, const char* argv[]) try {
    argparse::ArgumentParser parser;
    parser.addArgument("-m", "--modeldir", 1, false);
    parser.addArgument("-c", "--config", 1, false);
    parser.addFinalArgument("target");

    parser.parse(argc, argv);

    Device device(torch::kCUDA);

    SiamMask siammask(parser.retrieve<string>("modeldir"), device);
    State state;
    state.load_config(parser.retrieve<string>("config"));

    dlib::directory target_dir(parser.retrieve<string>("target"));
    vector<dlib::file> image_files = dlib::get_files_in_directory_tree(
        target_dir, dlib::match_endings("jpg png bmp"), 0
    );
    std::sort(image_files.begin(), image_files.end());

    cout << image_files.size() << " images found in " << target_dir << endl;

    vector<Mat> images;
    for(const auto& image_file : image_files) {
        images.push_back(cv::imread(image_file.full_name()));
    }

    cv::namedWindow("SiamMask");
    int64 toc = 0;
    Rect roi = cv::selectROI("SiamMask", images.front(), false);

    if(roi.empty())
        return EXIT_SUCCESS;

    for(unsigned long i = 0; i < images.size(); ++i) {
        int64 tic = cv::getTickCount();

        Mat& src = images[i];
        GpuMat gsrc;
        gsrc.upload(src);

        if (i == 0) {
            cout << "Initializing..." << endl;
            siameseInit(state, siammask, gsrc, roi, device);
            cv::rectangle(src, roi, Scalar(0, 255, 0));
            cv::imshow("SiamMask", src);
        } else {
            siameseTrack(state, siammask, gsrc, device);
            overlayMask(src, state.mask, src);
            drawBox(src, state.rotated_rect, Scalar(0, 255, 0));
            cv::imshow("SiamMask", src);
        }

        toc += cv::getTickCount() - tic;
        cv::waitKey(1);
    }

    double total_time = toc / cv::getTickFrequency();
    double fps = image_files.size() / total_time;
    printf("SiamMask Time: %.1fs Speed: %.1ffps (with visulization!)\n", total_time, fps);

    return EXIT_SUCCESS;
} catch (std::exception& e) {
    cout << "Exception thrown!\n" << e.what() << endl;
    return EXIT_FAILURE;
}

