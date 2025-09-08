// UNet NCNN C++ Inference Example
#include <ncnn/net.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

ncnn::Mat preprocess(const cv::Mat& input_image, int input_w, int input_h) {
    cv::Mat image;
    cv::resize(input_image, image, cv::Size(input_w, input_h));
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    image.convertTo(image, CV_32FC3, 1.0 / 255.0);
    ncnn::Mat in(input_w, input_h, 3);
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < input_h; y++) {
            for (int x = 0; x < input_w; x++) {
                in.channel(c).row(y)[x] = image.at<cv::Vec3f>(y, x)[c];
            }
        }
    }
    return in;
}

void postprocess(const ncnn::Mat& out, const cv::Mat& input_image, int input_w, int input_h, float mask_alpha_val,
                 cv::Mat& mask_resized, cv::Mat& overlay_img) {
    cv::Mat mask(input_h, input_w, CV_8UC1);
    for (int y = 0; y < input_h; y++) {
        for (int x = 0; x < input_w; x++) {
            float v = out.channel(0).row(y)[x];
            mask.at<uchar>(y, x) = static_cast<uchar>(v * 255.0f);
        }
    }
    cv::resize(mask, mask_resized, cv::Size(input_image.cols, input_image.rows), 0, 0, cv::INTER_LINEAR);

    cv::Mat mask_inv;
    cv::bitwise_not(mask_resized, mask_inv);

    cv::Mat overlay(input_image.size(), CV_8UC3, cv::Scalar(0, 255, 0));
    cv::Mat mask_alpha;
    mask_inv.convertTo(mask_alpha, CV_32FC1, mask_alpha_val / 255.0);
    cv::Mat input_image_f, overlay_f;
    input_image.convertTo(input_image_f, CV_32FC3);
    overlay.convertTo(overlay_f, CV_32FC3);
    cv::Mat blended = input_image_f.clone();
    for (int y = 0; y < blended.rows; y++) {
        for (int x = 0; x < blended.cols; x++) {
            float alpha = mask_alpha.at<float>(y, x);
            for (int c = 0; c < 3; c++) {
                blended.at<cv::Vec3f>(y, x)[c] =
                    (1.0f - alpha) * input_image_f.at<cv::Vec3f>(y, x)[c] + alpha * overlay_f.at<cv::Vec3f>(y, x)[c];
            }
        }
    }
    blended.convertTo(overlay_img, CV_8UC3);
}

int main(int argc, char** argv) {
    // --- Configurable Parameters (with defaults) ---
    std::string model_param = "../../checkpoints/exp_20250907_172056/ncnn_models/unet_jit.ncnn.param";
    std::string model_bin = "../../checkpoints/exp_20250907_172056/ncnn_models/unet_jit.ncnn.bin";
    std::string input_image_path = "input.jpg";
    std::string output_mask_path = "mask.jpg";
    std::string output_overlay_path = "overlay.jpg";
    int input_w = 256;
    int input_h = 256;
    std::string input_layer = "in0";
    std::string output_layer = "out0";
    float mask_alpha_val = 0.8f;

    // --- Parse Command Line Arguments (key=value format) ---
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") {
            std::cout << "Usage: ./unet_ncnn key=value ..." << std::endl;
            std::cout << "Keys: input_image, model_param, model_bin, output_mask, output_overlay, input_w, input_h, input_layer, output_layer, mask_alpha" << std::endl;
            std::cout << "Defaults:" << std::endl;
            std::cout << "  input_image: " << input_image_path << std::endl;
            std::cout << "  model_param: " << model_param << std::endl;
            std::cout << "  model_bin: " << model_bin << std::endl;
            std::cout << "  output_mask: " << output_mask_path << std::endl;
            std::cout << "  output_overlay: " << output_overlay_path << std::endl;
            std::cout << "  input_w: " << input_w << std::endl;
            std::cout << "  input_h: " << input_h << std::endl;
            std::cout << "  input_layer: " << input_layer << std::endl;
            std::cout << "  output_layer: " << output_layer << std::endl;
            std::cout << "  mask_alpha: " << mask_alpha_val << std::endl;
            return 0;
        }
        size_t eq = arg.find('=');
        if (eq == std::string::npos) continue;
        std::string key = arg.substr(0, eq);
        std::string val = arg.substr(eq + 1);
        if (key == "input_image") input_image_path = val;
        else if (key == "model_param") model_param = val;
        else if (key == "model_bin") model_bin = val;
        else if (key == "output_mask") output_mask_path = val;
        else if (key == "output_overlay") output_overlay_path = val;
        else if (key == "input_w") input_w = std::stoi(val);
        else if (key == "input_h") input_h = std::stoi(val);
        else if (key == "input_layer") input_layer = val;
        else if (key == "output_layer") output_layer = val;
        else if (key == "mask_alpha") mask_alpha_val = std::stof(val);
    }

    // --- Load and Preprocess Image ---
    cv::Mat input_image = cv::imread(input_image_path);
    if (input_image.empty()) {
        std::cerr << "Failed to load image: " << input_image_path << std::endl;
        return -1;
    }
    ncnn::Mat in = preprocess(input_image, input_w, input_h);

    // --- Load Model ---
    ncnn::Net net;
    net.load_param(model_param.c_str());
    net.load_model(model_bin.c_str());

    // --- Run Inference (Timing) ---
    auto time_start = std::chrono::high_resolution_clock::now();
    ncnn::Extractor ex = net.create_extractor();
    ex.input(input_layer.c_str(), in);
    ncnn::Mat out;
    ex.extract(output_layer.c_str(), out);
    auto time_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = time_end - time_start;
    double ms = elapsed.count() * 1000.0;
    double fps = 1000.0 / ms;
    std::cout << "Model inference time: " << ms << " ms" << std::endl;
    std::cout << "FPS: " << fps << std::endl;

    // --- Post-process and Save Mask & Overlay ---
    cv::Mat mask_resized, overlay_img;
    postprocess(out, input_image, input_w, input_h, mask_alpha_val, mask_resized, overlay_img);
    cv::imwrite(output_mask_path, mask_resized);
    cv::imwrite(output_overlay_path, overlay_img);

    // --- Final Output ---
    std::cout << "Inference done, mask saved as " << output_mask_path << ", overlay saved as " << output_overlay_path << "." << std::endl;
    return 0;
}
