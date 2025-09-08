// UNet NCNN C++ Inference Example
#include <ncnn/net.h>
#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: ./unet_ncnn <input_image>" << std::endl;
        return -1;
    }

    // Load image
    cv::Mat input_image = cv::imread(argv[1]);
    if (input_image.empty()) {
        std::cerr << "Failed to load image: " << argv[1] << std::endl;
        return -1;
    }
    cv::Mat image;
    cv::resize(input_image, image, cv::Size(256, 256)); // match training size

    // Convert BGR to RGB
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    // Convert to float and scale to [0, 1]
    image.convertTo(image, CV_32FC3, 1.0 / 255.0);

    // Convert to NCNN format (CHW)
    ncnn::Mat in(256, 256, 3);
    // HWC to CHW
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < 256; y++) {
            for (int x = 0; x < 256; x++) {
                in.channel(c).row(y)[x] = image.at<cv::Vec3f>(y, x)[c];
            }
        }
    }

    // Load model
    ncnn::Net net;
    net.load_param("../../checkpoints/exp_20250907_172056/ncnn_models/unet_jit.ncnn.param");
    net.load_model("../../checkpoints/exp_20250907_172056/ncnn_models/unet_jit.ncnn.bin");

    // Run inference
    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", in); // Use actual input layer name from pnnx output
    ncnn::Mat out;
    ex.extract("out0", out); // Use actual output layer name from pnnx output

    // Post-process and save mask
    cv::Mat mask(256, 256, CV_8UC1);
    for (int y = 0; y < 256; y++) {
        for (int x = 0; x < 256; x++) {
            float v = out.row(y)[x];
            mask.at<uchar>(y, x) = static_cast<uchar>(v * 255.0f);
        }
    }
    // Resize mask back to original image size (bilinear interpolation)
    cv::resize(mask, mask, cv::Size(input_image.cols, input_image.rows), 0, 0, cv::INTER_LINEAR);
    cv::imwrite("output_mask.jpg", mask);

    // Invert mask before overlay (white lane, black background)
    cv::Mat mask_inv;
    cv::bitwise_not(mask, mask_inv);

    // --- Overlay output ---
    // Create green overlay with alpha from inverted mask
    cv::Mat overlay(input_image.size(), CV_8UC3, cv::Scalar(0, 255, 0));
    cv::Mat mask_alpha;
    mask_inv.convertTo(mask_alpha, CV_32FC1, 0.8 / 255.0); // scale alpha to 0.8
    std::vector<cv::Mat> overlay_channels;
    cv::split(overlay, overlay_channels);
    cv::Mat blended;
    input_image.convertTo(input_image, CV_32FC3);
    overlay.convertTo(overlay, CV_32FC3);
    blended = input_image.clone();
    for (int y = 0; y < blended.rows; y++) {
        for (int x = 0; x < blended.cols; x++) {
            float alpha = mask_alpha.at<float>(y, x);
            for (int c = 0; c < 3; c++) {
                blended.at<cv::Vec3f>(y, x)[c] =
                    (1.0f - alpha) * input_image.at<cv::Vec3f>(y, x)[c] + alpha * overlay.at<cv::Vec3f>(y, x)[c];
            }
        }
    }
    blended.convertTo(blended, CV_8UC3);
    cv::imwrite("output_overlay.jpg", blended);
    std::cout << "Inference done, mask saved as output_mask.jpg, overlay saved as output_overlay.jpg." << std::endl;
    return 0;
}
