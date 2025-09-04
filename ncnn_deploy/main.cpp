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

    // Convert to NCNN format
    ncnn::Mat in = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR, 256, 256);

    // Load model
    ncnn::Net net;
    net.load_param("../ncnn_models/unet_jit.ncnn.param");
    net.load_model("../ncnn_models/unet_jit.ncnn.bin");

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
            mask.at<uchar>(y, x) = v > 0.5f ? 255 : 0;
        }
    }
    // resize to original resolution
    cv::resize(mask, mask, cv::Size(input_image.cols, input_image.rows));
    cv::imwrite("output_mask.jpg", mask);
    std::cout << "Inference done, mask saved as output_mask.jpg." << std::endl;
    return 0;
}
