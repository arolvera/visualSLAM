#include "VO.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>

VisualOdometry::VisualOdometry() {
    this->pl_ = this->pl_ = cv::Mat::eye(4, 4, CV_64F);
    this->kl_ = this->kr_ = cv::Mat::eye(3, 3, CV_64F);
    this->detector_ = cv::ORB::create();
    this->descriptor_ = cv::ORB::create();
    this->matcher_ = std::make_unique<cv::FlannBasedMatcher>();
}

VisualOdometry::VisualOdometry(cv::Mat pl, cv::Mat pr) : pl_(pl),
                                                         pr_(pr),
                                                         kl_(pl_(cv::Range(0, 3), cv::Range(0, 3))),
                                                         kr_(pr_(cv::Range(0, 3), cv::Range(0, 3))),
                                                         detector_(cv::ORB::create(3000)),
                                                         descriptor_(cv::ORB::create()),
                                                         matcher_(std::make_unique<cv::FlannBasedMatcher>(new cv::flann::LshIndexParams(6,12,1), new cv::flann::SearchParams(50)))
                                                         {}

/// @brief  Detect keypoints given input image
std::vector<cv::KeyPoint> VisualOdometry::detectKeypoints(const cv::Mat image) {
    std::vector<cv::KeyPoint> keypoints = {};
    this->detector_->detect(image, keypoints);
    return keypoints;
}

/// @brief  Given detected keypoints compute descriptors 
cv::Mat VisualOdometry::computeDescriptors(const cv::Mat image, std::vector<cv::KeyPoint> keypoints) {
    cv::Mat descriptors;
    this->descriptor_->compute(image, keypoints, descriptors);
    return descriptors;
}

/// @brief Detect and match descriptors given a set of images
std::vector<cv::DMatch> VisualOdometry::matchDescriptors(const cv::Mat image_first, const cv::Mat image_second) {  

    std::vector<cv::DMatch> matches = {};
    std::vector<cv::KeyPoint> keypoints_first, keypoints_second = {};
    cv::Mat descriptors_first, descriptors_second;

    this->detector_->detect(image_first, keypoints_first);
    this->detector_->detect(image_second, keypoints_second);

    this->descriptor_->compute(image_first, keypoints_first, descriptors_first);
    this->descriptor_->compute(image_second, keypoints_second, descriptors_second);

    this->matcher_->match(descriptors_first, descriptors_second, matches);

    return matches;
}

/// @brief Detect matches between a pair of images and return filtered points
std::pair<std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>, std::vector<cv::DMatch>> 
    VisualOdometry::getMatches(const cv::Mat image_first, const cv::Mat image_second) {  
    
    std::vector<std::vector<cv::DMatch>> matches = {};
    std::vector<cv::KeyPoint> keypoints_first, keypoints_second = {};
    cv::Mat descriptors_first, descriptors_second;

    this->detector_->detectAndCompute(image_first, cv::noArray(), keypoints_first, descriptors_first);

    this->detector_->detectAndCompute(image_second, cv::noArray(), keypoints_second, descriptors_second);

    this->matcher_->knnMatch(descriptors_first, descriptors_second, matches, 2);

    // Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.5f;
    std::vector<cv::DMatch> good_matches = {};
    for (uint16_t i = 0; i < matches.size(); i++) {
        if (matches[i][0].distance < ratio_thresh * matches[i][1].distance) {
            good_matches.push_back(matches[i][0]);
        }
    }
    std::vector<cv::Point2f> q1, q2 = {};

    for (auto i : good_matches) { 
        q1.push_back(keypoints_first[i.queryIdx].pt);
        q2.push_back(keypoints_second[i.trainIdx].pt);
    }

    return {{q1, q2}, good_matches};
}

/// @brief Get the rotation and translation of the camera given a set of matching image keypoints
cv::Mat VisualOdometry::getPose(const std::vector<cv::Point2f> q1, const std::vector<cv::Point2f> q2) {
    cv::Mat R1, R2, t, ret;

    // Find essential matrix
    const cv::Mat E = cv::findEssentialMat(q1, q2, this->kl_);
    
    // Decompose essential matrix
    cv::decomposeEssentialMat(E, R1, R2, t);

    // Create 4 possible transformation matricies
    std::vector<cv::Mat> T = {};  // Transformations
    std::vector<cv::Mat> P = {};  // Projections
    const cv::Mat padding_T = (cv::Mat_<double>(1, 4) << 0.0, 0.0, 0.0, 1.0);
    const cv::Mat padding_K = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 0.0);

    // Homogenize K
    cv::Mat K;
    cv::hconcat(this->kl_, padding_K, K);

    int8_t optimal_idx = -1;
    uint16_t max_positive = 0;

    for (uint8_t i = 0; i < 4; ++i) {
        cv::Mat T;
        cv::Mat P;
        if (i % 2) {
            cv::hconcat(R2, t, T);
            t = -t;
        } else {
            cv::hconcat(R1, t, T);
        }
        cv::vconcat(T, padding_T, T);

        P = K * T;
        cv::Mat Q1;
        cv::Mat Q2;
        cv::triangulatePoints(this->pl_, P, q1, q2, Q1);

        Q1.convertTo(Q1, CV_64FC1);
        Q2.convertTo(Q2, CV_64FC1);

        Q2 = T * Q1;

        // Unhomogenize Q
        this->unHomogenize(Q1);  // x,y,z points in world coordinates
        this->unHomogenize(Q2);

        // Detect solution where all points are in front of camera eg possitive z values
        uint16_t sum = 0;
        for (uint16_t j = 0; j < Q1.cols; ++j) {
            if (Q1.row(Q1.rows - 1).at<double>(j) > 0) {
                sum++;
            }
            if (Q2.row(Q2.rows - 1).at<double>(j) > 0) {
                sum++;
            }
        }
        if (sum > max_positive) {
            max_positive = sum;
            optimal_idx = i;
        }
    }
    // Return coresponding rotation and translation
    switch (optimal_idx) {
        case 0: cv::hconcat(R1, t, ret);  break;
        case 1: cv::hconcat(R2, t, ret);  break;
        case 2: cv::hconcat(R1, -t, ret); break;
        case 3: cv::hconcat(R2, -t, ret); break;
    }

    cv::vconcat(ret, padding_T, ret);
    return ret;
}

/// @brief Unhomogenize a given matrix 
void VisualOdometry::unHomogenize(cv::Mat& mat) {
    cv::Mat last = mat.row(mat.rows - 1);
    cv::Mat temp = cv::repeat(last, mat.rows - 1, 1);
    mat = mat(cv::Range(0, mat.rows - 1), cv::Range(0, mat.cols));
    mat = mat / temp;
}

int main(int argc, char **argv) {
    std::ifstream calibration_data("KITTI_sequence_2/calib.txt");
    std::ofstream pose_predicted("pose_predicted_2.txt");

    // const cv::Mat Pl = (cv::Mat_<double>(3, 4) << 7.070912000000e+02, 0.000000000000e+00, 6.018873000000e+02, 0.000000000000e+00, 
    //                                      0.000000000000e+00, 7.070912000000e+02, 1.831104000000e+02, 0.000000000000e+00,
    //                                      0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00);

    const cv::Mat Pl = (cv::Mat_<double>(3, 4) << 7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 0.000000000000e+00,
                                         0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00,
                                         0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00);

    const cv::Mat Pr = Pl;
    
    // Instantiate VO object
    std::unique_ptr<VisualOdometry> VO = std::make_unique<VisualOdometry>(Pl, Pr);

    char *filepath_left_prev = new char[100];
    char *filepath_left = new char[100];
    char *filepath_right = new char[100];

    cv::Mat current_pose = (cv::Mat_<double>(4, 4) << 1.000000e+00, 9.043683e-12, 2.326809e-11, 1.110223e-16, 
                                                      9.043683e-12, 1.000000e+00, 2.392370e-10, 2.220446e-16 ,
                                                      2.326810e-11, 2.392370e-10, 9.999999e-01, -2.220446e-16,
                                                      0, 0, 0, 1);

    for(uint8_t i = 1; i <= num_images; ++i) {

        sprintf(filepath_left_prev, "../KITTI_sequence_2/image_l/0000%02i.png", i-1);
        sprintf(filepath_left, "../KITTI_sequence_2/image_l/0000%02i.png", i);
        sprintf(filepath_right, "../KITTI_sequence_2/image_r/0000%02i.png", i);

        const cv::Mat left_image_prev = cv::imread(filepath_left_prev, cv::ImreadModes::IMREAD_GRAYSCALE);
        const cv::Mat left_image = cv::imread(filepath_left, cv::ImreadModes::IMREAD_GRAYSCALE);
        const cv::Mat right_image = cv::imread(filepath_right, cv::ImreadModes::IMREAD_GRAYSCALE);

        // Error Handling
        if (left_image.empty() || right_image.empty()) {
            std::cout << "Image File "
                      << "Not Found" << std::endl;
            std::cin.get();
            return -1;
        }

        const std::vector<cv::KeyPoint> keypoints_left_prev = VO->detectKeypoints(left_image_prev);
        const std::vector<cv::KeyPoint> keypoints_left = VO->detectKeypoints(left_image);
        const std::pair<std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>, std::vector<cv::DMatch>> 
            q = VO->getMatches(left_image_prev, left_image);
        const cv::Mat transformation = VO->getPose(q.first.first, q.first.second);
        current_pose = current_pose * transformation.inv();

        std::cout<<"Current Pose: "<<current_pose << std::endl;
    
        pose_predicted << current_pose.at<double>(0, 3) << " "
                       << current_pose.at<double>(1, 3) << " "
                       << current_pose.at<double>(2, 3) << "\n";
        
        cv::Mat matches;
        cv::drawMatches(left_image_prev, keypoints_left_prev, left_image, keypoints_left, q.second, matches);
        cv::imshow("matches", matches);
        cv::waitKey(0);
        
    }

    pose_predicted.close();
    return 0;
}
    
