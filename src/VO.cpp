#include "VO.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
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

std::vector<cv::Mat> loadData(std::ifstream& file) {
    std::vector<cv::Mat> ret;
    std::string line = "";
    while(std::getline(file, line)) { 
        std::istringstream data_str(line);
        std::vector<double> data(std::istream_iterator<double>(data_str), {});
        cv::Mat matrix = cv::Mat::ones(3, 4, CV_64FC1);
        std::memcpy(matrix.data, data.data(), data.size()*sizeof(double));
        ret.push_back(matrix);
    }
    return ret;
}


int main(int argc, char **argv) {

    std::ifstream calibration_data("../KITTI_sequence_" + std::to_string(sequence_number) + "/calib.txt");
    std::ifstream pose_ground_truth("../KITTI_sequence_" + std::to_string(sequence_number) + "/poses.txt");
    std::ofstream pose_predicted("../scripts/pose_predicted_" + std::to_string(sequence_number) + ".txt");

    const std::vector<cv::Mat> cal = loadData(calibration_data);
    const std::vector<cv::Mat> pose_gt = loadData(pose_ground_truth);

    const cv::Mat Pl = cal[0];
    const cv::Mat Pr = cal[1];
   
    // Instantiate VO object
    std::unique_ptr<VisualOdometry> VO = std::make_unique<VisualOdometry>(Pl, Pr);

    cv::Mat current_pose = pose_gt[0]; // Pose starts at ground truth
    const cv::Mat padding = (cv::Mat_<double>(1, 4, CV_64FC1) << 0.0, 0.0, 0.0, 1.0);
    current_pose.push_back(padding);

    std::filesystem::path path = std::filesystem::current_path().string() + 
        "/../KITTI_sequence_" + std::to_string(sequence_number) + "/image_l/";

    std::vector<std::filesystem::path> files;
    std::copy(std::filesystem::directory_iterator(path), std::filesystem::directory_iterator(), std::back_inserter(files));
    std::sort(files.begin(), files.end());

    for (auto file_it = files.begin(); file_it < files.end(); ++file_it) {

        if (file_it - files.begin() == 0) { continue; } // Skip first frame to collect image i-1

        const cv::Mat image_prev = cv::imread(files[file_it - files.begin() - 1], cv::ImreadModes::IMREAD_GRAYSCALE);
        const cv::Mat image = cv::imread(*file_it, cv::ImreadModes::IMREAD_GRAYSCALE);

        // Error Handling
        if (image_prev.empty() || image.empty()) {
            std::cout << "Image File "
                      << "Not Found" << std::endl;
            std::cin.get();
            return -1;
        }

        const std::vector<cv::KeyPoint> keypoints_prev = VO->detectKeypoints(image_prev);
        const std::vector<cv::KeyPoint> keypoints = VO->detectKeypoints(image);
        const std::pair<std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>, std::vector<cv::DMatch>> 
            q = VO->getMatches(image_prev, image);
        
        const cv::Mat transformation = VO->getPose(q.first.first, q.first.second);
        current_pose = current_pose * transformation.inv();
        std::cout<<"Current pose: " << current_pose << std::endl;
    
        pose_predicted << current_pose.at<double>(0, 3) << " "
                       << current_pose.at<double>(1, 3) << " "
                       << current_pose.at<double>(2, 3) << "\n";
        
        cv::Mat matches;
        cv::drawMatches(image_prev, keypoints_prev, image, keypoints, q.second, matches);
        cv::imshow("matches", matches);
        cv::waitKey(0);
    }

    pose_predicted.close();
    return 0;
}