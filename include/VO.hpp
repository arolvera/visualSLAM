#include <opencv2/opencv.hpp> 

#include <vector>

#define num_images 50

class VisualOdometry {
 public:
    
    VisualOdometry();
    VisualOdometry(cv::Mat Pl, cv::Mat Pr);
    cv::Mat getPl() const { return Pl; }
    cv::Mat getPr() const { return Pr; }
    cv::Mat getKl() const { return Kl; }
    cv::Mat getKr() const { return Kr; }

    std::vector<cv::KeyPoint> detectKeypoints(cv::Mat image);
    cv::Mat computeDescriptors(cv::Mat image, std::vector<cv::KeyPoint> keypoints);
    std::vector<cv::DMatch> matchDescriptors(cv::Mat descriptors_first, cv::Mat descriptors_second);

    cv::Mat computeDisparity(cv::Mat image_first, cv::Mat image_second);

    std::pair<std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>, std::vector<cv::DMatch>> getMatches(cv::Mat image_first, cv::Mat image_second);

    cv::Mat getPose(std::vector<cv::Point2f> q1, std::vector<cv::Point2f> q2);

    cv::Mat unHomogenize(cv::Mat mat);
    
 private:
    
    // Camera matrix P
    cv::Mat Pl;
    cv::Mat Pr;
    
    // Intrinsic matrix K
    cv::Mat Kl;
    cv::Mat Kr;
        
    // Feature detector and matcher
    cv::Ptr<cv::FeatureDetector> detector;
    cv::Ptr<cv::DescriptorExtractor> descriptor;
    std::unique_ptr<cv::FlannBasedMatcher> matcher;

};