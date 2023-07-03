#include <opencv2/opencv.hpp>

#include <vector>

#define sequence_number 1

class VisualOdometry {
   public:
    VisualOdometry();

    VisualOdometry(cv::Mat Pl, cv::Mat Pr);

    [[nodiscard]] inline const cv::Mat& getPl() const { return pl_; }
    [[nodiscard]] inline const cv::Mat& getPr() const { return pr_; }
    [[nodiscard]] inline const cv::Mat& getKl() const { return kl_; }
    [[nodiscard]] inline const cv::Mat& getKr() const { return kr_; }

    std::vector<cv::KeyPoint> detectKeypoints(const cv::Mat image);

    cv::Mat computeDescriptors(const cv::Mat image, std::vector<cv::KeyPoint> keypoints);

    std::vector<cv::DMatch> matchDescriptors(const cv::Mat descriptors_first, const cv::Mat descriptors_second);

    cv::Mat computeDisparity(const cv::Mat image_first, const cv::Mat image_second);

    std::pair<std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>, std::vector<cv::DMatch>> getMatches(const cv::Mat image_first, const cv::Mat image_second);

    cv::Mat getPose(const std::vector<cv::Point2f> q1, const std::vector<cv::Point2f> q2);

    void unHomogenize(cv::Mat& mat);

   private:
    // Camera matrix P
    cv::Mat pl_;
    cv::Mat pr_;

    // Intrinsic matrix K
    cv::Mat kl_;
    cv::Mat kr_;

    // Feature detector and matcher
    cv::Ptr<cv::FeatureDetector> detector_;
    cv::Ptr<cv::DescriptorExtractor> descriptor_;
    std::unique_ptr<cv::FlannBasedMatcher> matcher_;
    cv::Ptr<cv::StereoSGBM> disparity_generator_;

    // blocksize
    uint8_t block_size_;
};

std::vector<cv::Mat> loadData(std::ifstream& file);