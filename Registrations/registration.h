#ifndef REGISTRATION_H
#define REGISTRATION_H

#include "opencv2/opencv.hpp"
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafilters.hpp>

using namespace std;
using namespace cv;

class ImageProcessor {
public:
    Mat loadImage(const string& path);
    Mat convertToGray(const Mat& image);

    cuda::GpuMat uploadToGPU(const Mat& image);
};


class FeatureMatcher {
public:
    FeatureMatcher(int maxFeatures);
    void detectAndCompute(const cuda::GpuMat& image, vector<KeyPoint>& keypoints, cuda::GpuMat& descriptors);

private:
    int maxFeatures;
    Ptr<cuda::ORB> detector;
};


class AffineTransformer {
public:
    Mat estimateAffineTransformation(const vector<Point2f>& sourcePoints, const vector<Point2f>& targetPoints);
    cuda::GpuMat warpImage(const cuda::GpuMat& source, const Mat& affineMatrix, const Size& targetSize);
};



class ImageRegistration {
public:
    ImageRegistration(const ImageProcessor& processor, const FeatureMatcher& matcher, const AffineTransformer& transformer);
    void processImages(const string& sourcePath, const string& targetPath);

private:
    ImageProcessor processor;
    FeatureMatcher matcher;
    AffineTransformer transformer;

    vector<Point2f> convertKeyPoints(const vector<KeyPoint>& keypoints);
    vector<DMatch> matchFeatures(const cuda::GpuMat& sourceDescriptors, const cuda::GpuMat& targetDescriptors);
    vector<DMatch> filterMatches(const vector<DMatch>& matches, double maxDistance);

    vector<Point2f> convertMatchesToPoints(const vector<DMatch>& matches, const vector<Point2f>& points);

    void displayImages(const Mat& source, const Mat& target, const cuda::GpuMat& warped);
};



#endif // REGISTRATION_H
