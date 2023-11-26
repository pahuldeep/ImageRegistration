#include <iostream>
#include "registration.h"

// Basic processing
Mat ImageProcessor::loadImage(const string &path)
{
    Mat image = imread(path);
    if (image.empty()) {
        cerr << "Failed to load image: " << path << endl;
    }
    return image;
}
Mat ImageProcessor::convertToGray(const Mat& image) {
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);
    return grayImage;
}
cuda::GpuMat ImageProcessor::uploadToGPU(const Mat &image)
{
    cuda::GpuMat gpuImage;
    gpuImage.upload(image);
    return gpuImage;
}

// feature descriptions and matching
FeatureMatcher::FeatureMatcher(int maxFeatures) : maxFeatures(maxFeatures){
    detector = cuda::ORB::create(maxFeatures);
}

void FeatureMatcher::detectAndCompute(const cuda::GpuMat &image, vector<KeyPoint> &keypoints, cuda::GpuMat &descriptors)
{
    detector->detectAndCompute(image, cuda::GpuMat(), keypoints, descriptors);
}


// translation and warping
Mat AffineTransformer::estimateAffineTransformation(const vector<Point2f> &sourcePoints, const vector<Point2f> &targetPoints)
{
    if (sourcePoints.size() < 3 || targetPoints.size() < 3) {
        cerr << "Insufficient point correspondences for estimation." << endl;
        return Mat();
    }
    return estimateAffinePartial2D(sourcePoints, targetPoints);
}
cuda::GpuMat AffineTransformer::warpImage(const cuda::GpuMat &source, const Mat &affineMatrix, const Size &targetSize)
{
    cuda::GpuMat warpedImage;
    cuda::warpAffine(source, warpedImage, affineMatrix, targetSize);
    return warpedImage;
}



ImageRegistration::ImageRegistration(const ImageProcessor &processor, const FeatureMatcher &matcher, const AffineTransformer &transformer):
    processor(processor), matcher(matcher), transformer(transformer){}


void ImageRegistration::processImages(const string &sourcePath, const string &targetPath)
{
    Mat sourceImage = processor.loadImage(sourcePath);
    Mat targetImage = processor.loadImage(targetPath);

    Mat sourceGray = processor.convertToGray(sourceImage);
    Mat targetGray = processor.convertToGray(targetImage);

    cuda::GpuMat sourceGpuGray = processor.uploadToGPU(sourceGray);
    cuda::GpuMat targetGpuGray = processor.uploadToGPU(targetGray);

    vector<cv::KeyPoint> sourceKeypoints;
    vector<KeyPoint> targetKeypoints;

    cuda::GpuMat sourceDescriptors;
    cuda::GpuMat targetDescriptors;

    matcher.detectAndCompute(sourceGpuGray, sourceKeypoints, sourceDescriptors);
    matcher.detectAndCompute(targetGpuGray, targetKeypoints, targetDescriptors);

    vector<Point2f> sourcePoints = convertKeyPoints(sourceKeypoints);
    vector<Point2f> targetPoints = convertKeyPoints(targetKeypoints);

    vector<DMatch> matches = matchFeatures(sourceDescriptors, targetDescriptors);

    double maxDistance = 50;
    vector<DMatch> filteredMatches = filterMatches(matches, maxDistance);

    vector<Point2f> filteredSourcePoints = convertMatchesToPoints(filteredMatches, sourcePoints);
    vector<Point2f> filteredTargetPoints = convertMatchesToPoints(filteredMatches, targetPoints);

    cv::Mat affineMatrix = transformer.estimateAffineTransformation(filteredSourcePoints, filteredTargetPoints);

    if (!affineMatrix.empty()) {
        cuda::GpuMat warpedSource = transformer.warpImage(sourceGpuGray, affineMatrix, targetGpuGray.size());
        displayImages(sourceImage, targetImage, warpedSource);
    }
}

vector<Point2f> ImageRegistration::convertKeyPoints(const vector<KeyPoint> &keypoints)
{
    vector<Point2f> points;
    KeyPoint::convert(keypoints, points);
    return points;
}

vector<DMatch> ImageRegistration::matchFeatures(const cuda::GpuMat &sourceDescriptors, const cuda::GpuMat &targetDescriptors)
{
    Ptr<cuda::DescriptorMatcher> matcher = cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
    vector<DMatch> matches;
    matcher->match(sourceDescriptors, targetDescriptors, matches);
    return matches;
}

vector<DMatch> ImageRegistration::filterMatches(const vector<DMatch> &matches, double maxDistance)
{
    vector<DMatch> filteredMatches;
    copy_if(matches.begin(), matches.end(), back_inserter(filteredMatches), [maxDistance](const auto& match) {
        // cout << match.distance << '\n';
        return match.distance < maxDistance;
    });
    return filteredMatches;
}

vector<Point2f> ImageRegistration::convertMatchesToPoints(const vector<DMatch> &matches, const vector<Point2f> &points)
{
    vector<Point2f> result;
    transform(matches.begin(), matches.end(), back_inserter(result), [&points](const auto& match) {
        return points[match.queryIdx];
    });
    return result;
}

void ImageRegistration::displayImages(const Mat &source, const Mat &target, const cuda::GpuMat &warped)
{
    imshow("Source Image", source);
    imshow("Target Image", target);

    Mat warpedCPU;
    warped.download(warpedCPU);
    imshow("Warped Source Image", warpedCPU);

    waitKey(0);
}




