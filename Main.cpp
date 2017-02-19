// Main.cpp

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include<iostream>
#include<conio.h>           // may have to modify this line if not using Windows

// global constants ///////////////////////////////////////////////////////////////////////////////
const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_YELLOW = cv::Scalar(0.0, 255.0, 255.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 255.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);

const int MIN_PIXEL_WIDTH = 10;
const int MIN_PIXEL_HEIGHT = 10;

const double MAX_ASPECT_RATIO = 0.8;

const int MIN_PIXEL_AREA = 80;

// function prototypes ////////////////////////////////////////////////////////////////////////////
std::vector<std::vector<cv::Point> > findTrafficCones(cv::Mat imgOriginal);
bool isTrafficCone(std::vector<cv::Point> convexHull);
void drawGreenDotAtConeCenter(std::vector<cv::Point> trafficCone, cv::Mat &image);

///////////////////////////////////////////////////////////////////////////////////////////////////
int main() {

    cv::Mat imgOriginal;        // input image
    
    imgOriginal = cv::imread("image1.png");          // open image

    if (imgOriginal.empty()) {                                  // if unable to open image
        std::cout << "error: image not read from file\n\n";     // show error message on command line
        _getch();                                               // may have to modify this line if not using Windows
        return(0);                                              // and exit program
    }

    cv::imshow("imgOriginal", imgOriginal);

    std::vector<std::vector<cv::Point> > trafficCones = findTrafficCones(imgOriginal);
    
    cv::Mat imgOriginalWithCones = imgOriginal.clone();

    // draw yellow convex hull around outside of cones
    cv::drawContours(imgOriginalWithCones, trafficCones, -1, SCALAR_YELLOW, 2);

    for (auto &trafficCone : trafficCones) {        // for each found traffic cone . . .
        drawGreenDotAtConeCenter(trafficCone, imgOriginalWithCones);      // draw small green dot at center of mass of cone        
    }

    cv::imshow("imgOriginalWithCones", imgOriginalWithCones);

    if (trafficCones.size() <= 0) {
        std::cout << "\n" << "no traffic cones were found" << "\n\n";
    }
    else if (trafficCones.size() == 1) {
        std::cout << "\n" << "1 traffic cone was found" << "\n\n";
    } else if (trafficCones.size() > 1) {
        std::cout << "\n" << trafficCones.size() << " traffic cones were found" << "\n\n";
    }

    cv::waitKey(0);                 // hold windows open until user presses a key

    return(0);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<std::vector<cv::Point> > findTrafficCones(cv::Mat imgOriginal) {

    // declare images
    cv::Mat imgHSV;
    cv::Mat imgThreshLow;
    cv::Mat imgThreshHigh;
    cv::Mat imgThresh;
    cv::Mat imgThreshSmoothed;
    cv::Mat imgCanny;
    cv::Mat imgContours;
    cv::Mat imgAllConvexHulls;
    cv::Mat imgConvexHulls3To10;
    cv::Mat imgTrafficCones;
    cv::Mat imgTrafficConesWithOverlapsRemoved;

    // declare vectors
    std::vector<std::vector<cv::Point> > contours;
    std::vector<std::vector<cv::Point> > trafficCones;
    
    // convert to HSV color space
    cv::cvtColor(imgOriginal, imgHSV, CV_BGR2HSV);
    cv::imshow("imgHSV", imgHSV);

    // threshold on low range of HSV red
    cv::inRange(imgHSV, cv::Scalar(0, 135, 135), cv::Scalar(15, 255, 255), imgThreshLow);
    // threshold on high range of HSV red
    cv::inRange(imgHSV, cv::Scalar(159, 135, 135), cv::Scalar(179, 255, 255), imgThreshHigh);
    // combine (i.e. add) low and high thresh images
    cv::add(imgThreshLow, imgThreshHigh, imgThresh);
    cv::imshow("imgThresh", imgThresh);

    // open image (erode, then dilate)
    imgThreshSmoothed = imgThresh.clone();
    cv::Mat structuringElement3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::erode(imgThreshSmoothed, imgThreshSmoothed, structuringElement3x3);
    cv::dilate(imgThreshSmoothed, imgThreshSmoothed, structuringElement3x3);

    // smooth image (i.e. Gaussian blur)
    cv::GaussianBlur(imgThreshSmoothed, imgThreshSmoothed, cv::Size(3, 3), 0);

    // find Canny edges
    cv::Canny(imgThreshSmoothed, imgCanny, 80, 160);
    cv::imshow("imgCanny", imgCanny);

    // find and draw contours
    cv::findContours(imgCanny.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    imgContours = cv::Mat(imgOriginal.size(), CV_8UC3, SCALAR_BLACK);
    cv::drawContours(imgContours, contours, -1, SCALAR_WHITE);
    cv::imshow("imgContours", imgContours);
    
    // find convex hulls
    std::vector<std::vector<cv::Point> > allConvexHulls(contours.size());
    for (unsigned int i = 0; i < contours.size(); i++) {
        cv::convexHull(contours[i], allConvexHulls[i]);
    }
    imgAllConvexHulls = cv::Mat(imgOriginal.size(), CV_8UC3, SCALAR_BLACK);
    cv::drawContours(imgAllConvexHulls, allConvexHulls, -1, SCALAR_WHITE);
    cv::imshow("imgAllConvexHulls", imgAllConvexHulls);

    // loop through convex hulls, check if each is a traffic cone, add to vector of traffic cones if it is
    for (auto &convexHull : allConvexHulls) {
        if (isTrafficCone(convexHull)) {
            trafficCones.push_back(convexHull);
        }
    }

    imgTrafficCones = cv::Mat(imgOriginal.size(), CV_8UC3, SCALAR_BLACK);
    cv::drawContours(imgTrafficCones, trafficCones, -1, SCALAR_WHITE);
    cv::imshow("imgTrafficCones", imgTrafficCones);

    return trafficCones;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
bool isTrafficCone(std::vector<cv::Point> convexHull) {

    // first get dimensional information for the convex hull (bounding rect, bounding rect area, bounding rect aspect ratio, and y center
    cv::Rect boundingRect = cv::boundingRect(convexHull);
    int area = boundingRect.area();
    double aspectRatio = (double)boundingRect.width / (double)boundingRect.height;
    int yCenter = boundingRect.y + (int)((double)boundingRect.height / 2.0);
    
    // first do a gross dimensional check, return false if convex hull does not pass
    if (area < MIN_PIXEL_AREA || boundingRect.width < MIN_PIXEL_WIDTH || boundingRect.height < MIN_PIXEL_HEIGHT || aspectRatio > MAX_ASPECT_RATIO) {
        return false;
    }

    // now check if the convex Hull is pointing up

    // declare and populate a vector of all points above the y center, and all points below the y center
    std::vector<cv::Point> vectorOfPointsAboveCenter;
    std::vector<cv::Point> vectorOfPointsBelowCenter;
    for (auto &point : convexHull) {
        if (point.y < yCenter) vectorOfPointsAboveCenter.push_back(point);
        else vectorOfPointsBelowCenter.push_back(point);
    }

    // find the left most point below the y center
    int leftMostPointBelowCenter = vectorOfPointsBelowCenter[0].x;
    for (auto &point : vectorOfPointsBelowCenter) {
        if (point.x < leftMostPointBelowCenter) leftMostPointBelowCenter = point.x;
    }

    // find the right most point below the y center
    int rightMostPointBelowCenter = vectorOfPointsBelowCenter[0].x;
    for (auto &point : vectorOfPointsBelowCenter) {
        if (point.x > rightMostPointBelowCenter) rightMostPointBelowCenter = point.x;
    }

    // step through all the points above the y center
    for (auto &pointAboveCenter : vectorOfPointsAboveCenter) {
        // if any point above the y center is farther left or right than the extreme left and right below y center points, then the convex hull is not pointing up, so return false
        if (pointAboveCenter.x <= leftMostPointBelowCenter || pointAboveCenter.x >= rightMostPointBelowCenter) return false;
    }
    // if we get here, the convex hull has passed the gross dimensional check and the pointing up check, so we're convinced its a cone, so return true
    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawGreenDotAtConeCenter(std::vector<cv::Point> trafficCone, cv::Mat &image) {
    
    // find the contour moments
    cv::Moments moments = cv::moments(trafficCone);

    // using the moments, find the center of mass
    int xCenter = (int)(moments.m10 / moments.m00);
    int yCenter = (int)(moments.m01 / moments.m00);

    // draw the small green circle
    cv::circle(image, cv::Point(xCenter, yCenter), 3, SCALAR_GREEN, -1);
}

