#include "iostream"
#include "vector"
#include "math.h"
#include "string"

#include "opencv2/opencv.hpp"
#include "opencv2/face.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

// https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html

int main()
{
    // EDGE DETTECTION - CANNY / SOBEL

    cv::Mat img, gray, blurred, edge;

    img = cv::imread("image.jpeg");

    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    cv::imshow("original image", img);
    cv::imshow("gray image", gray);

    // laplacian variable
    int kernel_size = 3;
    int ddepth = CV_16S;

    // canny edge variable
    int lower_threshold = 0;
    int max_low_threshold = 100;

    cv::GaussianBlur(img, blurred, cv::Size(3, 3), 0, 0);
    cv::Laplacian(blurred, edge, ddepth, kernel_size);

    cv::convertScaleAbs(edge, edge); // converts to CV_8U from CV_16S

    cv::imshow("blurred image", blurred);
    cv::imshow("edge image", edge);

    cv::waitKey(0);

}

/*

// Line / Circle Detection -- Hough Transform
int main()
{
    cv::Mat imgLines, imgCircles, detectImgLines, detectImgLinesP, detectImgCircles;

    imgLines = cv::imread("lines2.jpeg");
    imgCircles = cv::imread("image.png");

    imgLines.copyTo(detectImgLines);
    imgLines.copyTo(detectImgLinesP);
    imgCircles.copyTo(detectImgCircles);

    cv::Canny(imgLines, detectImgLines, 200, 255);

    std::vector<cv::Vec2f> lines;

    cv::HoughLines(detectImgLines, lines, 1, CV_PI, 150);

    for (size_t i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0];
        float theta = lines[i][1];

        cv::Point pt1, pt2;

        double a = cos(theta);
        double b = sin(theta);
        double x0 = a * rho;
        double y0 = b * rho;

        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));

        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));

        cv::line(detectImgLines, pt1, pt2, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
    };

    /*
    cv::imshow("imgLines", imgLines);
    cv::imshow("detectImgLines", detectImgLines);

    // Probability Hough Transform

    cv::Canny(imgLines, detectImgLinesP, 200, 255);

    std::vector<cv::Vec4i> linesP;

    cv::HoughLinesP(detectImgLinesP, lines, 1, CV_PI / 180, 50, 50, 10);

    for (size_t i = 0; i < lines.size(); i++)
    {
        cv::Vec4i l = linesP[i];
        cv::line(detectImgLinesP, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
    };

    cv::imshow("imgLines", imgLines);
    cv::imshow("detectImgLinesP", detectImgLinesP);

    cv::waitKey(0);
    

// Circle

cv::Mat gray;

cv::cvtColor(imgCircles, gray, cv::COLOR_BGR2GRAY);
cv::medianBlur(gray, gray, 5);

std::vector<cv::Vec3f> circles;

cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1, gray.rows / 16, 100, 30, 0, 500);

for (size_t i = 0; i < circles.size(); i++)
{
    cv::Vec3i c = circles[i];

    cv::Point center = cv::Point(c[0], c[1]);

    cv::circle(detectImgCircles, center, 1, cv::Scalar(0, 100, 100), 3, cv::LINE_AA);

    size_t radius = c[2];

    cv::circle(detectImgCircles, center, radius, cv::Scalar(255, 0, 255), 3, cv::LINE_AA);

    std::cout << "center: " << center << " radius: " << radius << std::endl;
};

cv::imshow("imgCircles", imgCircles);
cv::imshow("detectImgCircles", detectImgCircles);

cv::waitKey(0);

return EXIT_SUCCESS;
}
;

* /

    /*

    int main()
    {
        cv::Mat image = cv::imread("image.png",cv::IMREAD_GRAYSCALE);

        cv::equalizeHist(image, image);

        cv::imshow("image", image);
        cv::waitKey(0);
    }

    */

    // STERIO VISION

    /*

    class SterioVision
    {
    public:
        SterioVision(float baseline, float alpha, float focalLength) : baseline(baseline), alpha(alpha), focalLength(focalLength) {}

        // Callibrate the frames
        void undistortFrame(cv::Mat &frame);

        // Add HSV filter
        cv::Mat addHSVFilter(cv::Mat &frame, int camera);

        // find the circle in the frame - only find the largest one - reduce false positives
        cv::Point findCircle(cv::Mat &frame, cv::Mat &mask);

        // calculate the depth
        float calulateDepth(cv::Point circleLeft, cv::Point circleRight, cv::Mat &leftFrame, cv::Mat &rightFrame);

    private:
        float baseline = 0.0f;
        float alpha = 0.0f;
        float focalLength = 0.0f;
    };

    void SterioVision::undistortFrame(cv::Mat &frame)
    {
        cv::undistort(frame, frame, focalLength, baseline);
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(frame, frame, cv::Size(5, 5), 0, 0);
        cv::equalizeHist(frame, frame);
        cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
        cv::resize(frame, frame, cv::Size(480, 480));
        cv::imshow("undistorted", frame);
        cv::waitKey(0);
    }

    cv::Mat SterioVision::addHSVFilter(cv::Mat &frame, int camera)
    {
        cv::GaussianBlur(frame, frame, cv::Size(5, 5), 0, 0);
        cv::cvtColor(frame, frame, cv::COLOR_BGR2HSV);

        cv::Mat mask;

        std::vector<int> lowerLimitRight = {60, 110, 50};
        std::vector<int> uppweLimitRight = {255, 255, 255};
        std::vector<int> lowerLimitLeft = {140, 110, 50};
        std::vector<int> upperLimitLeft = {255, 255, 255};

        if (camera == 1)
        {
            cv::inRange(frame, lowerLimitRight, uppweLimitRight, mask);
        }
        else
        {
            cv::inRange(frame, lowerLimitLeft, upperLimitLeft, mask);
        };

        cv::erode(mask, mask, cv::Mat(), cv::Point(3, 3), 1);
        cv::dilate(mask, mask, cv::Mat(), cv::Point(3, 3), 1);

        return mask;
    };

    cv::Point SterioVision::findCircle(cv::Mat &frame, cv::Mat &mask)
    {
        std::vector<std::vector<cv::Point>> contours;

        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // find the bigger contour
        std::sort(contours.begin(), contours.end(), [](const std::vector<cv::Point> &a, const std::vector<cv::Point> &b)
                  { return cv::contourArea(a) > cv::contourArea(b); });

        if (contours.size() > 0)
        {
            std::vector<cv::Point> largetContour = contours[contours.size() - 1];

            cv::Point2f center;
            float radius;

            cv::minEnclosingCircle(largetContour, center, radius);
            cv::Moments moments = cv::moments(largetContour);
            cv::Point centerOfMass = cv::Point(moments.m10 / moments.m00, moments.m01 / moments.m00);

            if (radius > 10)
            {
                cv::circle(frame, centerOfMass, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
                cv::circle(frame, center, (int)radius, cv::Scalar(0, 255, 0), 3, 8, 0);
                cv::putText(frame, std::to_string(radius), center, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            };

            return centerOfMass;
        };

        return cv::Point(0, 0);
    };

    float SterioVision::calulateDepth(cv::Point circleLeft, cv::Point circleRight, cv::Mat &leftFrame, cv::Mat &rightFrame)
    {
        int focalPixels = 0;

        if (rightFrame.cols == leftFrame.cols)
        {
            focalPixels = (rightFrame.cols * 0.5) / (tan(alpha * CV_PI / 180));
        }
        else
        {
            std::cout << "ERROR: left and right frames have different sizes" << std::endl;
        };

        int xLeft = circleLeft.x;
        int xRight = circleRight.x;

        int disparity = xLeft - xRight;

        float zDepth = (float(focalPixels) * baseline) / float(disparity);

        return abs(zDepth);
    };

    int main()
    {

        float baseline = 7.0f;
        float alpha = 6.0f;
        float focalLength = 56.0f;

        cv::Mat leftFrame, rightFrame;
        cv::VideoCapture capRight(1);
        cv::VideoCapture capLeft(0);

        SterioVision sterioVision(baseline, alpha, focalLength);

        if (!capRight.isOpened() || !capLeft.isOpened())
        {
            std::cout << "Could not open the video device" << std::endl;
            return -1;
        };

        cv::Mat leftMask, rightMask;
        cv::Mat leftResFrame, rightResFrame;
        cv::Point leftCircle, rightCircle;

        float depth = 0;

        while (true)
        {
            capLeft.read(leftFrame);
            capRight.read(rightFrame);

            leftMask = sterioVision.addHSVFilter(leftFrame, 1);
            rightMask = sterioVision.addHSVFilter(rightFrame, 2);

            cv::bitwise_and(leftFrame, leftFrame, leftResFrame, leftMask);
            cv::bitwise_and(rightFrame, rightFrame, rightResFrame, rightMask);

            leftCircle = sterioVision.findCircle(leftResFrame, leftMask);
            rightCircle = sterioVision.findCircle(rightResFrame, rightMask);

            if (!leftCircle.x || !rightCircle.x)
            {
                cv::putText(leftFrame, "No Circle", cv::Point(75, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
                cv::putText(rightFrame, "No Circle", cv::Point(75, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            }
            else
            {
                depth = sterioVision.calulateDepth(leftCircle, rightCircle, leftFrame, rightFrame);
                std::cout << "depth" << depth << std::endl;

                cv::putText(leftFrame, std::to_string(depth), cv::Point(75, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
                cv::putText(rightFrame, std::to_string(depth), cv::Point(75, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            };

            cv::imshow("leftFrame", leftFrame);
            cv::imshow("rightFrame", rightFrame);
            cv::imshow("leftMask", leftMask);
            cv::imshow("rightMask", rightMask);

            if ((char)cv::waitKey(1) == 'x')
            {
                break;
            };
        };

        capLeft.release();
        capRight.release();

        cv::destroyAllWindows();

        return EXIT_SUCCESS;
    };

    */
    /*

    int main()
    {
        std::vector<cv::String> files;
        cv::glob("chess_images.png", files, true);
        cv::Size patternSize = cv::Size(25 - 1, 18 - 1);

        std::vector<std::vector<cv::Point2f>> q(files.size());
        std::vector<std::vector<cv::Point3f>> Q;

        int checkBoard[2] = {25, 18};
        int fieldSize = 15;

        std::vector<cv::Point3f> objP;

        for (size_t i = 0; i < checkBoard[1]; i++) // Corrected the loop termination condition
        {
            for (size_t j = 0; j < checkBoard[0]; j++) // Corrected the inner loop variable name
            {
                objP.push_back(cv::Point3f(fieldSize * j, fieldSize * i, 0.0f)); // Corrected the Point3f initialization
            }
        }

        std::vector<cv::Point2f> imgPoint;

        for (size_t i = 0; i < files.size(); i++) // Changed to use size_t and fixed the loop termination condition
        {
            std::cout << std::string(files[i]) << std::endl;

            cv::Mat img = cv::imread(files[i]);

            cv::Mat grayImg;
            cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY); // Changed COLOR_RGB2GRAY to COLOR_BGR2GRAY

            bool patternFound = cv::findChessboardCorners(grayImg, patternSize, q[i], cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);

            if (patternFound)
            {
                cv::cornerSubPix(grayImg, q[i], cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
                Q.push_back(objP);
            }

            cv::drawChessboardCorners(img, patternSize, q[i], patternFound);

            cv::imshow("img", img);
            cv::waitKey(0);
        }

        cv::Matx33f K(cv::Matx33f::eye());  // Fixed the initialization of intrinsic camera matrix
        cv::Vec<float, 5> k(0, 0, 0, 0, 0); // Fixed the initialization of distortion coefficients
        std::vector<cv::Mat> rvecs, tvecs;  // Fixed the vector type
        std::vector<double> perViewErrors;  // Removed stdIntrinsics and stdExtrinsics as they are unused
        int flags = cv::CALIB_FIX_ASPECT_RATIO + cv::CALIB_FIX_K3 +
                    cv::CALIB_ZERO_TANGENT_DIST + cv::CALIB_FIX_PRINCIPAL_POINT;
        cv::Size frameSize(1440, 1080);
        std::cout << "Calibrating..." << std::endl;
        float error = cv::calibrateCamera(Q, q, frameSize, K, k, rvecs, tvecs, flags); // Changed the order of Q and q
        std::cout << "Reprojection error = " << error << "\nK =\n"
                  << K << "\nk=\n"
                  << k << std::endl;

        cv::Mat mapx, mapy;
        cv::initUndistortRectifyMap(K, k, cv::Mat(), K, frameSize, CV_32FC1, mapx, mapy); // Fixed the initialization of rectification maps

        for (auto const &f : files)
        {
            std::cout << std::string(f) << std::endl;
            cv::Mat img = cv::imread(f, cv::IMREAD_COLOR);
            cv::Mat imgUndistorted;
            cv::remap(img, imgUndistorted, mapx, mapy, cv::INTER_LINEAR);
            cv::imshow("undistorted image", imgUndistorted);
            cv::waitKey(0);
        }

        return 0;
    }

    */
    /*
    int main()
    {
        // video capture from camera
        cv::VideoCapture cap(0,0);
        cv::Mat frame;

        while (true)
        {
            cap >> frame;
            cv::imshow("frame", frame);
            if (cv::waitKey(1) == 27)
            {
                break;
            }
        }

        return 0;
    }
    */

    /*
    // https://www.youtube.com/watch?v=gpu9p3d53fg&list=PLkmvobsnE0GHMmTF7GTzJnCISue1L9fJn&index=10
    int main()
    {
        cv::Mat image = cv::imread("image.png");

        cv::Mat histogram;

        int histSize = 256;
        const int channels[] = {0, 1, 2};
        float range[] = {0.0, 256.0};
        const float *histRange[] = {range};

        cv::calcHist(&image, 1, channels, cv::Mat(), histogram, 1, &histSize, histRange);
        cv::normalize(histogram, histogram, 0, 255, cv::NORM_MINMAX, -1, cv::Mat());

        cv::imshow("image", image);
        cv::imshow("histogram", histogram);

        cv::waitKey(0);
    }

    */

    /*

    int main()
    {
        cv::Mat img = cv::imread("image.png");

        cv::Mat dst;

        cv::getRectSubPix(img, cv::Size(100, 100), cv::Point(500, 450), dst);

        cv::Rect roi(298, 298, 5, 5);
        cv::Mat roi_dst = img(roi);

        cv::imshow("dst", dst);
        cv::imshow("roi_dst", roi_dst);
        cv::waitKey(0);

    }

    */

    /*
    using namespace cv;

    int main() {
        // Create a black image
        int width = 500;
        int height = 500;
        Mat image(height, width, CV_8UC3, Scalar(0, 0, 0));

        // Center of the image
        Point center(width / 2, height / 2);

        // Radius of the circle
        int radius = height / 2;

        // Draw the rainbow circle
        for (int i = 0; i < 360; ++i) {
            // Convert angle to radians
            double angle = i * CV_PI / 180;

            // Calculate color
            Scalar color;
            if (i < 60) {
                color = Scalar(0, 0, 255); // Red
            } else if (i < 120) {
                color = Scalar(0, 255, 255); // Yellow
            } else if (i < 180) {
                color = Scalar(0, 255, 0); // Green
            } else if (i < 240) {
                color = Scalar(255, 255, 0); // Cyan
            } else if (i < 300) {
                color = Scalar(255, 0, 0); // Blue
            } else {
                color = Scalar(255, 0, 255); // Magenta
            }

            // Polar to Cartesian coordinates
            int x = center.x + radius * cos(angle);
            int y = center.y + radius * sin(angle);

            // Draw the point
            circle(image, Point(x, y), 1, color, FILLED);
        }

        // Display the image
        imshow("Rainbow Circle", image);
        waitKey(0);
        destroyAllWindows();

        return 0;
    }

    */

    /*
    int main()
    {
        cv::Mat _img = cv::imread("image.png");

        cv::Mat _dst = cv::Mat::zeros(_img.size(), _img.type());
        cv::Mat map_x = cv::Mat::zeros(_img.size(), CV_32FC1);
        cv::Mat map_y = cv::Mat::zeros(_img.size(), CV_32FC1);

        for (int i = 0; i < _img.rows; i++)
        {
            for (int j = 0; j < _img.cols; j++)
            {
                map_x.at<float>(i, j) = static_cast<float>(_img.cols - i);
                map_y.at<float>(i, j) = static_cast<float>(_img.rows - j);
            }
        }

        cv::remap(_img, _dst, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));

        map_x.convertTo(map_x, CV_8UC1);
        map_y.convertTo(map_y, CV_8UC1);

        cv::imshow("img", _img);
        cv::imshow("dst", _dst);

        cv::imshow("map_x", map_x);
        cv::imshow("map_y", map_y);

        cv::waitKey(0);
    }

    */
    /*
    int main()
    {
        cv::Mat _16sc2Mat = cv::Mat::zeros(cv::Size(5, 5), CV_32FC1);
        cv::Mat _16uc1Mat = cv::Mat::zeros(cv::Size(5, 5), CV_32FC1);

        cv::Mat out2 = cv::Mat::zeros(cv::Size(480, 480), CV_16SC2);
        cv::Mat out1 = cv::Mat::zeros(cv::Size(480, 480), CV_16SC1);

        // _16sc2Mat.at<cv::Vec3s>(0, 0)[0] = 255;
        _16sc2Mat.at<float>(cv::Point(0, 4)) = 8.1;
        _16sc2Mat.at<float>(cv::Point(0, 3)) = 5.8;
        _16uc1Mat.at<float>(cv::Point(1, 3)) = 3.4;

        cv::convertMaps(_16sc2Mat, _16sc2Mat, out1, out2, CV_16SC2);
    }

    CV_8U - 8-bit unsigned integers ( 0..255 )
    CV_85 - 8-bit signed integers (-128.. 127 )
    CV_16U - 16-bit unsigned integers ( 0..65535 )
    CV_165 -
    16-bit signed integers (-32768..32767 )
    CV_
    32S
    32-bit signed integers (-2147483648.. 2147483647 )
    CV_32F -
    32-bit floating-point numbers ( -FLT_MAX.. FLT_MAX, INF, NAN)
    CV
    64F - 64-bit floating-point numbers (-DBL_MAX. DBL_MAX, INF, NAN)
    int main()
    {
        cv::Mat _8uc1 = cv::Mat::ones(cv::Size(3, 4), CV_8UC1);

        std::cout << _8uc1 << std::endl;

        std::cout << _8uc1.depth() << std::endl;
        std::cout << _8uc1.channels() << std::endl;
        std::cout << _8uc1.cols  << std::endl;
    }
    */

    /*
    int main()
    {
        cv::Mat img = cv::Mat(480, 480, CV_8UC3, cv::Scalar(0, 0, 0));

        cv::namedWindow("img", cv::WINDOW_NORMAL);
        cv::resizeWindow("img", 480, 480);

        for (int i = 0; i < img.rows; i++)
        {
            for (int j = 0; j < img.cols; j++)
            {
                img.at<cv::Vec3b>(i, j)[0] = 255;
                img.at<cv::Vec3b>(i, j)[1] = 255;
                img.at<cv::Vec3b>(i, j)[2] = random() % 256;
            }
        }

        cv::imshow("img", img);
        cv::resize(img, img, cv::Size(480, 480));
        cv::waitKey(0);

        return EXIT_SUCCESS;
    }


    int main()
    {
        cv::Mat img = cv::imread("image.png");
        cv::resize(img, img, cv::Size(480, 480));

        std::vector<cv::Mat> bgr_planes;
        cv::split(img, bgr_planes);

        int histsize = 256;
        float range[] = {0.0, 256.0};

        const float *histRange = {range};

        bool uniform = true, acumulate = false;

        cv::Mat b_hist, g_hist, r_hist;

        cv::calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histsize, &histRange, uniform, acumulate);
        cv::calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histsize, &histRange, uniform, acumulate);
        cv::calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histsize, &histRange, uniform, acumulate);

        int hist_w = 512, hist_h = 400;
        int bin_w = cvRound((double)hist_w / histsize);

        cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

        cv::normalize(b_hist, b_hist, 0, hist_h, cv::NORM_MINMAX, -1, cv::Mat());
        cv::normalize(g_hist, g_hist, 0, hist_h, cv::NORM_MINMAX, -1, cv::Mat());
        cv::normalize(r_hist, r_hist, 0, hist_h, cv::NORM_MINMAX, -1, cv::Mat());

        for (int i = 1; i < histsize; i++)
        {
            cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
                     cv::Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
                     cv::Scalar(255, 0, 0), 2, 8, 0);

            cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
                     cv::Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
                     cv::Scalar(0, 255, 0), 2, 8, 0);

            cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
                     cv::Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
                     cv::Scalar(0, 0, 255), 2, 8, 0);
        };

        cv::imshow("img", img);
        cv::imshow("Histogram", histImage);

        cv::waitKey(0);
    };

    int main()
    {
        cv::Mat img = cv::imread("img.png",cv::IMREAD_GRAYSCALE);
        cv::equalizeHist(img, img);

        cv::imshow("img", img);
        cv::waitKey(0);
    }



    int main()
    {
        cv::Mat img = cv::imread("img.png");
        cv::resize(img, img, cv::Size(480, 480), 0, 0, cv::INTER_NEAREST);

        cv::Mat grayImg;
        cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY); // Convert to grayscale

        cv::MatND histogram;
        int histSize = 256;
        const int channels[] = {0}; // For grayscale image
        float range[] = {0.0, 256.0};
        const float* histRange[] = {range};

        cv::calcHist(&grayImg, 1, channels, cv::Mat(), histogram, 1, &histSize, histRange);

        int histogramWidth = 512 , histogramHeight = 400;
        int binWidth = cvRound((double)histogramWidth / histSize);
        cv::Mat histImage = cv::Mat(histogramHeight, histogramWidth, CV_8UC3, cv::Scalar(0,0,0));
        cv::normalize(histogram, histogram, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

        for (int i = 1; i < histSize; i++)
        {
            cv::line(histImage, cv::Point(binWidth*(i-1), histogramHeight - cvRound(histogram.at<float>(i-1))),
                    cv::Point(binWidth*(i), histogramHeight - cvRound(histogram.at<float>(i))),
                    cv::Scalar(255, 255, 255), 2, 8, 0);
        }

        cv::imshow("img", img);
        cv::imshow("Histogram", histImage);
        cv::waitKey(0);

        return 0;
    };


    int main()
    {
        const int MAX_VALUE_H = 360 / 2;
        const int MAX_VALUE = 255;

        // cv::Mat M(7,7, CV_8UC1, cv::Scalar(1,3));

        cv::Mat img = cv::imread("img.png");

        cv::resize(img, img, cv::Size(480, 480));

        std::vector<int> lowerBound = {170, 80, 50};

        int low_h = lowerBound[0];
        int low_s = lowerBound[1];
        int low_v = lowerBound[2];

        int high_h = MAX_VALUE_H;
        int high_s = MAX_VALUE;
        int high_v = MAX_VALUE;

        cv::Mat hsvImg, imgThreshold;
        cv::Mat medianBlurImg, guassianBlurImg;

        // RGB to HSV
        cv::cvtColor(img, hsvImg, cv::COLOR_BGR2HSV);

        // Detect the object in the masked HSV image
        cv::inRange(hsvImg, cv::Scalar(low_h, low_s, low_v), cv::Scalar(high_h, high_s, high_v), imgThreshold);

        cv::medianBlur(img, medianBlurImg, 9);
        cv::GaussianBlur(img, guassianBlurImg, cv::Size(1, 1), 9, 9);

        cv::imshow("orignal image", img);
        cv::imshow("imgThreshold", imgThreshold);
        cv::imshow("medianBlurImg", medianBlurImg);
        cv::imshow("guassianBlurImg", guassianBlurImg);

        cv::waitKey(0);
    }


    cv::Mat imageGrayScale(cv::Mat &img)
    {
        cv::Mat grayImg = cv::Mat(img.rows, img.cols, CV_8UC1);

        for (size_t i = 0; i < img.rows; i++)
        {
            for (size_t j = 0; j < img.cols; j++)
            {
                cv::Vec3b bgrPixel = img.at<cv::Vec3b>(i, j);

                // Calculate grayscale using luminosity method
                uchar grayScale = static_cast<uchar>(0.2126 * bgrPixel[2] + 0.7152 * bgrPixel[1] + 0.0722 * bgrPixel[0]);

                grayImg.at<uchar>(i, j) = grayScale; // Set grayscale value in the grayscale image
            }
        }

        return grayImg;
    }

    int main()
    {
        cv::Mat _img = cv::imread("img.png");

        cv::namedWindow("image", cv::WINDOW_NORMAL);
        cv::resizeWindow("image", 480, 480);
        cv::imshow("image", _img);

        cv::Mat grayImg = imageGrayScale(_img); // Convert image to grayscale
        cv::namedWindow("output", cv::WINDOW_NORMAL);
        cv::resizeWindow("output", 480, 480);
        cv::imshow("output", grayImg);

        cv::waitKey(0);
    }


    int main()
    {
        // affine transformation :: dst (i) = (x',, y), srci) = (xi, yi), i = 1, 2, 3, 4
        // dst (x. y) = src(Miix + Mizy + M13, M21x + M2zy + M23)

        int cnt = 1;

        while (cnt++)
        {
            cv::Mat inputImg = cv::imread("img.png");
            cv::Mat outputImg = cv::Mat::zeros(cv::Size(inputImg.cols, inputImg.rows), inputImg.type());

            cv::namedWindow("input", 0);
            cv::namedWindow("output", 0);

            cv::Point2f inputMat[3];
            cv::Point2f ouputMat[3];

            inputMat[0] = cv::Point2f(0.0, 0.0);
            inputMat[1] = cv::Point2f(inputImg.cols, 0.0);
            inputMat[2] = cv::Point2f(0, inputImg.rows);

            ouputMat[0] = cv::Point2f(0, 200);
            ouputMat[1] = cv::Point2f(500, 100 + cnt);
            ouputMat[2] = cv::Point2f(170, 520 - cnt);

            std::cout << cnt << std::endl;

            cv::Mat M = cv::getAffineTransform(inputMat, ouputMat);

            cv::warpAffine(inputImg, outputImg, M, outputImg.size());

            cv::imshow("input", inputImg);
            cv::imshow("output", outputImg);

            cv::waitKey(0);
        };
    }


    int main()
    {
        cv::Mat image = cv::imread("img.png");

        cv::resize(image, image, cv::Size(480, 480));

        // chaning the persepective of the image ::
        cv::Point2f src[] = {cv::Point(741, 122), cv::Point(1228, 453), cv::Point(240, 919), cv::Point(735, 1200)};
        cv::Point2f dst[] = {cv::Point2f(0, 0), cv::Point2f(image.cols, 0), cv::Point2f(image.cols, image.rows), cv::Point2f(0, image.rows)};

        cv::Mat matrix = cv::getPerspectiveTransform(src, dst);

        cv::warpPerspective(image, image, matrix, image.size());

        cv::imshow("image", image);

        cv::waitKey(0);
    }


    int main()
    {
        cv::Mat frame = cv::Mat::zeros(cv::Size(480, 480), CV_8UC3);

        cv::namedWindow("Frame", 0);

        std::vector<std::vector<cv::Point>> contours = {{cv::Point(100, 100), cv::Point(200, 100), cv::Point(200, 200), cv::Point(100, 200)}};

        cv::drawContours(frame, contours, 0, cv::Scalar(255, 255, 255), 2);

        cv::resize(frame, frame, cv::Size(480, 480));

        cv::imshow("Frame", frame);

        cv::waitKey(0);
    }


    void onTrackbarX(int, void *) {}
    void onTrackbarY(int, void *) {}

    int main()
    {
        cv::Mat masking = cv::Mat::zeros(cv::Size(480, 480), CV_8UC3);

        int pointX = 100;
        int pointY = 100;
        int radius = 3;
        int thickness = 1;
        int ratio = 1;

        int startAngle = 0;
        int endAngle = 360;
        int angle = 0;

        cv::namedWindow("masking", 0);

        cv::createTrackbar("X", "masking", &pointX, 480, onTrackbarX);
        cv::createTrackbar("Y", "masking", &pointY, 480, onTrackbarY);
        cv::createTrackbar("Radius", "masking", &radius, 100 , onTrackbarX);
        cv::createTrackbar("Thickness", "masking", &thickness, 100 , onTrackbarX);
        cv::createTrackbar("Ratio", "masking", &ratio, 100 , onTrackbarX);

        cv::createTrackbar("Angle", "masking", &angle, 360, onTrackbarX);
        cv::createTrackbar("Start Angle", "masking", &startAngle, 360, onTrackbarX);
        cv::createTrackbar("End Angle", "masking", &endAngle, 360, onTrackbarY);

        cv::moveWindow("masking", 100, 10);

        while (true)
        {
            // cv::circle(masking, cv::Point(pointX, pointY), radius, cv::Scalar(255, 255, 255), thickness, cv::LINE_8);
            // cv::arrowedLine(masking, cv::Point(pointX, pointY), cv::Point(100, 100), cv::Scalar(255, 255, 255), thickness, cv::LINE_AA,0,(double)ratio/10);
            // cv::line(masking, cv::Point(pointX, pointY), cv::Point(100, 100), cv::Scalar(255, 255, 255), thickness, cv::LINE_AA,0);
            // cv::circle(masking, cv::Point(pointX, pointY), radius, cv::Scalar(255, 255, 255), thickness, cv::LINE_8);
            // cv::ellipse(masking, cv::Point(pointX, pointY), cv::Size(radius, radius), angle, startAngle, endAngle, cv::Scalar(255, 255, 255), thickness, cv::LINE_8, 0);
            // cv::ellipse(masking,cv::RotatedRect(cv::Point(pointX, pointY), cv::Size(radius, radius), angle), cv::Scalar(255, 255, 255), thickness, cv::LINE_8, 0);

            std::vector<cv::Point>points;

            cv::ellipse2Poly(cv::Point(pointX, pointY), cv::Size(radius, radius), angle, 0, 360, 1, points);

            for(int i = 0 ; i < points.size() ; i++)
            {
                masking.at<cv::Vec3b>(points[i]) = cv::Vec3b(255, 255, 255);

                masking.at<cv::Vec3b>(points[i])[0] = 0;
                masking.at<cv::Vec3b>(points[i])[1] = 255;
                masking.at<cv::Vec3b>(points[i])[2] = 255;
            }

            cv::imshow("masking", masking);

            masking = cv::Mat::zeros(cv::Size(480, 480), CV_8UC3);
            if ((char)cv::waitKey(1) == 'q') break;
        }

        return 0;
    }


    int main()
    {
        cv::Mat image = cv::imread("img.png");
        cv::resize(image, image, cv::Size(480, 480));

        cv::Mat outputImage,outputImageLaplacian;

        int dx = 1;
        int dy = 1;
        int sobelKernelSize = 3;
        int scalerFactor = 1;
        int deltaValue = 1;

        while (true)
        {
            cv::Sobel(image, outputImage, image.depth(), dx, dy, sobelKernelSize, scalerFactor, deltaValue, cv::BORDER_DEFAULT);
            cv::Laplacian(image, outputImageLaplacian, image.depth(), sobelKernelSize, scalerFactor, deltaValue, cv::BORDER_DEFAULT);

            int c = cv::waitKey(1);

            if ((char)c == 'x')
            {
                if (dx && dy)
                {
                    dx = 0;
                }
                else
                {
                    dx = 1;
                }
            };

            if ((char)c == 'y')
            {
                if (dx && dy)
                {
                    dy = 0;
                }
                else
                {
                    dy = 1;
                }
            };

            if ((char)c == 's')
            {
                if (sobelKernelSize > 1)
                {
                    sobelKernelSize--;
                }
            };

            if ((char)c == 'f')
            {
                scalerFactor++;
            };

            if ((char)c == 'q')
            {
                break;
            };

            cv::imshow("inputImage", image);
            cv::imshow("outputImage", outputImage);
            cv::imshow("outputImageLaplacian", outputImageLaplacian);
        }
    }


    int main()
    {
        // cv::Mat_<float> matrix(3,3);

        cv::Mat matrix = cv::imread("img.png");
        cv::Mat_<float> kernel(3,3);
        cv::Mat matrix2,kernel2,filtered2D,gaussian;

        // matrix << 1,2,5,6,8,1,0,1,2;
        kernel << 0.0625,0.125,0.0625,0.125,0.25,0.125,0.0625,0.125,0.0625;

        // create window of size 640 * 480
        cv::resize(matrix, matrix, cv::Size(480, 480));

        cv::imshow("Matrix", matrix);
        matrix.convertTo(matrix2, CV_8UC1);
        cv::imshow("Converted Matrix", matrix);

        cv::imshow("Kernel", kernel);
        kernel.convertTo(kernel2, CV_8UC1);
        cv::resize(kernel, kernel, cv::Size(480, 480));
        cv::imshow("Converted Kernel", kernel);

        cv::filter2D(matrix2, filtered2D, matrix.type(), kernel2, cv::Point(-1,-1), 0, cv::BORDER_DEFAULT);
        cv::imshow("filtered2D", filtered2D);

        cv::GaussianBlur(matrix, gaussian, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT);
        cv::imshow("Gaussian", gaussian);

        cv::erode(matrix, gaussian, cv::Mat(), cv::Point(-1,-1), 1, cv::BORDER_DEFAULT);
        cv::imshow("Eroded", gaussian);




        cv::waitKey(0);
        return EXIT_SUCCESS;
    }


     */