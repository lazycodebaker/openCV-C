#include "iostream"
#include "vector"

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"

int main()
{
    cv::Mat img = cv::imread("image.png");

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
}

/*

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