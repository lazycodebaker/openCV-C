#include "iostream"
#include "vector"

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"

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

/*
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