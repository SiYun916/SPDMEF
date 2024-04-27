#ifndef MULTIIMAGESFUISON_H
#define MULTIIMAGESFUISON_H

#include <math.h>
#include <dirent.h>

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <iterator>
#include <map>

#include <QDir>
#include <QString>
#include <QStringList>

#include "opencv2/core.hpp"
#include "opencv2/photo.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"

using namespace std;
using namespace cv;

//
bool SortMatCompare(const Mat &mat1, const Mat &mat2);


class MultiImagesFusion {
public:
    explicit MultiImagesFusion(const string &aligned_images_folder_path);
    ~MultiImagesFusion();
    void DoLinearAddAndMean();
    void DoGaussMEF();
    void DoMertensMEF();
    void DoSPDMEF();
    //下面是测试用的
    void TestFunction();



private:
    string images_folder_path_ = "";
    vector<Mat> images_mat_ = {};
    vector<Mat> images_gray_mat_ = {};
    vector<Mat> images_gray_float_mat_ = {};
    //AdjustImageSize参数，长宽最大值512
    int image_max_worh_size_ = 512;
    //SPDMEF用到的参数，之后再修改接口
    int patch_size_ = 21;
    int step_size_ = 2;
    float Ts_ = 0.8;
    float Tm_ = 0.1;
    float global_gaussian_ = 0.2;
    float local_gaussian_ = 0.5;
    float p_ = 4;
    float exp_thres_ = 0.01; //Specify the exposure threshold to determine under- and over-exposed patches.

    void GetImagesMat();
    void GetImagesGrayMat();
    void AdjustImageSize();
    Mat MatchHistograms(const Mat &source_image, const Mat &reference_image);
    vector<Mat> IMFConsistency(const vector<Mat> local_mean_intensity, int reference_image_index, int size_images_num);
    Mat SPDMEF();

    //下面是测试用的
    void SaveImage2Txt(const Mat &tosave_image);
    //这里需要对hist修改，所以不用const
    void DrawHist(Mat &hist, int type);

};



#endif // MULTIIMAGESFUISON_H
