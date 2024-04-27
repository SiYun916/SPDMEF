#include <iostream>
#include <vector>
#include <fstream>

#include "opencv2/photo.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include "MultiImagesFusion.h"

using namespace std;
using namespace cv;

int main(){
    MultiImagesFusion do_images_fusion = MultiImagesFusion("E:\\Project\\QtPro\\cal_exposure\\data\\Temp0\\test\\");
    do_images_fusion.DoGaussMEF();
    //do_images_fusion.DoMertensMEF();
    //do_images_fusion.DoLinearAddAndMean();
    //do_images_fusion.TestFunction();
    //do_images_fusion.DoSPDMEF();
    return 0;
}
