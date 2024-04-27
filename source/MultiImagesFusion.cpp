#include "MultiImagesFusion.h"

//Function SortMatCompare 用于sort排序，以总体灰度值大小代表曝光度
//从小到大进行排序
bool SortMatCompare(const Mat &mat1, const Mat &mat2) {
    return sum(mat1)[0] < sum(mat2)[0];
}

MultiImagesFusion::MultiImagesFusion(const string &aligned_images_folder_path) {
    images_folder_path_ = aligned_images_folder_path;
}

MultiImagesFusion::~MultiImagesFusion() {

}

//TODO:需要加上try catch 这里不能直接从文件中读取，之后需要修改成
//由CImage读取raw文件，然后转换成Mat
//Function GetImagesMat 根据构造类时所填的文件夹，获取其下面
//所有图片的Mat并保存在一个类的私有vector images_mat_中
void MultiImagesFusion::GetImagesMat() {
//    下面注释掉的代码是用dirent库实现的读取指定文件夹下的所有文件
//    读出来的东西包括文件夹  暂时不用   先用QT
//    const char *path = images_folder_path_.data();
//    vector<string> images_path = {};
//    DIR *dir = opendir(path);
//    struct dirent *diread;

//    if (dir != nullptr) {
//        while ((diread = readdir(dir)) != nullptr) {
//            string image_string_name = diread->d_name;
//            string full_path = images_folder_path_ + image_string_name;
//            cout<<full_path<<endl;
//            images_path.push_back(full_path);
//        }
//        closedir(dir);
//    } else {
//        perror("opendir");
//    }
    QString path = QString::fromStdString(images_folder_path_);
    QDir directory(path);
    QStringList filter;
    filter << "*.*";
    QStringList images_path = {};

    QFileInfoList images_info_list = directory.entryInfoList(filter,QDir::Files);
    foreach (const QFileInfo &image_info, images_info_list) {
        QString image_name = image_info.fileName();
        QString full_path = path + image_name;
        images_path.append(full_path);
    }

    foreach (const QString &image_full_path,images_path) {
        Mat image_mat_tmp = imread(image_full_path.toStdString());
        images_mat_.push_back(image_mat_tmp);
    }
    return ;
}

//TODO:后面需要重新改
//Function GetImagesGrayMat 将图像类型转换成8U1C，结果存储在
//images_gray_mat_类变量中
void MultiImagesFusion::GetImagesGrayMat() {
    for (int i = 0; i < images_mat_.size(); i++) {
        Mat tmp_gray_image;
        cvtColor(images_mat_[i], tmp_gray_image, CV_BGR2GRAY);
        images_gray_mat_.push_back(tmp_gray_image);
    }
    return ;
}

//Function DoMertensMEF 先调用GetImagesMat函数读取所有图像Mat
//保存至类私有变量images_mat_，调用opencv库中mertens的Exposure fusion
//方法进行融合，直接将结果保存至当前文件夹下
void MultiImagesFusion::DoMertensMEF() {
    GetImagesMat();
    auto merge_mertens = createMergeMertens();
    Mat fusioned_result;
    merge_mertens->process(images_mat_, fusioned_result);
    //Mat数据类型查看img.type()
    fusioned_result = fusioned_result * 255;
    fusioned_result.convertTo(fusioned_result, CV_8UC3);
    imwrite(".\\MertensMEF.png", fusioned_result);
    return ;
}

//TODO:如果图片width height不一致则报错。可能需要
//Function DoLinearAddAndMean 针对选出来的图片，转换成灰度图
//对应位置像素点相加然后取平均，直接将结果保存至当前文件夹下
//Attention 8UC3不能直接转到32FC1
void MultiImagesFusion::DoLinearAddAndMean() {
    GetImagesMat();
    vector<Mat> gray_images_mats = { };
    for (int i = 0; i < images_mat_.size(); i++) {
        Mat tmp_gray_mat;
        cvtColor(images_mat_[i], tmp_gray_mat, CV_BGR2GRAY);
        gray_images_mats.push_back(tmp_gray_mat);
    }
    Mat fusioned_result = Mat::zeros(images_mat_[0].rows,images_mat_[0].cols, CV_32FC1);
    for (int i = 0; i < gray_images_mats.size(); i++) {
        Mat float_mat;
        gray_images_mats[i].convertTo(float_mat, CV_32FC1, 1.0/255);
        fusioned_result += float_mat;
    }
    fusioned_result /= gray_images_mats.size();
    fusioned_result.convertTo(fusioned_result, CV_8UC1, 255);
    imwrite(".\\LinearMEF.png", fusioned_result);
    return ;
}

//Function DoGaussMEF 高斯融合
void MultiImagesFusion::DoGaussMEF() {
    GetImagesMat();
    const float micro = 0.5;
    const float sigma = 0.5;
    //存放像素值为0-1的图像
    vector<Mat> gray_float_images_mats = { };
    for (int i = 0; i < images_mat_.size(); i++) {
        Mat tmp_gray_mat;
        cvtColor(images_mat_[i], tmp_gray_mat, CV_BGR2GRAY);
        Mat float_gray_mat;
        tmp_gray_mat.convertTo(float_gray_mat, CV_32FC1, 1.0/255);
        gray_float_images_mats.push_back(float_gray_mat);
    }
    //系数矩阵
    const int image_rows = gray_float_images_mats[0].rows;
    const int image_cols = gray_float_images_mats[0].cols;
    vector<Mat> weight_map = {};
    Mat weight_sum = Mat::zeros(image_rows, image_cols, CV_32FC1);
    for (int i = 0; i < gray_float_images_mats.size(); i++) {
        Mat tmp_weight = Mat::zeros(image_rows, image_cols, CV_32FC1);
        tmp_weight = gray_float_images_mats[i].clone();
        tmp_weight = tmp_weight - 0.5;
        pow(tmp_weight, 2, tmp_weight);
        tmp_weight = -tmp_weight;
        tmp_weight /= 2*sigma*sigma;
        exp(tmp_weight, tmp_weight);
        weight_map.push_back(tmp_weight);
        weight_sum += tmp_weight;
    }
    Mat result = Mat::zeros(image_rows, image_cols, CV_32FC1);
    for (int i = 0; i < gray_float_images_mats.size(); i++) {
        Mat norm_weight;
        divide(weight_map[i], weight_sum, norm_weight);
        Mat part_image;
        multiply(norm_weight, gray_float_images_mats[i], part_image);
        result += part_image;
    }
    Mat fusioned_image;
    result.convertTo(fusioned_image, CV_8UC1, 255);
    imwrite(".\\GaussMEF.png", fusioned_image);
    return ;
}

//TODO:实际项目中，按要求修改图片大小具体调整，或加上try catch
//Function AdjustImageSize 调整图片的大小，保证长宽最大值不大于定义的
//image_max_worh_size，这里假定所有图像长宽都是一样的
void MultiImagesFusion::AdjustImageSize() {
    int image_height = images_gray_float_mat_[0].rows;
    int image_width = images_gray_float_mat_[0].cols;
    double scale = 0.0;
    //size第一个参数是宽第二个是高
    Size adapted_size = Size(0.0,0.0);
    if (image_height >= image_width && image_height > image_max_worh_size_) {
        scale = image_height / image_max_worh_size_;
        adapted_size = Size(image_width/scale, image_max_worh_size_);
    } else if (image_height < image_width && image_width > image_max_worh_size_) {
        scale = image_width / image_max_worh_size_;
        adapted_size = Size(image_max_worh_size_, image_height/scale);
    } else {
        //不用调整的话直接返回
        return ;
    }
    //逐个调整
    for (int i = 0; i < images_gray_float_mat_.size(); i++) {
        resize(images_gray_float_mat_[i], images_gray_float_mat_[i], adapted_size);
    }
    return ;
}


//已完成功能，实际应用性能待测试
//TODO:测试性能
//Function MatchHistorgam 函数功能:直方图匹配，将source_image的直方图
//调整成类似于reference_image的直方图，最终返回source_image匹配好后的图
//像。
//Input:均为8U1C的灰度图
//Output:8U1C的灰度图
/*
void calcHist( const Mat* images, int nimages,
               const int* channels, InputArray mask,
               OutputArray hist, int dims, const int* histSize,
               const float** ranges, bool uniform=true, bool accumulate=false );
1.输入的图像数组   2.输入数组的个数        3.通道数              4.掩码                5.直方图
6.直方图维度       7.直方图每个维度的尺寸数组   8.每一维数组的范围    9.直方图是否是均匀   10.累加标志
*/
Mat MultiImagesFusion::MatchHistograms(const Mat &source_image, const Mat &reference_image) {
    //calcHist parameters define
    int gray_img_num = 1;   //图像数
    int gray_channels = 0;  //图像通道数
    const int gray_hist_dim = 1; //直方图维数
    const int gray_hist_size = 256; //直方图每一维bin个数
    float gray_ranges[2] = {0, 256}; //灰度值统计范围
    const float *gray_hist_ranges[1] = {gray_ranges}; //灰度值统计范围指针
    //计算直方图的结果保存在下面两个矩阵中
    Mat source_image_hist;
    Mat reference_image_hist;

    calcHist(&source_image, gray_img_num, &gray_channels, Mat(), source_image_hist,
             gray_hist_dim, &gray_hist_size, gray_hist_ranges);
    calcHist(&reference_image, gray_img_num, &gray_channels, Mat(), reference_image_hist,
             gray_hist_dim, &gray_hist_size, gray_hist_ranges);

    //如果有必要，均衡化处理
    //equalizeHist(src, dst);
    //equalizeHist(src, dst);

    //计算累积概率
    //cdf cumulative distribution function
    //这里假定图像大小都一致
    int pixel_totalnum = source_image.rows * source_image.cols;
    float source_image_hist_cdf[256] = {source_image_hist.at<float>(0)};
    float reference_image_hist_cdf[256] = {reference_image_hist.at<float>(0)};
    for (int i = 1; i < 256; i++) {
        source_image_hist_cdf[i] = source_image_hist_cdf[i-1] + source_image_hist.at<float>(i);
        reference_image_hist_cdf[i] = reference_image_hist_cdf[i-1] + reference_image_hist.at<float>(i);
    }
    for (int i = 0; i < 256; i++) {
        source_image_hist_cdf[i] /= pixel_totalnum;
        reference_image_hist_cdf[i] /= pixel_totalnum;
    }

    //累积概率误差矩阵
    float diff_cdf[256][256];
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            diff_cdf[i][j] = fabs(source_image_hist_cdf[i] - reference_image_hist_cdf[j]);
        }
    }

    //LUT映射表
    Mat lut(1, 256, CV_8U);
    for (int i = 0; i < 256; i++) {
        float min = diff_cdf[i][0];
        int index = 0;
        for (int j = 0; j < 256; j++) {
            if (min > diff_cdf[i][j]) {
                min = diff_cdf[i][j];
                index = j;
            }
        }
        lut.at<uchar>(i) = index;
    }

    //结果
    Mat match_result;
    LUT(source_image, lut, match_result);
    return match_result;
}

//TODO:
//Function IMFConsistency 这里是根据那个Tm来计算一个map
vector<Mat> MultiImagesFusion::IMFConsistency(const vector<Mat> local_mean_intensity,
                                              int reference_image_index,
                                              int size_images_num) {
    //初始化imf_map 参考帧全1 其余全0
    vector<Mat> imf_map = {};
    for (int i = 0; i < size_images_num; i++) {
        Mat tmp_zeros = Mat::zeros(local_mean_intensity[0].rows, local_mean_intensity[0].cols, CV_32F);
        imf_map.push_back(tmp_zeros);
    }
    imf_map[reference_image_index] = Mat::ones(imf_map[0].rows, imf_map[0].cols, CV_32F);
    //参考帧的local mean intensity
    //为了匹配直方图，先转换成8UC1
    Mat ref_mean_intensity_32fc1 = local_mean_intensity[reference_image_index];
    Mat ref_mean_intensity_8uc1;
    ref_mean_intensity_32fc1.convertTo(ref_mean_intensity_8uc1, CV_8U, 255);
    for (int i = 0; i < 1; i++) {
        if (i != reference_image_index) {
            //计算某一帧和参考帧  直方图匹配
            //由于直方图匹配输入输出都是8UC1类型的，所以先转化
            Mat tmp_mean_intensity_32fc1 = local_mean_intensity[i];
            Mat tmp_mean_intensity_8uc1;
            tmp_mean_intensity_32fc1.convertTo(tmp_mean_intensity_8uc1, CV_8U, 255);
            //下面进行匹配 得到float类型的结果
            Mat tmp_match_res_8uc1 = MatchHistograms(tmp_mean_intensity_8uc1, ref_mean_intensity_8uc1);
            Mat tmp_match_res_32fc1;
            tmp_match_res_8uc1.convertTo(tmp_match_res_32fc1, CV_32F, 1.0/255);
            //计算match后图像和参考图像的区别
            Mat diff;
            absdiff(tmp_match_res_32fc1, ref_mean_intensity_32fc1, diff);
            Mat to_compare_mat = Mat::ones(diff.rows, diff.cols, CV_32F);
            to_compare_mat = to_compare_mat * Tm_;
            //到这正确生成 比较矩阵
            //下面比较大小Tm小于Tm的为1大于等于的为0
            Mat tmp_res;
            compare(diff, to_compare_mat, tmp_res, CMP_LT);
            tmp_res.convertTo(tmp_res, CV_32F, 1.0/255);
            imf_map[i] = tmp_res;
        }
    }
    return imf_map;
}



//TODO:实现SPDMEF论文算法，修改接口文件的输入和输出，try catch
//Function SPDMEF 算法的具体实现
Mat MultiImagesFusion::SPDMEF() {
    //取patch 窗口 每个点都是均值，方便之后计算
    //测试过，window输出无误，每个点均为1/441
    Mat window = Mat::ones(patch_size_, patch_size_, CV_32FC1);
    window /= window.total();
    //行=高=.shape[0] 列=宽=.shape[1]
    int size_height = images_gray_float_mat_[0].rows;
    int size_width = images_gray_float_mat_[0].cols;
    int size_images_num = images_gray_float_mat_.size();
    //超参数，不太懂干什么，
    float param_c = pow(0.03, 2) / 2;

    //以patch_size大小对图片进行卷积后，结果数据的高宽
    //row_max_size=x_idx_max
    int row_max_size = size_height - patch_size_ + 1;
    int col_max_size = size_width - patch_size_ + 1;

    //参考帧选择，以一张图片所有像素点之和为曝光度，选择曝光度最中间的一张
    //为基准，因为已经排好序了取images_gray_float_mat_最中间的一张即可
    int reference_image_index = size_images_num / 2;

    //除参考帧外，其余每张曝光的图片都和参考帧进行直方图匹配
    //先把参考帧转化成8uc1的
    Mat tmp_8uc1_reference;
    images_gray_float_mat_[reference_image_index].convertTo(tmp_8uc1_reference, CV_8UC1, 255);
    for (int i = 0; i < size_images_num; i++) {
        if (i != reference_image_index) {
            //先转换成8UC1的，函数只支持8UC1的转化
            Mat tmp_8uc1_input;
            images_gray_float_mat_[i].convertTo(tmp_8uc1_input, CV_8UC1, 255);
            Mat tmp_match_res = MatchHistograms(tmp_8uc1_input, tmp_8uc1_reference);
            //到这里正常匹配
            Mat tmp_32fc1_res;
            tmp_match_res.convertTo(tmp_32fc1_res, CV_32FC1, 1.0/255);
            images_gray_float_mat_.push_back(tmp_32fc1_res);
        }
    }
    //上面这个循环结束后，images_gray_float_mat_中图片的数量变成了2*size_images_num-1
    int size_newimages_num = images_gray_float_mat_.size();
    //Global Mean Intensity
    //论文中所说的x=cs+l中，l参数中的一项
    vector<Mat> global_mean_intensity = {};
    for (int i = 0; i < size_newimages_num; i++) {
        Mat tmp_global_mean = Mat::ones(row_max_size, col_max_size, CV_32FC1);
        Scalar tmp_val = mean(images_gray_float_mat_[i]);
        float mean_value = tmp_val.val[0];
        tmp_global_mean *= mean_value;
        global_mean_intensity.push_back(tmp_global_mean);
    }
    //以上完成全局intensity计算，下面计算Local Mean Intensity
    vector<Mat> local_mean_intensity = {};
    vector<Mat> local_mean_square = {};
    //确定左上角顶点xy值和长宽用于裁取，因为filter2D必须填充
    //这里要注意，可能边界出错误
    int top_left_x = (patch_size_ - 1) / 2;
    int top_left_y = (patch_size_ - 1) / 2;
    int cut_width = size_width - patch_size_ + 1;
    int cut_height = size_height - patch_size_ + 1;
    for (int i = 0; i < size_newimages_num; i++) {
        Mat tmp_local_mean;
        Mat tmp_local_mean_cut;
        Mat tmp_local_square;
        //计算卷积，然后把需要的部分裁剪出来
        filter2D(images_gray_float_mat_[i], tmp_local_mean, CV_32F, window);
        Rect cut_rect(top_left_x, top_left_y, cut_width, cut_height);
        tmp_local_mean_cut = tmp_local_mean(cut_rect);
        local_mean_intensity.push_back(tmp_local_mean_cut);
        //平方
        pow(tmp_local_mean_cut, 2, tmp_local_square);
        local_mean_square.push_back(tmp_local_square);
    }
    //计算信号强度Signal Strength
    vector<Mat> signal_strength = {};
    for (int i = 0; i < size_newimages_num; i++) {
        //原矩阵，每个值平方
        Mat tmp_source_mat_pow;
        pow(images_gray_float_mat_[i], 2, tmp_source_mat_pow);
        //对pow后数据进行卷积，然后裁出需要的部分
        Mat tmp_filter_res;
        Mat tmp_filter_res_cut;  //492*492
        filter2D(tmp_source_mat_pow, tmp_filter_res, CV_32F, window);
        Rect cut_rect(top_left_x, top_left_y, cut_width, cut_height);
        tmp_filter_res_cut = tmp_filter_res(cut_rect);
        //相减之后的结果，以及相减之后绝对值的结果
        //两者相加除以2，就能保证小于0的数全部重新赋值为0
        Mat cut_sub_lms = tmp_filter_res_cut - local_mean_square[i];
        Mat cut_sub_lms_abs;
        absdiff(tmp_filter_res_cut, local_mean_square[i], cut_sub_lms_abs);
        Mat cut_sub_lms_res;
        add(cut_sub_lms, cut_sub_lms_abs, cut_sub_lms_res);
        //用convertTo来做矩阵除法
        cut_sub_lms_res.convertTo(cut_sub_lms_res, CV_32F, 1.0/2);
        Mat tmp_signal_strength;
        sqrt(cut_sub_lms_res, tmp_signal_strength);
        //本来tmp_signal_strength乘的应该是sqrt(patch_size ** 2 * size_3)
        //因为灰度图通道为1，所以就只剩下patch_size了
        tmp_signal_strength = tmp_signal_strength * patch_size_ + 0.001;
        signal_strength.push_back(tmp_signal_strength);
    }
    //****************************************到上面整体测试结果正常
    //结构一致性计算，这里是来保证运动不会造成影响
    //用map来存数据，这里先初始化一个全0的map
    map<int, vector<Mat>> structural_consist_map;
    for (int i = 0; i < size_images_num; i++) {
        vector<Mat> mats;
        for (int j = 0; j < size_images_num; j++) {
            Mat tmp_mat = Mat::zeros(row_max_size, col_max_size, CV_32FC1);
            mats.push_back(tmp_mat);
        }
        structural_consist_map.insert(pair<int, vector<Mat>>(i, mats));
    }
    //结构一致性计算
    for (int i = 0; i < size_images_num; i++) {
        for (int j = i+1; j < size_images_num; j++) {
            Mat tmp_cross_intens;
            multiply(local_mean_intensity[i], local_mean_intensity[j], tmp_cross_intens);
            //tmp_imgimulimgj是原尺寸的图像512*512
            Mat tmp_imgimulimgj;
            multiply(images_gray_float_mat_[i], images_gray_float_mat_[j], tmp_imgimulimgj);
            //先卷积，然后再裁剪
            Mat tmp_filter_res;
            Mat tmp_filter_res_cut;
            filter2D(tmp_imgimulimgj, tmp_filter_res, CV_32F, window);
            Rect cut_rect(top_left_x, top_left_y, cut_width, cut_height);
            tmp_filter_res_cut = tmp_filter_res(cut_rect);
            //至此得到tmp_cross_intens和tmp_filter_res_cut，两者相减得到交叉结构
            //这里其实在计算ρk，至于为什么这样算，没搞明白
            Mat cross_strg = tmp_filter_res_cut - tmp_cross_intens;
            Mat cross_strg_add_c = cross_strg + param_c;
            Mat cross_sig_strength;
            multiply(signal_strength[i], signal_strength[j], cross_sig_strength);
            Mat cross_sig_strength_add_c = cross_sig_strength + param_c;
            //计算map
            Mat tmp_map;
            divide(cross_strg_add_c, cross_sig_strength_add_c, tmp_map);
            //保证权重为正，让tmp_map中的负数元素为0
            Mat tmp_map_abs = abs(tmp_map);
            Mat tmp_map_res;
            add(tmp_map, tmp_map_abs, tmp_map_res);
            tmp_map_res.convertTo(tmp_map_res, CV_32F, 1.0/2);
            //存储到map中
            structural_consist_map[i][j] = tmp_map_res;
        }
    }
    //参考帧
    vector<Mat> structural_reference_map = {};
    for (int i = 0; i < size_images_num; i++) {
        structural_reference_map.push_back(structural_consist_map[reference_image_index][i] +
                                           structural_consist_map[i][reference_image_index]);
    }
    Mat tmp_ref_ones = Mat::ones(row_max_size, col_max_size, CV_32F);
    structural_reference_map[reference_image_index] = tmp_ref_ones;
    //如果有出错，上面这里要检查下

    //structural_new_ref_map是在structural_reference_map的基础上和Ts参数进行比较得来的
    //其中元素大于等于0.8的改成1，小于的改成0
    vector<Mat> structural_new_ref_map = {};
    for (int i = 0; i < size_images_num; i++) {
        //这里structural_reference_map[i]是正常的，有大于0.8的也有小于的
        //转换，float类型比较的结果有问题，先转化成uchar类型的，然后把参数值Ts*255
        //比较完后再转换回来
        Mat to_compare_mat = Mat::ones(row_max_size, col_max_size, CV_32F);
        to_compare_mat = to_compare_mat * Ts_;
        Mat tmp_compare_res;
        compare(structural_reference_map[i], to_compare_mat, tmp_compare_res, CMP_GE);
        tmp_compare_res.convertTo(tmp_compare_res, CV_32F, 1.0/255);
        structural_new_ref_map.push_back(tmp_compare_res);
    }
    //下面这里可能也要检查下
    //这里的intens_idx_map还不太清楚其作用，应该是根据曝光度再对structural_new_ref_map进行筛选
    //理论上每个mat的值都是一样的
    Mat intens_idx_map;
    Mat exp_thres_compare_mat_1 = Mat::ones(row_max_size, col_max_size, CV_32F);
    Mat exp_thres_compare_mat_2 = Mat::ones(row_max_size, col_max_size, CV_32F);
    Mat exp_thres_compare_res_1;
    Mat exp_thres_compare_res_2;

    exp_thres_compare_mat_1 = exp_thres_compare_mat_1 * exp_thres_;
    exp_thres_compare_mat_2 = exp_thres_compare_mat_2 * (1 - exp_thres_);
    compare(local_mean_intensity[reference_image_index], exp_thres_compare_mat_1, exp_thres_compare_res_1, CMP_LT);
    compare(local_mean_intensity[reference_image_index], exp_thres_compare_mat_2, exp_thres_compare_res_2, CMP_GT);
    //这里等价于|运算 即取两个之间的最大值
    intens_idx_map = max(exp_thres_compare_res_1, exp_thres_compare_res_2);
    //归一
    intens_idx_map.convertTo(intens_idx_map, CV_32F, 1.0/255);
    //intens_idx_map计算应该是正确的
    //structural_new_ref_map的每个分量和intens_idx_map相与，也就是取max
    //更新完structural_new_ref_map
    for (int i = 0; i < size_images_num; i++) {
        Mat tmp_or_mat;
        tmp_or_mat = max(structural_new_ref_map[i], intens_idx_map);
        structural_new_ref_map[i] = tmp_or_mat;
    }

    //这里创建一个mask用于开运算，去除图像中小的亮点。已验证创建正确
    //****这里可以看着改改能不能提高
    Mat structural_elem = Mat::zeros(41, 41, CV_8UC1);
    int tmp_n = 11;
    for (int i = 0; i < structural_elem.rows; i++) {
        if (abs(i - int(structural_elem.cols/2)) > 10) {
            tmp_n = tmp_n - 1;
        } else {
            tmp_n = 0;
        }
        for (int j = abs(tmp_n); j < structural_elem.cols - abs(tmp_n); j++) {
            structural_elem.at<uchar>(i, j) = 1;
        }
    }
    //对于structural_new_ref_map再一次进行修改，使用opencv自带的开运算
    for (int i = 0; i < size_images_num; i++) {
        //morphologyEx最后一个mask的参数只能是uchar类型的
        morphologyEx(structural_new_ref_map[i], structural_new_ref_map[i], CV_MOP_OPEN, structural_elem);
    }

    //IMFmap 什么算法
    vector<Mat> imf_reference_map = IMFConsistency(local_mean_intensity, reference_image_index, size_images_num);
    //下面计算最终的map矩阵  即需要哪个点
    //[0,size_images_num)是正常的
    vector<Mat> exp_ref_map = {};
    for (int i = 0; i < size_images_num; i++) {
        Mat tmp_mul_res;
        multiply(structural_new_ref_map[i], imf_reference_map[i], tmp_mul_res);
        exp_ref_map.push_back(tmp_mul_res);
    }
    for (int i = 0; i < size_images_num; i++) {
        if (i != reference_image_index) {
            Mat tmp_map;
            tmp_map = 1 - exp_ref_map[i];
            exp_ref_map.push_back(tmp_map);
        }
    }
    //这里的local_mean_intensity和global_mean_intensity和刚算出来的一样
    //全部的size_newsize
    vector<Mat> mean_intensity_map = {};
    for (int i = 0; i < size_newimages_num; i++) {
        //全局分量
        Mat tmp_global_subpow;
        pow(global_mean_intensity[i]-0.5, 2, tmp_global_subpow);
        float global_gauss_pow = pow(global_gaussian_, 2);
        Mat tmp_global_subpowdiv = tmp_global_subpow / global_gauss_pow;
        //局部分量
        Mat tmp_local_subpow;
        pow(local_mean_intensity[i]-0.5, 2, tmp_local_subpow);
        float local_gauss_pow = pow(local_gaussian_, 2);
        Mat tmp_local_subpowdiv = tmp_local_subpow / local_gauss_pow;
        //两者之和
        Mat tmp_global_add_local = tmp_global_subpowdiv + tmp_local_subpowdiv;
        Mat tmp_total = -0.5 * tmp_global_add_local;
        //结果
        Mat tmp_res;
        exp(tmp_total, tmp_res);
        mean_intensity_map.push_back(tmp_res);
    }
    for (int i = 0; i < size_newimages_num; i++) {
        Mat tmp_mul_res;
        multiply(mean_intensity_map[i], exp_ref_map[i], tmp_mul_res);
        mean_intensity_map[i] = tmp_mul_res;
    }
    //到这里正确计算出mean_intens_map = mean_intens_map * exp_ref_map
    //一定要注意，一定要新建一个zeros而不是直接赋值
    Mat mean_norm = Mat::zeros(mean_intensity_map[0].rows, mean_intensity_map[0].cols, CV_32FC1);
    for (int i = 0; i < size_newimages_num; i++) {
        mean_norm = mean_norm + mean_intensity_map[i];
    }
    //到这里正确计算出normalizer = np.sum(mean_intens_map, axis=2) 求和
    for (int i = 0; i < size_newimages_num; i++) {
        Mat tmp_div_res;
        divide(mean_intensity_map[i], mean_norm, tmp_div_res, 1.0);
        mean_intensity_map[i] = tmp_div_res;
    }
    //到这里应该正确计算出mean_intens_map = mean_intens_map / np.repeat(np.expand_dims(normalizer, axis=2), exp_img_num, axis=2)
    //下面计算Signal Structure Weighting Map
    //signal_strength[0]前后没变化
    vector<Mat> stru_consist_map = {};
    for (int i = 0; i < size_newimages_num; i++) {
        Mat tmp_sig_pow;
        pow(signal_strength[i], p_, tmp_sig_pow);
        stru_consist_map.push_back(tmp_sig_pow);
    }
    //下面改写stru_consist_map = stru_consist_map * exp_ref_map + 0.001
    for (int i = 0; i < size_newimages_num; i++) {
        Mat tmp_mul_res;
        multiply(stru_consist_map[i], exp_ref_map[i], tmp_mul_res);
        tmp_mul_res = tmp_mul_res + 0.001;
        stru_consist_map[i] = tmp_mul_res;
    }
    //进行归一化处理，应该是看各部分占比
    Mat stru_norm = Mat::zeros(stru_consist_map[0].rows, stru_consist_map[0].cols, CV_32FC1);
    for (int i = 0; i < size_newimages_num; i++) {
        stru_norm = stru_norm + stru_consist_map[i];
    }
    for (int i = 0; i < size_newimages_num; i++) {
        Mat tmp_div_res;
        divide(stru_consist_map[i], stru_norm, tmp_div_res, 1.0);
        stru_consist_map[i] = tmp_div_res;
    }
    //应该正确，下面是信号强度找最强
    vector<Mat> max_exp = {};
    for (int i = 0; i < size_newimages_num; i++) {
        Mat tmp_mul_res;
        multiply(signal_strength[i], exp_ref_map[i], tmp_mul_res);
        max_exp.push_back(tmp_mul_res);
    }
    //选择最强的信号
    Mat max_exp_signal = Mat::zeros(max_exp[0].rows, max_exp[0].cols, CV_32FC1);
    for (int i = 0; i < max_exp[0].rows; i++) {
        for (int j = 0; j < max_exp[0].cols; j++) {
            float tmp_max = 0;
            for (int k = 0; k < max_exp.size(); k++) {
                tmp_max = max(tmp_max, max_exp[k].at<float>(i, j));
            }
            max_exp_signal.at<float>(i,j) = tmp_max;
        }
    }
    //理论上选择了最强的信号，但感觉数据有点怪
    //下面是计算最终结果时，选择哪三张图片的矩阵
    vector<Mat> idx_matrix = {};
    for (int i = 0; i < size_images_num; i++) {
        if (i < reference_image_index) {
            Mat tmp_mul_res1 = exp_ref_map[i] * i;
            Mat tmp_mul_res2 = exp_ref_map[i+size_images_num] * (i + size_images_num);
            Mat tmp_res = tmp_mul_res1 + tmp_mul_res2;
            idx_matrix.push_back(tmp_res);
        } else if (i == reference_image_index) {
            Mat tmp_res = Mat::ones(row_max_size, col_max_size, CV_32FC1);
            idx_matrix.push_back(tmp_res);
        } else if (i > reference_image_index) {
            Mat tmp_mul_res1 = exp_ref_map[i] * i;
            Mat tmp_mul_res2 = exp_ref_map[i+size_images_num-1] * (i + size_images_num - 1);
            Mat tmp_res = tmp_mul_res1 + tmp_mul_res2;
            idx_matrix.push_back(tmp_res);
        }
    }
    //循环列表 x_idx对应于原代码的话是shape[0]也就是高
    vector<int> x_idx, y_idx;
    for (int i = 0; i < row_max_size; i += step_size_) {
        x_idx.push_back(i);
    }
    x_idx.push_back(row_max_size - 1);
    for (int i = 0; i < col_max_size; i += step_size_) {
        y_idx.push_back(i);
    }
    y_idx.push_back(col_max_size - 1);
    //最终结果
    Mat final_img = Mat::zeros(size_height, size_width, CV_32FC1);
    Mat count_map = Mat::zeros(size_height, size_width, CV_32FC1);
    Mat count_window = Mat::ones(patch_size_, patch_size_, CV_32FC1);
    //偏移量
    int offset = patch_size_;
    //主循环
    for (int row = 0; row < x_idx.size(); row++) {
        for (int col = 0; col < y_idx.size(); col++) {
            int i = x_idx[row];
            int j = y_idx[col];
            //根据i j来截取patch
            vector<Mat> tmp_blocks = {};
            //这里idx_matrix.size就是size_images_num，就是选取图片的数量
            for (int k = 0; k < idx_matrix.size(); k++) {
                Mat tmp_one_block;
                int tmp_image_index = idx_matrix[k].at<float>(i,j);
                images_gray_float_mat_[tmp_image_index].rowRange(i, i+offset).colRange(j, j+offset).copyTo(tmp_one_block);
                tmp_blocks.push_back(tmp_one_block);
            }
            //计算block的结果
            Mat r_block = Mat::zeros(patch_size_, patch_size_, CV_32FC1);
            //各个block块乘以权重，然后相加。tmp_blocks.size()=size_images_num
            for (int k = 0; k < tmp_blocks.size(); k++) {
                float num_stru_consist_map = stru_consist_map[k].at<float>(i, j);
                float num_lcl_mean_intens = local_mean_intensity[k].at<float>(i, j);
                float num_sig_strength = signal_strength[k].at<float>(i ,j);
                Mat tmp_res = num_stru_consist_map * (tmp_blocks[k] - num_lcl_mean_intens) / num_sig_strength;
                r_block = r_block + tmp_res;
            }
            float norm_val = norm(r_block, NORM_L2);
            if (norm_val > 0) {
                r_block = r_block / norm_val * max_exp_signal.at<float>(i, j);
            }
            //SaveImage2Txt(r_block);
            float intens_total = 0.0;
            for (int k = 0; k < size_newimages_num; k++) {
                float tmp_intens = mean_intensity_map[k].at<float>(i, j) * local_mean_intensity[k].at<float>(i, j);
                intens_total = intens_total + tmp_intens;
            }
            r_block = r_block + intens_total;
            final_img.rowRange(i, i+offset).colRange(j, j+offset) += r_block;
            count_map.rowRange(i, i+offset).colRange(j, j+offset) += count_window;
            //记得删除下面这一行
            cout<<i<<endl;
        }
    }
    Mat img_res;
    divide(final_img, count_map, img_res, 1.0);
    for (int i = 0; i < img_res.rows; i++) {
        for (int j = 0; j < img_res.cols; j++) {
            if (img_res.at<float>(i, j) > 1) {
                img_res.at<float>(i, j) = 1;
            } else if (img_res.at<float>(i, j) < 0) {
                img_res.at<float>(i, j) = 0;
            }
        }
    }

    Mat return_img_res;
    img_res.convertTo(return_img_res, CV_8UC1, 255);
    return return_img_res;
}



//Function DoSPDMEF 首先对图像处理，选取参考帧。根据SPDMEF的算法对图片进行融合，
//结果暂存至当前文件夹下。这里也是假设的是都是同等大小的一系列图片。
void MultiImagesFusion::DoSPDMEF() {
    GetImagesMat();
    GetImagesGrayMat();
    //清除images_mat_的内存，防止占用太多了
    vector<Mat>().swap(images_mat_);
    //用convertTo转化成0-1的数据 images_gray_float_mat_[0].type()为32fc1
    for (int i = 0; i < images_gray_mat_.size(); i++) {
        Mat tmp_float_mat;
        images_gray_mat_[i].convertTo(tmp_float_mat, CV_32FC1, 1.0/255);
        images_gray_float_mat_.push_back(tmp_float_mat);
    }
    //清除images_gray_mat_的内存
    vector<Mat>().swap(images_gray_mat_);
    //按曝光度排序
    sort(images_gray_float_mat_.begin(), images_gray_float_mat_.end(), SortMatCompare);
    //限制图的最大尺寸，如果长宽大于512，则等比缩放自适应调整至512
    //经测试，到这一步可以正常调整图片大小，图片类型转换无误
    //images_gray_float_mat_[0].convertTo(tmp, CV_8UC1, 255)转换成255的像素值
    AdjustImageSize();
    //SPDMEF算法执行，返回最终的结果Mat
    Mat result_image = SPDMEF();
    imwrite("SPDMEF_new.png", result_image);
}




//用于查看图片具体值
void MultiImagesFusion::SaveImage2Txt(const Mat &tosave_image) {
    string filename = "float.txt";
    ofstream outFile(filename.c_str(), ios_base::out);
    for (int r = 0; r < tosave_image.rows; r++) {
        for (int c = 0; c < tosave_image.cols; c++) {
            //****************************这里用的时候需要具体修改
            float data = tosave_image.at<float>(r,c);    //读取数据，at<type> - type 是矩阵元素的具体数据格式
                  outFile << data << "\t" ;    //每列数据用 tab 隔开
        }
             outFile << endl;    //换行
    }
}

//查看直方图
void MultiImagesFusion::DrawHist(Mat &hist, int type) {
    int hist_w=512;
        int hist_h=400;
        int width=2;
        Mat histImage=Mat::zeros(hist_h,hist_w,CV_8UC3);
        normalize(hist,hist,1,0,type,-1,Mat());
        for(int i=1;i<=hist.rows;++i){
            rectangle(histImage,Point(width*(i-1),hist_h-1),
                      Point(width*i-1,hist_h-cvRound(hist_h*hist.at<float>(i-1))-1),
                      Scalar(255,255,255),-1);
        }
        imwrite("testhist.png",histImage);
}


//测试函数
void MultiImagesFusion::TestFunction() {
//    GetImagesMat();
//    Mat gray;
//    cvtColor(images_mat_[0],gray,CV_BGR2GRAY);
//    cout<<images_mat_[0].type()<<endl;
//      Mat tmp_gray_mat;
//      cvtColor(images_mat_[2],tmp_gray_mat,CV_BGR2RGB);
//      imwrite("D:\\Code\\MatlabCode\\ef\\data\\test\\22.png",tmp_gray_mat);
    //
    //
//    GetImagesMat();
//    GetImagesGrayMat();
//    cout<<images_gray_mat_[0].type()<<endl;
    /*//MatchHistograms函数测试
    string image1_path = "C:\\Users\\siyun\\Desktop\\tmp\\1g.png";
    string image2_path = "C:\\Users\\siyun\\Desktop\\tmp\\2g.png";
    Mat image1 = imread(image1_path);
    Mat image1_gray;
    cvtColor(image1, image1_gray, COLOR_BGR2GRAY);
    Mat image2 = imread(image2_path);
    Mat image2_gray;
    cvtColor(image2, image2_gray, COLOR_BGR2GRAY);
    //match
    Mat result = MatchHistograms(image1_gray, image2_gray);
    //imwrite("matchres.png",result);
    */
    /* 测试图MatchHistograms测试
    GetImagesMat();
    GetImagesGrayMat();
    Mat result = MatchHistograms(images_gray_mat_[2], images_gray_mat_[1]);
    imwrite("2-1.png",result);
    */
}
