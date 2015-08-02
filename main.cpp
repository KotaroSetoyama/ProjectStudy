#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include <sstream>
#include <string>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <math.h>

#include "hog.h"
#include "cv.h"
#include "highgui.h"

std::vector<std::string> read(std::string folder){
    
    std::vector<std::string> filelist;
    DIR *dp;
    dirent* entry;
    int n = sizeof(folder) / sizeof(folder[0]);
    int i = 0;
    char *dir = new char[n]();
    
    do {
        dir[i] = folder[i];
        i++;
    }while(folder[i] != '\0');
    std::cout << dir << std::endl;
    
    dp = opendir(dir);
    if (dp==NULL)   exit(1);
    
    i = 0;
    while((entry = readdir(dp))){
        if(entry->d_name[0] == '.')  continue;
        std::cout << entry->d_name << std::endl;
        filelist.push_back(entry->d_name);
        i++;
    }
    
    delete [] dir;
    
    return filelist;
}

void trainImageClassification(std::vector<std::string> _files, std::vector<int> _labels){
    
    int SIZE_X = 40;    // ピクセル数
    int SIZE_Y = 40;    // ピクセル数
    cv::Mat src, dst;
    
    int FileSize = (int)_files.size();
    cv::Mat samples(FileSize, 9*9*((SIZE_X / 5)-2)*((SIZE_Y / 5)-2), CV_32FC1);
    cv::Mat labels(FileSize, 1, CV_32SC1);
    std::vector<float> hog;
    
    for(int i = 0; i < FileSize; i++){
        std::cout << i+1 << "/" << FileSize << std::endl;
        labels.at<int>(i, 0) = _labels[i];
        src = cv::imread(_files[i], 0);
        cv::resize(src, dst, cv::Size(SIZE_X, SIZE_Y));
        hog = GetHoG(dst.data, dst.cols, dst.rows);
        for(int j = 0; j < hog.size(); j++){
            samples.at<float>(i, j) = hog[j];
        }
    }
    
    CvSVM classifier;
    CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 100, 0.01);
    CvSVMParams svm_param = CvSVMParams(CvSVM::C_SVC, CvSVM::LINEAR, 0, 10, 0, 10.0, 0, 0, NULL, criteria);
    classifier.train(samples, labels, cv::Mat(), cv::Mat(), svm_param);
    
    classifier.save("../../ProjectStudy_test/ProjectStudy_test/svm.xml");
    std::cout << "succeed" << std::endl;
    
}

std::vector<float> GetHoG(unsigned char* img, int _SIZE_X, int _SIZE_Y, int _CELL_BIN, int _CELL_X, int _CELL_Y, int _BLOCK_X, int _BLOCK_Y){
    
    // HoG：ブロック(3×3セル)分の特徴ベクトルから81次元(9×9)のベクトルVを作る。1セルに何ピクセル(画素)か含まれる。1セル単位で全てのブロックのベクトルを合わせた次元ベクトルを作る。
    int CELL_X = _CELL_X;		// 1セル内の横画素数
    int CELL_Y = _CELL_Y;		// 1セル内の縦画素数
    int CELL_BIN = _CELL_BIN;	// 輝度勾配方向の分割数（普通は９）(20°ずつ)
    int BLOCK_X = _BLOCK_X;		// 1ブロック内の横セル数
    int BLOCK_Y = _BLOCK_Y;		// 1ブロック内の縦セル数
    
    int SIZE_X = _SIZE_X;		// リサイズ後の画像の横幅
    int SIZE_Y = _SIZE_Y;		// リサイズ後の画像の縦幅
    SIZE_X = 40;                // ピクセル数
    SIZE_Y = 40;                // ピクセル数
    
    int CELL_WIDTH = SIZE_X / CELL_X;						// セルの数（横）
    int CELL_HEIGHT = SIZE_Y / CELL_Y;						// セルの数（縦）
    int BLOCK_WIDTH = CELL_WIDTH - BLOCK_X + 1;				// ブロックの数（横）
    int BLOCK_HEIGHT = CELL_HEIGHT - BLOCK_Y + 1;			// ブロックの数（縦）
    
    int BLOCK_DIM = BLOCK_X * BLOCK_Y * CELL_BIN;			// １ブロックの特徴量次元
    int TOTAL_DIM = BLOCK_DIM * BLOCK_WIDTH * BLOCK_HEIGHT;	// HoG全体の次元
    
    double PI = 3.14;
    
    std::vector<float> feat(TOTAL_DIM, 0);
    
    //各セルの輝度勾配ヒストグラム
    std::vector<std::vector<std::vector<double> > > hist;
    hist.resize(CELL_WIDTH);
    for (int i = 0; i < hist.size(); i++){
        hist[i].resize(CELL_HEIGHT);
        for (int j = 0; j < hist[i].size(); j++){
            hist[i][j].resize(CELL_BIN, 0);
        }
    }
    
    //各ピクセルにおける輝度勾配強度mと勾配方向degを算出し、ヒストグラムへ
    //※端っこのピクセルでは、計算しない
    for (int y = 1; y<SIZE_Y - 1; y++){
        for (int x = 1; x<SIZE_X - 1; x++){
            double dx = img[y*SIZE_X + (x + 1)] - img[y*SIZE_X + (x - 1)];
            double dy = img[(y + 1)*SIZE_X + x] - img[(y - 1)*SIZE_X + x];
            double m = sqrt(dx*dx + dy*dy);
            double deg = (atan2(dy, dx) + PI) * 180.0 / PI;	//0.0〜360.0の範囲になるよう変換
            int bin = CELL_BIN * deg / 360.0;
            if (bin < 0) bin = 0;
            if (bin >= CELL_BIN) bin = CELL_BIN - 1;
            hist[(int)(x / CELL_X)][(int)(y / CELL_Y)][bin] += m;
        }
    }
    
    //ブロックごとで正規化
    for (int y = 0; y<BLOCK_HEIGHT; y++){
        for (int x = 0; x<BLOCK_WIDTH; x++){
            
            //このブロックの特徴ベクトル（次元BLOCK_DIM=BLOCK_X*BLOCK_Y*CELL_BIN）
            std::vector<double> vec;
            vec.resize(BLOCK_DIM, 0);
            
            for (int j = 0; j<BLOCK_Y; j++){
                for (int i = 0; i<BLOCK_X; i++){
                    for (int d = 0; d<CELL_BIN; d++){
                        int index = j*(BLOCK_X*CELL_BIN) + i*CELL_BIN + d;
                        vec[index] = hist[x + i][y + j][d];
                    }
                }
            }
            
            //ノルムを計算し、正規化
            double norm = 0.0;
            for (int i = 0; i<BLOCK_DIM; i++){
                norm += vec[i] * vec[i];
            }
            for (int i = 0; i<BLOCK_DIM; i++){
                vec[i] /= sqrt(norm + 1.0);
            }
            
            //featに代入
            for (int i = 0; i<BLOCK_DIM; i++){
                int index = y*BLOCK_WIDTH*BLOCK_DIM + x*BLOCK_DIM + i;
                feat[index] = vec[i];
            }
        }
    }
    return feat;
}

void train(){
    
    std::vector<int> labels;
    std::vector<std::string> files;
    std::vector<std::string> directory;
    
    directory.push_back("setoyama");
    directory.push_back("others");
    
    for (int i = 0; i < directory.size(); i++){
        std::cout << directory[i] << std::endl;
        std::vector<std::string> filelist = read("./images/" + directory[i] + "/");
        
        for(int j = 0; j < filelist.size(); j++){
            files.push_back("./images/" + directory[i] + "/" + filelist[j]);
            labels.push_back(i);        // setoyamaが0，othersが1
        }
    }
    trainImageClassification(files, labels);
}

// 顔検出(静止画)
void facedetection(){
    
    std::vector<std::string> directory;
    
    directory.push_back("setoyama");
    directory.push_back("others");
    
    // 正面顔検出器の読み込み
    CvHaarClassifierCascade* cvHCC = (CvHaarClassifierCascade*)cvLoad("/Users/setoyama/Programming/OpenCV/OpenCV-2.4.10/data/haarcascades/haarcascade_frontalface_default.xml", NULL, NULL, NULL);
    
    // 検出に必要なメモリストレージを用意する
    CvMemStorage* cvMStr = cvCreateMemStorage(0);
    
    // 顔検出対象の画像データ用
    IplImage* tarImg;
    
    // 検出情報を受け取るためのシーケンスを用意する
    CvSeq* face;
    
    //　検出対象の画像ファイルパス
    std::string ImgPath = "./images/";
    std::string number;
    std::string jpg = ".jpg";
    std::string tarDirectoryPath;
    std::string OutputDirectory;
    std::string tarFilePath;
    std::string OutputFile;
    int j;
    
    CvRect* faceRect;
    cv::Mat img;
    
    for (int i = 0; i < directory.size(); i++){
        
        tarDirectoryPath = ImgPath;
        tarDirectoryPath += directory[i];
        tarDirectoryPath += "/";
        tarDirectoryPath += directory[i];
        tarDirectoryPath += "-";
        OutputDirectory = ImgPath;
        OutputDirectory += directory[i];
        OutputDirectory += "2";
        OutputDirectory += "/";
        OutputDirectory += directory[i];
        j = 1;
        
        while(1){
            
            number = std::to_string(j);
            tarFilePath = tarDirectoryPath;
            tarFilePath += number;
            tarFilePath += jpg;
            OutputFile = OutputDirectory;
            OutputFile += number;
            OutputFile += jpg;
            
            // 画像データの読み込み
            tarImg = cvLoadImage(tarFilePath.c_str(), CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
            
            // 全ての画像を出力したら終了
            if(!tarImg) break;
            
            // 画像中から検出対象の情報を取得する
            face = cvHaarDetectObjects(tarImg, cvHCC, cvMStr, 1.1, 3, 0);
            
            for (int i = 0; i < face->total; i++) {
                
                // 検出情報から顔の位置情報を取得
                faceRect = (CvRect*)cvGetSeqElem(face, i);
                
                img = cv::cvarrToMat(tarImg);
                cv::Mat cut_img(img, cv::Rect(faceRect->x, faceRect->y, faceRect->width, faceRect->height));
                imwrite(OutputFile, cut_img);
            }
            
            // イメージの解放
            cvReleaseImage(&tarImg);
            
            // 次の画像の準備
            tarFilePath.clear();
            OutputFile.clear();
            j++;
        }
        
        tarDirectoryPath.clear();
        OutputDirectory.clear();
        
    }
    
    // 用意したメモリストレージを解放
    cvReleaseMemStorage(&cvMStr);
    
    // カスケード識別器の解放
    cvReleaseHaarClassifierCascade(&cvHCC);
    
}

int main(int argc, char* argv[]) {
    
    facedetection();
    
    train();
}