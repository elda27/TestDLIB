#include <iostream>

#include <opencv2/core.hpp>

#include <dlib/opencv.h>
#include <dlib/data_io.h>
#include <dlib/dnn.h>
#include <dlib/array2d.h>
#include <dlib/image_transforms.h>

#include "ReadMNISTDatas.hpp"
#include "TrainFilePath.hpp"

//#if defined(DEBUG) || defined(_DEBUG)
//#pragma comment(lib, "opencv_core310d.lib")
//#pragma comment(lib, "opencv_highgui310d.lib")
//#pragma comment(lib, "dlibd.lib")
//#else
//#pragma comment(lib, "opencv_core310.lib")
//#pragma comment(lib, "opencv_highgui310.lib")
//#pragma comment(lib, "dlib.lib")
//#endif

void outputResult(std::ostream& output, std::vector<unsigned long> answer, std::vector<unsigned long> predict);

int main()
{
  std::vector<cv::Mat> cv_images;
  std::vector<int> tmp_labels;
  
  // 教師データの読み込み．
  readImages(INPUT_TRAIN_IMAGE_PATH, cv_images);
  readLabels(INPUT_TRAIN_LABEL_PATH, tmp_labels);

  // OpenCV -> dlib へ変換
  std::vector<dlib::array2d<uchar>> train_images(cv_images.size());
  std::vector<unsigned long> train_labels(std::begin(tmp_labels), std::end(tmp_labels));
  std::for_each(std::begin(cv_images), std::end(cv_images),
    [&train_images](cv::Mat const& cv_img) 
    {
      static int i = 0;
      dlib::assign_image(train_images[i], dlib::cv_image<uchar>(cv_img));
      ++i;
    }
  );

  // 非教師データの読み込み
  cv_images.clear();
  tmp_labels.clear();
  readImages(INPUT_UNKNOWN_IMAGE_PATH, cv_images);
  readLabels(INPUT_UNKNOWN_LABEL_PATH, tmp_labels);
  std::vector<dlib::array2d<uchar>> unknown_images(cv_images.size());
  std::vector<unsigned long> unknown_labels(std::begin(tmp_labels), std::end(tmp_labels));
  std::for_each(std::begin(cv_images), std::end(cv_images),
    [&unknown_images](cv::Mat const& cv_img)
    {
      static int i = 0;
      dlib::assign_image(unknown_images[i], dlib::cv_image<uchar>(cv_img));
      ++i;
    }
  );



  // CNN の定義
  // おそらく A<B<C>> となっている場合 Cが入力で，Bが中間層，Aが出力層
  // なので，出力はfcにするのがいいと思う．
  // fc<10, ...>:Fully connected layerでノード数は10
  // relu:活性化関数の名前．詳しくはReLUで調べてください
  // con<16,5,5,1,1,SUBNET> で5✕5のフィルタサイズを1✕1のstrideで畳み込みするノードが16個ある
  // max_pool<2, 2, 2, 2, SUBNET> 2✕2のウインドウサイズで2✕2のstrideでプーリングを行う．
  // relu<fc<84, ...>> この場合活性化関数がReLUで84ノードからなる層を定義している．
  // max_pool<2,2,2,2,relu<con<16,5,5,1,1,SUBNET>>> これでconvolutionした結果をReLU関数で活性化してそれをMax poolingする
  // input<array2d<uchar>> cv_image<uchar>を入力に取る．現在cv::Matを入力に取れるように試行錯誤中
  using net_type = dlib::loss_multiclass_log<
    dlib::fc<10,
    dlib::relu<dlib::fc<84,
    dlib::relu<dlib::fc<120,
    dlib::max_pool<2, 2, 2, 2, dlib::relu<dlib::con<16, 5, 5, 1, 1,
    dlib::max_pool<2, 2, 2, 2, dlib::relu<dlib::con<6, 5, 5, 1, 1,
    dlib::input<dlib::array2d<uchar>>
    >>>>>>>>>>>>;

  // 上のCNN場合
  // -FC-> : Fully connectedな接続
  // -> : 重みを共有した接続
  // 入力画像->[6ノードの畳み込み層]->プーリング層->[16ノードの畳み込み層]->プーリング層-FC-> ...
  //          [120ノードの普通のNN]-FC>[84ノードの普通のNN]-FC>出力(10次元ベクトルで各次元に各数字の確率が保存される)
  
  net_type net;
  dlib::dnn_trainer<net_type> trainer(net);

  trainer.set_learning_rate(0.01);          // 学習計数
  trainer.set_min_learning_rate(0.00001);   // 学習計数の最低値．上の学習計数から学習回数を増やす毎にこの値へと減少していく
  trainer.set_mini_batch_size(128);         // なにこれ？バッチサイズ．
  trainer.be_verbose();                     // おそらくログの出力

  // DNNの学習には尋常ではないほどの時間がかかるので20秒おきに自動で現在の状態を保存する
  // なお，途中で終了した場合もmnist_syncファイルがあれば再起してくれる．
  trainer.set_synchronization_file("mnist_sync", std::chrono::seconds(20));
  trainer.train(train_images, train_labels);

  // 保存前にclear．
  net.clean();

  // 多分学習結果の保存．
  dlib::serialize("mnist_network.dat") << net;


  std::cout << "[Input trained labels]" << std::endl;
  std::vector<unsigned long> predicted_labels = net(train_images);
  outputResult(std::cout, train_labels, predicted_labels);

  std::cout << "[Input unknown labels]" << std::endl;
  predicted_labels = net(unknown_images);
  outputResult(std::cout, unknown_labels, predicted_labels);

  return 0;
}


void outputResult(std::ostream& output, std::vector<unsigned long> answer, std::vector<unsigned long> predict)
{
  int num_correct = 0;
  int num_wrong = 0;
  // And then let's see if it classified them correctly.
  for (size_t i = 0; i < answer.size(); ++i)
  {
    if (predict[i] == answer[i])
    {
      ++num_correct;
    }
    else
    {
      ++num_wrong;
    }
  }

  output << "testing num_right: " << num_correct << std::endl;
  output << "testing num_wrong: " << num_wrong << std::endl;
  output << "testing accuracy:  " << num_correct / static_cast<double>(num_correct + num_wrong) << std::endl;
}
