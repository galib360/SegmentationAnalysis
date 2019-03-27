#include <fstream>
#include <sstream>
#include <iostream>
#include <string.h>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>


using namespace cv;
using namespace dnn;
using namespace std;
using namespace std::chrono;

// Initialize the parameters
float confThreshold = 0.5; // Confidence threshold
float maskThreshold = 0.8; // Underestimation mask; more, the smaller the mask
float maskThreshold2 = 0.15;// Overestimation mask; less, the bigger the mask
int nFrames = 201;//Number of frames in a set
int view = 8;//Number of sets
int startView = 0;//Starting set, will run until it reaches the number of sets(View), Has to have same no. of frames
int startFrame =0;
int rectPad = 10;
float R =0, P=0, F=0, A=0;//Total evaluation
Rect BB;
ofstream myfile;


////For manipulating brightness and contrast
//float alpha = 2.2;
//int beta = 20;

vector<string> classes;
vector<Scalar> colors;

// Draw the predicted bounding box
void drawBox(Mat& frame, int classId, float conf, Rect box, Mat& objectMask);

// Postprocess the neural network's output for each frame
Mat postprocess(Mat& frame, const vector<Mat>& outs, int& countFrame, int& countView, Rect& BB);

int main()
{

	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	// Load names of classes
    string classesFile = "mscoco_labels.names";
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    // Load the colors
    string colorsFile = "colors.txt";
    ifstream colorFptr(colorsFile.c_str());
    while (getline(colorFptr, line)) {
        char* pEnd;
        double r, g, b;
        r = strtod (line.c_str(), &pEnd);
        g = strtod (pEnd, NULL);
        b = strtod (pEnd, NULL);
        Scalar color = Scalar(r, g, b, 255.0);
        colors.push_back(Scalar(r, g, b, 255.0));
    }

    // Give the configuration and weight files for the model
    String textGraph = "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
    String modelWeights = "frozen_inference_graph.pb";
    //String modelWeights = "saved_model.pb";

    // Load the network
    Net net = readNetFromTensorflow(modelWeights, textGraph);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    // Open a video file or an image file or a camera stream.
    string pathGT, pathData, outputFile;
    Mat frame, blob, silGT;

    // Create a window
    static const string kWinName = "Deep learning object detection in OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL);


    for (int countView = startView; countView < view; countView++) {
    	float Rset=0,Pset=0,Fset=0,Aset=0;
		for (int countFrame = startFrame; countFrame < nFrames; countFrame++) {
//			pathData = "experiment/cam0" + to_string(countView) + "/"
//					+ to_string(countFrame) + ".png";

			if (countView < 10) {
				pathData = "dancer4D/data/cam0" + to_string(countView) + "/"
						+ to_string(countFrame) + ".png";

				pathGT = "dancer4D/Ground/cam0" + to_string(countView) + "/"
						+ to_string(countFrame) + ".pbm";

			} else if (countView >= 10) {
				pathData = "dancer4D/data/cam" + to_string(countView) + "/"
						+ to_string(countFrame) + ".png";

				pathGT = "dancer4D/Ground/cam" + to_string(countView) + "/"
						+ to_string(countFrame) + ".pbm";

			}

//			pathData = "dancer4D/data/cam0" + to_string(countView) + "/"
//								+ to_string(countFrame) + ".png";
//
//			pathGT = "dancer4D/Ground/cam0" + to_string(countView) + "/"+ to_string(countFrame) + ".pbm";

			frame = imread(pathData);//original picture to estimate silhouette
			silGT = imread(pathGT);//Ground truth silhouette

			//resize(frame, frame, cv::Size(), 0.5, 0.5);
			//frame.convertTo(frame, -1, alpha, beta);

			if (frame.empty()) {
				cout << "No picture found !!!" << endl;
				break;
			}
//			imshow("frame", frame);
//			waitKey(0);
			blobFromImage(frame, blob, 1.0, Size(frame.cols, frame.rows),
					Scalar(), true, false);

			net.setInput(blob);

			// Runs the forward pass to get output from the output layers
			std::vector<String> outNames(2);
			outNames[0] = "detection_out_final";
			outNames[1] = "detection_masks";
			vector<Mat> outs;
			net.forward(outs, outNames);

			// Extract the bounding box, object mask and silhouettes
			Mat silEst = postprocess(frame, outs, countFrame, countView, BB);


			//Evaluation code goes here ------------>

			cvtColor( silGT, silGT, CV_BGR2GRAY );
			//cvtColor( silEst, silEst, CV_BGR2GRAY );
			cv::threshold(silGT, silGT, 20, 255, cv::THRESH_BINARY);
			cv::threshold(silEst, silEst, 20, 255, cv::THRESH_BINARY);
//			silEst.convertTo(silEst, CV_8U);
//			silGT.convertTo(silGT, CV_8U);

//			Mat ord(silEst.size(), CV_8U);
//			bitwise_or(silGT, silEst, ord);
//			imshow("OR", ord);
//			imshow("GT", silGT);
//			imshow("Est", silEst);
//			waitKey(0);

//			cout<<"ROWS: "<<silGT.rows<<", Cols : "<<silGT.cols<<endl;
//			cout<<"ROWS: "<<silEst.rows<<", Cols : "<<silEst.cols<<endl;

			float tn = 0, fp = 0, fn = 0, tp = 0;
			float recall, precision, f1, accuracy;

			int countG =0;
			int countE =0;
			for(int row =0; row<silGT.rows; row++){
				for(int col =0; col<silGT.cols; col++){
					if (silGT.at<uchar>(row,col)!=0){
						countG++;
					}
					if (silEst.at<uchar>(row, col) != 0) {
						countE++;
					}

				}
			}

//			for (int row = 0; row < silGT.rows; row++) {
//				for (int col = 0; col < silGT.cols; col++) {
//
//					if (silEst.at<uchar>(row, col) != 0) {
//						countE++;
//					}
//				}
//			}

//			cout<<"G 1s: "<<countG<<endl;
//			cout<<"E 1s: "<<countE<<endl;

			int x = BB.x;
			int y = BB.y;
			int height = BB.height;
			int width = BB.width;

			for(int row =y; row<y+height; row++){
				for(int col =x; col<x+width; col++){
					if (silGT.at<uchar>(row, col) == 0 && silEst.at<uchar>(row, col)!=0){
						fp+=1;
					}
					else if (silGT.at<uchar>(row, col) != 0 && silEst.at<uchar>(row, col)==0){
						fn+=1;
					}
					else if (silGT.at<uchar>(row, col) != 0 && silEst.at<uchar>(row, col)!=0){
						tp+=1;
					}
					else if (silGT.at<uchar>(row, col) == 0 && silEst.at<uchar>(row, col)==0){
						tn+=1;
					}
				}
			}

//			cout<<"tp, tn, fp, fn : "<< tp<<", "<<tn<<", "<<fp<<", "<<fn<<endl;
//			cout<<silGT.rows*silGT.cols<<endl;
//			cout<<(int)tp+tn+fp+fn<<endl;

			recall = tp/(tp+fn);
			precision = tp/(tp+fp);
			f1 = 2 * ((precision*recall)/(precision+recall));
			accuracy = (tp+tn)/(tp+tn+fp+fn);

//			cout<<"for set0"<<countView<<" ,frame "<<countFrame<<" ::::"<<endl;
//			cout<<"Recall: "<<recall<<endl;
//			cout<<"Precision: "<<precision<<endl;
//			cout<<"F1 Score: "<<f1<<endl;
//			cout<<"Accuracy: "<<accuracy<<endl;
//			cout<<" "<<endl;

			frame.release();
			silGT.release();
			silEst.release();

			Rset+=recall;
			Pset+=precision;
			Fset+=f1;
			Aset+=accuracy;

		}
		Rset = Rset/nFrames;
		Pset = Pset/nFrames;
		Fset = Fset/nFrames;
		Aset = Aset/nFrames;

		myfile.open("result.txt");
		myfile<<"Dancer4D"<<endl;
		myfile<<""<<endl;

		cout << "for set0" << countView<< ": "<< endl;
		cout << "Recall: " << Rset << endl;
		cout << "Precision: " << Pset << endl;
		cout << "F1 Score: " << Fset << endl;
		cout << "Accuracy: " << Aset << endl;
		cout <<" "<<endl;

		myfile << "for set0" << countView << ": " << endl;
		myfile << "Recall: " << Rset << endl;
		myfile << "Precision: " << Pset << endl;
		myfile << "F1 Score: " << Fset << endl;
		myfile << "Accuracy: " << Aset << endl;
		myfile << " "<<endl;

		R += Rset;
		P += Pset;
		F += Fset;
		A += Aset;

    }

    R = R/view;
    P = P/view;
    F = F/view;
    A = A/view;

	cout << "for all the sets: "<< endl;
	cout << "Recall: " << R << endl;
	cout << "Precision: " << P << endl;
	cout << "F1 Score: " << F << endl;
	cout << "Accuracy: " << A << endl;
	cout <<" "<<endl;

	myfile << "for all the sets: " << endl;
	myfile << "Recall: " << R << endl;
	myfile << "Precision: " << P << endl;
	myfile << "F1 Score: " << F << endl;
	myfile << "Accuracy: " << A << endl;
	myfile << " " << endl;
	myfile.close();

    high_resolution_clock::time_point t2 = high_resolution_clock::now();

	auto durationms = duration_cast<milliseconds>(t2 - t1).count();
	auto durations = duration_cast<seconds>(t2 - t1).count();
	cout << "Total code execution time: " << durationms << " ms" << endl;
	cout <<"Total code execution time: " <<durations<<" sec"<<endl;

    return 0;
}

// For each frame, extract the bounding box, object mask and silhouettes for each detected object in the input frame
Mat postprocess(Mat& frame, const vector<Mat>& outs, int& countFrame, int& countView, Rect& BB)
{
    Mat outDetections = outs[0];
    Mat outMasks = outs[1];

    // Output size of masks is NxCxHxW where
    // N - number of detected boxes
    // C - number of classes (excluding background)
    // HxW - segmentation shape
    const int numDetections = outDetections.size[2];
    const int numClasses = outMasks.size[1];

    outDetections = outDetections.reshape(1, outDetections.total() / 7);
    for (int i = 0; i < 1; ++i)//only taking one detection for all detection, i<numDetections
    {
        float score = outDetections.at<float>(i, 2);
        if (score > confThreshold)
        {
            // Extract the bounding box
            int classId = static_cast<int>(outDetections.at<float>(i, 1));
            int left = static_cast<int>(frame.cols * outDetections.at<float>(i, 3));
            int top = static_cast<int>(frame.rows * outDetections.at<float>(i, 4));
            int right = static_cast<int>(frame.cols * outDetections.at<float>(i, 5));
            int bottom = static_cast<int>(frame.rows * outDetections.at<float>(i, 6));

            left = max(0, min(left, frame.cols - 1))-rectPad;
            top = max(0, min(top, frame.rows - 1))-rectPad;
            right = max(0, min(right, frame.cols - 1))+rectPad;
            bottom = max(0, min(bottom, frame.rows - 1))+rectPad;
            Rect box = BB = Rect(left, top, right - left + 1, bottom - top + 1);

            // Extract the mask for the object
            Mat objectMask(outMasks.size[2], outMasks.size[3],CV_32F, outMasks.ptr<float>(i,classId));//small
            Mat objectMask2(outMasks.size[2], outMasks.size[3],CV_32F, outMasks.ptr<float>(i,classId));//big
			resize(objectMask, objectMask, Size(box.width, box.height));
			resize(objectMask2, objectMask2, Size(box.width, box.height));
			Mat mask = (objectMask > maskThreshold);
			Mat mask2 = (objectMask2 > maskThreshold2);
			Mat fg = Mat::zeros(frame.rows, frame.cols, CV_8U);
			Mat fg2 = Mat::zeros(frame.rows, frame.cols, CV_8U);
			mask.copyTo(fg(box));
			mask2.copyTo(fg2(box));
			cv::threshold(fg, fg, 20, 255, cv::THRESH_BINARY);
			cv::threshold(fg2, fg2, 20, 255, cv::THRESH_BINARY);
			//mask.convertTo(mask, CV_8UC1);
//			Mat coloredRoi = (0.3 * color + 0.7 * frame(box));
//			coloredRoi.convertTo(coloredRoi, CV_8UC3);

            // Draw bounding box, colorize and show the mask on the image
            //drawBox(frame, classId, score, box, objectMask);
//            imshow("fg", fg);
//            imshow("fg2", fg2);
//            waitKey(0);

            //Grabcut-------------------->

            Mat1b markers(frame.rows, frame.cols);
            // let's set all of them to possible background first
            markers.setTo(cv::GC_PR_BGD);

            // cut out a small area in the middle of the image
//            int m_rows = 0.1 * frame.rows;
//            int m_cols = 0.1 * frame.cols;
            int x = box.x;
            int y = box.y;
            int height = box.height;
            int width = box.width;

            // of course here you could also use cv::Rect() instead of cv::Range to select
            // the region of interest
//            cv::Mat1b fg_seed = markers(Range(((x+width/2) - 10), ((x+width/2) + 10)),
//            							Range(((y+height/2) - 10), ((y+height/2) + 10)));

            //Mat1b fg_seed = markers(Rect(((x+width)/2)-5, ((y+height)/2)-5, ((x+width)/2)+5, ((y+height)/2)+5));
            Mat1b fg_seed = markers(Rect(((x+width)/2)-5, ((y+height)/2)-5, 10, 10));
            Mat1b fg_rect = markers(Rect(left, top, right - left + 1, bottom - top + 1));
//            Mat1b fg_seed2 = markers(cv::Range(frame.rows/2 - m_rows/2, frame.rows/2 + m_rows/2),
//            		                 cv::Range(frame.cols/2 - m_cols/2, frame.cols/2 + m_cols/2));
//            Mat1b fg_vert = markers(Rect(((x+width)/2)-2, y+5, 20, height));
//            cv::Mat1b fg_seed = markers(cv::Range(frame.rows/2 - m_rows/2, frame.rows/2 + m_rows/2),
//                                        cv::Range(frame.cols/2 - m_cols/2, frame.cols/2 + m_cols/2));
            //cv::Mat1b fg_seed = markers(Rect(x, y, x+width, y+height));
            // mark it as foreground

            //fg_rect.setTo(GC_PR_FGD);
            //fg_seed2.setTo(GC_FGD);
            Mat1b fgp;
//            cout<<"width: "<<width<<" , height: "<<height<<" ,x: "<<x<<" ,y: "<<y<<endl;
//            cout<<"left "<<((x+width)/2)-5<<" , top: "<<((y+height)/2)-5<<" ,right: "<<((x+width)/2)+5<<" ,bottom: "<<((y+height)/2)+5<<endl;



            //Mat curr_sil_bin(fg.rows, fg.cols, CV_32F);
			//int count = 0;
			for (int row = y; row < y+height; row++) {
				for (int col = x; col < x+width; col++) {
					if (fg2.at<uchar>(row, col) != 0) {
						if (fg.at<uchar>(row, col) != 0) {
							//curr_sil_bin.at<float>(row, col) = 1;
							//count++;
							markers.at<uchar>(row, col) = GC_FGD;

						}
						else{
							markers.at<uchar>(row, col) = GC_PR_FGD;
						}
					}
					else{
						markers.at<uchar>(row, col) = GC_BGD;
					}

//					if(fg.at<uchar>(row, col) == 0){
//						//curr_sil_bin.at<float>(row, col) = 0;
//						markers.at<uchar>(row, col) = GC_PR_FGD;
//					}
//					if (fg2.at<uchar>(row, col) == 0) {
//						//curr_sil_bin.at<float>(row, col) = 0;
//						markers.at<uchar>(row, col) = GC_BGD;
//					}
				}
			}

			//fg_seed.setTo(GC_FGD);
			//fg_vert.setTo(GC_FGD);
			//fg_seed=GC_FGD;
			//fg_vert=GC_FGD;


            // select pixels that lie outside the bounding rect
            //cv::Mat1b bg_seed = markers(cv::Range(0, 5),cv::Range::all());
            Mat1b bg_seed1 = markers(cv::Range::all(),cv::Range(0, x));
            Mat1b bg_seed2 = markers(cv::Range::all(),cv::Range(x+width, frame.cols));
            Mat1b bg_seed3 = markers(cv::Range(y+height, frame.rows),cv::Range::all());
            Mat1b bg_seed4 = markers(cv::Range(0, y),cv::Range::all());

//            cv::Mat1b bg_seed5 = markers(cv::Range(y+height-20, y+height),cv::Range(x, x+20));
//            cv::Mat1b bg_seed6 = markers(cv::Range(y+height-5, y+height),cv::Range((x+width)/2, x+width));
//            Mat1b bg_seed;
//            bg_seed1.copyTo(bg_seed);
//            bg_seed2.copyTo(bg_seed);

            //Setting surely background pixels
            //bg_seed.setTo(cv::GC_BGD);
            bg_seed1.setTo(GC_BGD);
            bg_seed2.setTo(GC_BGD);
            bg_seed3.setTo(GC_BGD);
            bg_seed4.setTo(GC_BGD);
            //bg_seed5.setTo(GC_BGD);
            //bg_seed6.setTo(GC_BGD);

            //with Mask

            Mat bgd, fgd;
            int iterations = 1;
            cv::grabCut(frame, markers, Rect(), bgd, fgd, iterations, cv::GC_EVAL);
            //cv::grabCut(frame, markers, Rect(), bgd, fgd, iterations, cv::GC_INIT_WITH_MASK);
            // let's get all foreground and possible foreground pixels
            cv::Mat1b mask_fgpf = ( markers == cv::GC_FGD) | ( markers == cv::GC_PR_FGD);
            // and copy all the foreground-pixels to a temporary image
            cv::Mat3b tmp = cv::Mat3b::zeros(frame.rows, frame.cols);
//            imshow("temp", mask_fgpf);
//            waitKey(0);
            frame.copyTo(tmp, mask_fgpf);

            //smoothen
            for ( int i = 1; i < 7; i = i + 2 ){
            	medianBlur ( mask_fgpf, mask_fgpf, i );
            }

            // show it
//            cv::imshow("foreground", tmp);
//            cv::waitKey(0);

            string outputPath;
            if(countView<10){
            	outputPath = "dancer4D/Est/cam0" + to_string(countView)+ "/"+to_string(countFrame)+".png";
            }
            else if(countView>=10){
            	outputPath = "dancer4D/Est/cam" + to_string(countView)+ "/"+to_string(countFrame)+".png";
            }

//            string outputPath = "dancer4D/Est/cam0" + to_string(countView)+ "/"+to_string(countFrame)+".png";
//            string outputPath = "experiment/cam0" + to_string(countView) + "/silscam0" + to_string(countView)+ "/"+to_string(countFrame)+".jpg";
            imwrite(outputPath, mask_fgpf);
            return mask_fgpf;

        }
    }
}

// Draw the predicted bounding box, colorize and show the mask on the image
void drawBox(Mat& frame, int classId, float conf, Rect box, Mat& objectMask)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(box.x, box.y), Point(box.x+box.width, box.y+box.height), Scalar(255, 178, 50), 3);

    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }

    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    box.y = max(box.y, labelSize.height);
    rectangle(frame, Point(box.x, box.y - round(1.5*labelSize.height)), Point(box.x + round(1.5*labelSize.width), box.y + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(box.x, box.y), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);

    Scalar color = colors[classId%colors.size()];

    // Resize the mask, threshold, color and apply it on the image
    resize(objectMask, objectMask, Size(box.width, box.height));
    Mat mask = (objectMask > maskThreshold);
    Mat coloredRoi = (0.3 * color + 0.7 * frame(box));
    coloredRoi.convertTo(coloredRoi, CV_8UC3);

    // Draw the contours on the image
    vector<Mat> contours;
    Mat hierarchy;
    mask.convertTo(mask, CV_8U);
    findContours(mask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
    drawContours(coloredRoi, contours, -1, color, 5, LINE_8, hierarchy, 100);
    coloredRoi.copyTo(frame(box), mask);

}
