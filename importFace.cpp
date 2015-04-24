#include <stdio.h>
#include <string>
#include <iostream>
#include <stdlib.h>
#include "opencv2/core/core_c.h"
#include "opencv2/core/core.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/objdetect/objdetect.hpp"

using namespace cv;
using namespace std;

void getFilename(const string &path,string &filename, string &type){
  int slapos = path.find_last_of("/");
  filename = path.substr(slapos+1, path.size()-slapos);
  int dotpos = filename.find_last_of(".");
  type = filename.substr(dotpos+1, filename.size() - dotpos);
  filename = filename.substr(0, dotpos);
}

void int2str(const int &int_temp,string &string_temp)
{
        stringstream stream;
        stream<<int_temp;
        string_temp=stream.str();   
}


int main(int argc, char** argv)
{
	printf("Cutface processing...\n");
    CvMemStorage * pStorage = 0;        // expandable memory buffer
    CvSeq * pFaceRectSeq;               // list of detected faces
    int i;
	string index;
	string filename;
	string type;
	string finalname;
	getFilename(argv[1],filename,type);
    // initializations
    IplImage * pInpImg =  cvLoadImage(argv[1], CV_LOAD_IMAGE_COLOR);
    pStorage = cvCreateMemStorage(0);
    string face_cascade_name = "haarcascade_frontalface_alt.xml";
    CascadeClassifier pCascade;

    if( !pCascade.load(face_cascade_name) )
    {
       cerr << "ERROR: Could not load classifier cascade" << endl;
       return -1;
    }

	IplImage *tmp;
    // validate that everything initialized properly
    if( !pInpImg || !pStorage ){
        exit(-1);
    }

    // detect faces in image
    std::vector<Rect> faces;
    pCascade.detectMultiScale( pInpImg, faces, 1.1, 3, 0|CV_HAAR_SCALE_IMAGE, Size(0, 0) );

    // draw a rectangular outline around each detection
    for(  i = 0; i < faces.size(); i++ )
	{
        
        int2str(i,index);
		finalname = "/data/temp_image/"+filename+"_"+index+"."+type;
		int x1 = (faces[i].x - 0.2*faces[i].width)>=0?(faces[i].x - 0.2*faces[i].width):0;
		int y1 = (faces[i].y - 0.5*faces[i].height)>=0?(faces[i].y - 0.5*faces[i].height):0;
		int x2 = (faces[i].x + 1.2*faces[i].width)<=pInpImg->width?(faces[i].x + 1.2*faces[i].width):pInpImg->width;
		int y2 = (faces[i].y + 1.5*faces[i].height)<=pInpImg->height?(faces[i].y + 1.5*faces[i].height):pInpImg->height;

        CvPoint pt1 = { x1 , y1 };
		CvPoint pt2 = { x2 , y2 };
        //cvRectangle(pInpImg, pt1, pt2, CV_RGB(0,255,0), 3, 4, 0);

		cvSetImageROI(pInpImg, cvRect(x1, y1, x2-x1, y2-y1));
		tmp = cvCreateImage(cvGetSize(pInpImg),
                               pInpImg->depth,
                               pInpImg->nChannels);

		cvCopy(pInpImg, tmp, NULL);

		Mat mat_img(tmp);
		Mat output_img;
		resize(mat_img,output_img,Size(240,300),0,0,CV_INTER_LINEAR);

		if(imwrite(finalname,output_img))
		{
			cout<<"save "<<finalname<<"\n";
		}

		cvResetImageROI(pInpImg);
		
    }


    // clean up and release resources
    cvReleaseImage(&pInpImg);
	cvReleaseImage(&tmp);
    //if(pCascade) cvReleaseHaarClassifierCascade(&pCascade);
    if(pStorage) cvReleaseMemStorage(&pStorage);

	cout<<"all finish.";
    return 0;
}