#include<opencv2/opencv.hpp>
#include<iostream>
#include <unordered_map>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"

using namespace std;
using namespace cv;

// ------------------------------------------------------------------------ //
Mat src;
Mat src_gray;
Mat src_gray_filtered;
Mat canny_output;
RNG rng(12345);
// ------------------------------------------------------------------------ //
// trackbar 
const int alpha_slider_max = 500, k_slider_max = 16;
int alpha_sliderT1 = 60, alpha_sliderT2 = 230, alpha_sliderTH = 170, alpha_sliderML = 13, alpha_sliderMG = 5, alpha_sliderFS = 15, alpha_sliderK = 3, alpha_sliderURG = 60;
double alphaT1, alphaT2, alphaTH, alphaML, alphaMG, alphaFS, alphaK, alphaURG;
double betaT1, betaT2, betaTH, betaML, betaMG, betaFS, betaK, betaURG;
Point pBL, pUR;
int nbClick = 0;
vector<Point> box_numbers;

static void on_trackbar(int, void*)
{
	alphaT1 = (double)alpha_sliderT1 / alpha_slider_max;
	betaT1 = (1.0 - alphaT1);
	alphaT2 = (double)alpha_sliderT2 / alpha_slider_max;
	betaT2 = (1.0 - alphaT2);
	alphaTH = (double)alpha_sliderTH / alpha_slider_max;
	betaTH = (1.0 - alphaTH);
	alphaML = (double)alpha_sliderML / alpha_slider_max;
	betaML = (1.0 - alphaML);
	alphaMG = (double)alpha_sliderMG / alpha_slider_max;
	betaMG = (1.0 - alphaMG);
	alphaFS = (double)alpha_sliderFS / alpha_slider_max;
	betaFS = (1.0 - alphaFS);
	alphaK = (double)alpha_sliderK / k_slider_max;
	betaK = (1.0 - alphaK);
	alphaURG = (double)alpha_sliderURG / alpha_slider_max;
	betaURG = (1.0 - alphaURG);
}

static void onClick(int event, int x, int y, int, void*) {
	if (event != EVENT_LBUTTONDOWN)
		return;
	if (nbClick == 0) {
		pBL = Point(x, y);
		nbClick++;
		return;
	}
	if (nbClick == 1) {
		pUR = Point(x, y);
		nbClick++;
		return;
	}
}

int main()
{
	namedWindow("Parameters", WINDOW_NORMAL);
	char Thresh1[50];
	sprintf(Thresh1, "Thresh1");
	createTrackbar(Thresh1, "Parameters", &alpha_sliderT1, alpha_slider_max, on_trackbar);
	char Thresh2[50];
	sprintf(Thresh2, "Thresh2");
	createTrackbar(Thresh2, "Parameters", &alpha_sliderT2, alpha_slider_max, on_trackbar);
	char ThreshH[50];
	sprintf(ThreshH, "ThreshH");
	createTrackbar(ThreshH, "Parameters", &alpha_sliderTH, alpha_slider_max, on_trackbar);
	char MinLineLength[50];
	sprintf(MinLineLength, "MLL");
	createTrackbar(MinLineLength, "Parameters", &alpha_sliderML, alpha_slider_max, on_trackbar);
	char MaxLineGap[50];
	sprintf(MaxLineGap, "MLG");
	createTrackbar(MaxLineGap, "Parameters", &alpha_sliderMG, alpha_slider_max, on_trackbar);
	char FilterSize[50];
	sprintf(FilterSize, "FilterSize");
	createTrackbar(FilterSize, "Parameters", &alpha_sliderFS, alpha_slider_max, on_trackbar);
	char Kmeans[50];
	sprintf(Kmeans, "K", k_slider_max);
	createTrackbar(Kmeans, "Parameters", &alpha_sliderK, k_slider_max, on_trackbar);
	char URG[50];
	sprintf(URG, "Seuil URG");
	createTrackbar(URG, "Parameters", &alpha_sliderURG, alpha_slider_max, on_trackbar);

	// Recuperation de l'image 
	int current_image = 0;
	vector<string> images_color = { "Paquet 3/planks-color/planks-color-1.jpeg","Paquet 3/planks-color/planks-color-2.jpeg","Paquet 3/planks-color/planks-color-3.jpeg","Paquet 3/planks-color/planks-color-4.jpeg","Paquet 3/planks-color/planks-color-5.jpeg" };
	vector<string> images_billon = { "Paquet 3/cross-section-scale/A01a.jpeg","Paquet 3/cross-section-scale/A01c.jpeg","Paquet 3/cross-section-scale/A03a.jpeg","Paquet 3/cross-section-scale/A03c.jpeg","Paquet 3/cross-section-scale/A04a.jpeg","Paquet 3/cross-section-scale/A04c.jpeg","Paquet 3/cross-section-scale/A05a.jpeg","Paquet 3/cross-section-scale/A05c.jpeg","Paquet 3/cross-section-scale/B01a.jpeg","Paquet 3/cross-section-scale/B01c.jpeg","Paquet 3/cross-section-scale/B08a.jpeg","Paquet 3/cross-section-scale/B08c.jpeg","Paquet 3/cross-section-scale/B09a.jpeg","Paquet 3/cross-section-scale/B09c.jpeg","Paquet 3/cross-section-scale/B10a.jpeg","Paquet 3/cross-section-scale/B10c.jpeg","Paquet 3/cross-section-scale/C02a.jpeg","Paquet 3/cross-section-scale/C02c.jpeg","Paquet 3/cross-section-scale/C04a.jpeg","Paquet 3/cross-section-scale/C04c.jpeg","Paquet 3/cross-section-scale/C08a.jpeg","Paquet 3/cross-section-scale/C08c.jpeg","Paquet 3/cross-section-scale/C13a.jpeg","Paquet 3/cross-section-scale/C13c.jpeg","Paquet 3/cross-section-scale/D02a.jpeg","Paquet 3/cross-section-scale/D02c.jpeg","Paquet 3/cross-section-scale/D03a.jpeg","Paquet 3/cross-section-scale/D03c.jpeg","Paquet 3/cross-section-scale/D09a.jpeg","Paquet 3/cross-section-scale/D09c.jpeg","Paquet 3/cross-section-scale/D12a.jpeg","Paquet 3/cross-section-scale/D12c.jpeg" };
	vector<string> images_number = { "Paquet 3/planks-numbers/planks-numbers-1.jpeg", "Paquet 3/planks-numbers/planks-numbers-2.jpeg", "Paquet 3/planks-numbers/planks-numbers-3.jpeg", "Paquet 3/planks-numbers/planks-numbers-4.jpeg", "Paquet 3/planks-numbers/planks-numbers-5.jpeg" };
	//src = imread(images_number[current_image]);
	namedWindow("Image source", WINDOW_NORMAL);
	//imshow("Image source", src);

	while (true) {
		char key = waitKey(0);
		cout << key << endl;

		// Morphologic horizontale and vertical 
		if (key == 'y') {
			cvtColor(src, src_gray, CV_BGR2GRAY);
			Mat bw;
			adaptiveThreshold(src_gray, bw, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2);
			// Create the images that will use to extract the horizontal and vertical lines
			Mat horizontal = bw.clone();
			Mat vertical = bw.clone();
			// Create structure element for extracting horizontal lines through morphology operations
			Mat horizontalStructure = getStructuringElement(MORPH_RECT, Size(alpha_sliderFS, 1));
			// Apply morphology operations
			erode(horizontal, horizontal, horizontalStructure, Point(-1, -1));
			dilate(horizontal, horizontal, horizontalStructure, Point(-1, -1));
			// Show extracted horizontal lines
			namedWindow("horizontal", WINDOW_NORMAL);
			imshow("horizontal", horizontal);
			imwrite("horizontal.jpg", horizontal);
			// Create structure element for extracting vertical lines through morphology operations
			Mat verticalStructure = getStructuringElement(MORPH_RECT, Size(1, alpha_sliderFS));
			// Apply morphology operations
			erode(vertical, vertical, verticalStructure, Point(-1, -1));
			dilate(vertical, vertical, verticalStructure, Point(-1, -1));
			// Show extracted vertical lines
			namedWindow("vertical", WINDOW_NORMAL);
			imshow("vertical", vertical);
			imwrite("vertical.jpg", vertical);

			// Preparation image resultat 
			bitwise_or(vertical, horizontal, bw);

			namedWindow("Contours", WINDOW_NORMAL);
			imshow("Contours", bw);



		}
		// Canny + Hough 
		if (key == 'm') {
			Mat dst, cdst;
			//GaussianBlur(src, cdst, Size(9, 9), 2);
			cvtColor(src, src_gray, CV_BGR2GRAY);
			Canny(src, dst, alpha_sliderT1, alpha_sliderT2, 3);
			cvtColor(src_gray, cdst, CV_GRAY2BGR);

			vector<Vec2f> lines;
			HoughLines(dst, lines, 1, CV_PI / 180, alpha_sliderTH);

			for (size_t i = 0; i < lines.size(); i++)
			{
				float rho = lines[i][0];
				float theta = lines[i][1];
				double a = cos(theta), b = sin(theta);
				double x0 = a * rho, y0 = b * rho;
				Point pt1(cvRound(x0 + 10000 * (-b)),
					cvRound(y0 + 10000 * (a)));
				Point pt2(cvRound(x0 - 10000 * (-b)),
					cvRound(y0 - 10000 * (a)));
				line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
			}

			namedWindow("Res hough", WINDOW_NORMAL);
			imshow("Res hough", cdst);

		}
		// Canny + Hough proba
		if (key == 'p') {
			Mat dst, cdst;
			//GaussianBlur(src, cdst, Size(9, 9), 2);
			cvtColor(src, src_gray, CV_BGR2GRAY);
			Canny(src, dst, alpha_sliderT1, alpha_sliderT2, 3);
			cvtColor(src_gray, cdst, CV_GRAY2BGR);

			vector<Vec4i> lines;
			HoughLinesP(dst, lines, 1, CV_PI / 180, alpha_sliderTH, alpha_sliderML, alpha_sliderMG);
			for (size_t i = 0; i < lines.size(); i++)
			{
				Vec4i l = lines[i];
				line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 5, CV_AA);
			}

			namedWindow("Res", WINDOW_NORMAL);
			imshow("Res", cdst);

		}
		// Canny + CannyClean
		if (key == 'o') {
			src = imread("Paquet 3/Zhongli.png");
			// Conversion en niveau de gris 
			cvtColor(src, src_gray, CV_BGR2GRAY);
			namedWindow("Image gray", WINDOW_NORMAL);
			imshow("Image gray", src_gray);
			/*
			// On diminue la texture de bois
			//bilateralFilter(src_gray, src_gray_filtered, 9, 200, 200);
			GaussianBlur(src_gray, src_gray_filtered, Size(9, 9), 2);
			imwrite("filtered.jpg", src_gray_filtered);
			namedWindow("Image gray filtered", WINDOW_NORMAL);
			imshow("Image gray filtered", src_gray_filtered);
			*/
			/// Detect edges using canny
			Canny(src_gray, canny_output, alpha_sliderT1, alpha_sliderT2, 3);
			namedWindow("Canny", WINDOW_NORMAL);
			imshow("Canny", canny_output);
			imwrite("Canny.jpg", canny_output);
			// Truc un peu magique 
			// On fait une fermeture sur canny avec un gros filtre 
			// On extrait les bord de la fermeture avec une egalite entre la dilatation des deux composante par un filtre 5x5 
			// On floute fortement ce qui n'est pas un bord puis on remet les bords non flouté dans l'image
			// On detecte a nouveau les bord avec Canny 

			Mat mask, mask1, mask2;
			morphologyEx(canny_output, mask1, MORPH_CLOSE, getStructuringElement(CV_SHAPE_RECT, Size(alpha_sliderFS, alpha_sliderFS)));
			bitwise_not(mask1, mask2);

			dilate(mask1, mask1, getStructuringElement(CV_SHAPE_RECT, Size(5, 5)));
			dilate(mask2, mask2, getStructuringElement(CV_SHAPE_RECT, Size(5, 5)));

			bitwise_and(mask1, mask2, mask);

			Mat test_b, test_o;
			GaussianBlur(src_gray, test_b, Size(9, 9), 2);
			bitwise_and(src_gray, mask, test_o);
			bitwise_not(mask, mask);
			bitwise_and(test_b, mask, test_b);
			mask = test_b + test_o;

			Canny(mask, canny_output, alpha_sliderT1, alpha_sliderT2, 3);
			namedWindow("CannyClean", WINDOW_NORMAL);
			imshow("CannyClean", canny_output);
			imwrite("CannyClean.jpg", canny_output);

			vector<Vec2f> lines;
			HoughLines(canny_output, lines, 1, CV_PI / 180, alpha_sliderTH);
			cvtColor(src_gray, src_gray, COLOR_GRAY2RGB);
			for (size_t i = 0; i < lines.size(); i++)
			{
				float rho = lines[i][0];
				float theta = lines[i][1];
				double a = cos(theta), b = sin(theta);
				double x0 = a * rho, y0 = b * rho;
				Point pt1(cvRound(x0 + 10000 * (-b)),
					cvRound(y0 + 10000 * (a)));
				Point pt2(cvRound(x0 - 10000 * (-b)),
					cvRound(y0 - 10000 * (a)));
				if (theta > CV_PI / 2 - CV_PI / 32 && theta < CV_PI / 2 + CV_PI / 32)
					line(src_gray, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
			}

			namedWindow("Res hough", WINDOW_NORMAL);
			imshow("Res hough", src_gray);

		}
		// Watershed
		if (key == 'i') {
			src = imread(images_number[current_image]);
			//GaussianBlur(src, src, Size(alpha_sliderFS, alpha_sliderFS), 3, 2);
			// Create a kernel that we will use to sharpen our image
			Mat kernel = (Mat_<float>(3, 3) <<
				1, 1, 1,
				1, -8, 1,
				1, 1, 1);
			Mat imgLaplacian;
			filter2D(src, imgLaplacian, CV_32F, kernel);
			Mat sharp;
			src.convertTo(sharp, CV_32F);
			Mat imgResult = sharp - imgLaplacian;
			// convert back to 8bits gray scale
			imgResult.convertTo(imgResult, CV_8UC3);
			imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
			namedWindow("Laplace Filtered Image", WINDOW_NORMAL);
			imshow("Laplace Filtered Image", imgLaplacian);
			namedWindow("New Sharped Image", WINDOW_NORMAL);
			imshow("New Sharped Image", imgResult);
			// Create binary image from source image
			Mat bw;
			cvtColor(imgResult, bw, COLOR_BGR2GRAY);
			threshold(bw, bw, 40, 255, THRESH_BINARY | THRESH_OTSU);
			// Supression du bruit 
			GaussianBlur(bw, bw, Size(5, 5), 2);
			//Mat element = Mat::ones(Size(3, 3), CV_8U);
			//morphologyEx(bw, bw, MORPH_CLOSE, element);
			namedWindow("Binary Image", WINDOW_NORMAL);
			imshow("Binary Image", bw);
			// Perform the distance transform algorithm
			Mat dist;
			distanceTransform(bw, dist, DIST_L2, 3);
			// Normalize the distance image for range = {0.0, 1.0}
			// so we can visualize and threshold it
			normalize(dist, dist, 0, 1.0, NORM_MINMAX);
			namedWindow("Distance Transform Image", WINDOW_NORMAL);
			imshow("Distance Transform Image", dist);
			imwrite("dst.jpg", dist);
			// Threshold to obtain the peaks
			// This will be the markers for the foreground objects
			threshold(dist, dist, 0.4, 1.0, THRESH_BINARY);
			// Dilate a bit the dist image
			Mat kernel1 = Mat::ones(3, 3, CV_8U);
			dilate(dist, dist, kernel1);
			namedWindow("Peaks", WINDOW_NORMAL);
			imshow("Peaks", dist);
			// Create the CV_8U version of the distance image
			// It is needed for findContours()
			Mat dist_8u;
			dist.convertTo(dist_8u, CV_8U);
			// Find total markers
			vector<vector<Point> > contours;
			findContours(dist_8u, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
			// Create the marker image for the watershed algorithm
			Mat markers = Mat::zeros(dist.size(), CV_32S);
			// Draw the foreground markers
			for (size_t i = 0; i < contours.size(); i++)
			{
				drawContours(markers, contours, static_cast<int>(i), Scalar(static_cast<int>(i) + 1), -1);
			}
			// Draw the background marker
			circle(markers, Point(5, 5), 3, Scalar(255), -1);
			namedWindow("Markers", WINDOW_NORMAL);
			imshow("Markers", markers * 10000);
			// Perform the watershed algorithm
			watershed(imgResult, markers);
			Mat mark;
			markers.convertTo(mark, CV_8U);
			bitwise_not(mark, mark);
			//    imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
			// image looks like at that point
			// Generate random colors
			vector<Vec3b> colors;
			for (size_t i = 0; i < contours.size(); i++)
			{
				int b = theRNG().uniform(0, 256);
				int g = theRNG().uniform(0, 256);
				int r = theRNG().uniform(0, 256);
				colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
			}
			// Create the result image
			Mat dst = Mat::zeros(markers.size(), CV_8UC3);
			// Fill labeled objects with random colors
			for (int i = 0; i < markers.rows; i++)
			{
				for (int j = 0; j < markers.cols; j++)
				{
					int index = markers.at<int>(i, j);
					if (index > 0 && index <= static_cast<int>(contours.size()))
					{
						dst.at<Vec3b>(i, j) = colors[index - 1];
					}
				}
			}
			// Visualize the final image
			namedWindow("Final Result", WINDOW_NORMAL);
			imshow("Final Result", dst);
			imwrite("WatershedBad.jpg", dst);
		}
		// My canny 
		if (key == 't') {
			// Filtre 
			Mat filtered;
			cvtColor(src, src_gray, CV_BGR2GRAY);
			bilateralFilter(src_gray, filtered, 15, 2000, 2000);
			//GaussianBlur(src_gray, filtered, Size(9, 9), 2);

			namedWindow("Apres filtre", WINDOW_NORMAL);
			imshow("Apres filtre", filtered);

			// Gradient 
			Mat grad;
			Mat element = Mat::ones(Size(3, 3), CV_8U);
			morphologyEx(filtered, grad, MORPH_GRADIENT, element);


			namedWindow("grad", WINDOW_NORMAL);
			imshow("grad", grad);

			// Maxima locaux 
			Mat maxima = grad.clone(), mask;
			morphologyEx(maxima, maxima, MORPH_DILATE, element);
			compare(grad, maxima, mask, CMP_EQ);
			divide(255, mask, mask);
			maxima = grad.mul(mask);
			equalizeHist(maxima, maxima);

			// Double seuillage 
			int seuil_bas = 125, seuil_haut = 2 * seuil_bas;
			Mat point_faible, point_fort, double_seuil;
			threshold(maxima, point_faible, seuil_bas, 255, CV_THRESH_BINARY);
			threshold(maxima, point_fort, seuil_haut, 255, CV_THRESH_BINARY);
			point_faible = point_faible - point_fort;
			double_seuil = point_fort.clone();

			/*
			namedWindow("Haut", WINDOW_NORMAL);
			imshow("Haut", point_fort);
			namedWindow("Bas", WINDOW_NORMAL);
			imshow("Bas", point_faible);
			*/

			// Hysteresis
			Mat corner, fort_dilate;
			morphologyEx(point_fort, fort_dilate, MORPH_DILATE, element);
			//compare(point_faible, fort_dilate, corner, CMP_EQ);
			bitwise_and(fort_dilate, point_faible, corner);
			bitwise_or(corner, point_fort, corner);

			// Affichage resultat 
			namedWindow("My canny", WINDOW_NORMAL);
			imshow("My canny", corner);
			imwrite("mycanny.jpg", corner);

			src_gray = src.clone();
			vector<Vec2f> lines;
			HoughLines(corner, lines, 1, CV_PI / 180, alpha_sliderTH);
			for (size_t i = 0; i < lines.size(); i++)
			{
				float rho = lines[i][0];
				float theta = lines[i][1];
				double a = cos(theta), b = sin(theta);
				double x0 = a * rho, y0 = b * rho;
				Point pt1(cvRound(x0 + 10000 * (-b)),
					cvRound(y0 + 10000 * (a)));
				Point pt2(cvRound(x0 - 10000 * (-b)),
					cvRound(y0 - 10000 * (a)));
				line(src_gray, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
			}

			namedWindow("Res hough", WINDOW_NORMAL);
			imshow("Res hough", src_gray);
		}
		// Seuilage kmean
		if (key == 'k') {
			src = imread(images_color[current_image]);
			//cvtColor(src, src, COLOR_BGR2GRAY);
			//cvtColor(src, src, COLOR_GRAY2BGR);
			//morphologyEx(src, src, MORPH_ERODE, getStructuringElement(CV_SHAPE_RECT, Size(alpha_sliderFS, alpha_sliderFS)));
			//src = imread("image_cut.jpg");
			// Initialisation K
			int K = alpha_sliderK;
			// Mise en forme de ligne de l'image
			int tailleLigne = src.rows * src.cols;
			Mat src_ligne = src.reshape(1, tailleLigne);
			src_ligne.convertTo(src_ligne, CV_32F);
			// K means 
			vector<int> labels;
			Mat1f colors;
			kmeans(src_ligne, K, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.), 2, KMEANS_PP_CENTERS, colors);
			for (unsigned int i = 0; i < tailleLigne; i++) {
				src_ligne.at<float>(i, 0) = colors(labels[i], 0);
				src_ligne.at<float>(i, 1) = colors(labels[i], 1);
				src_ligne.at<float>(i, 2) = colors(labels[i], 2);
			}
			// Remise en forme image
			Mat outputImage = src_ligne.reshape(3, src.rows);
			outputImage.convertTo(outputImage, CV_8U);
			// Affichage
			namedWindow("kmeans", WINDOW_NORMAL);
			imshow("kmeans", outputImage);
			imwrite("kmean9.jpg", outputImage);
			/*
			// Create a kernel that we will use to sharpen our image
			Mat kernel = (Mat_<float>(3, 3) <<
				1, 1, 1,
				1, -8, 1,
				1, 1, 1);
			Mat imgLaplacian;
			filter2D(outputImage, imgLaplacian, CV_32F, kernel);
			Mat sharp;
			outputImage.convertTo(sharp, CV_32F);
			Mat imgResult = sharp - imgLaplacian;
			// convert back to 8bits gray scale
			imgResult.convertTo(imgResult, CV_8UC3);
			imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
			namedWindow("Laplace Filtered Image", WINDOW_NORMAL);
			imshow("Laplace Filtered Image", imgLaplacian);
			namedWindow("New Sharped Image", WINDOW_NORMAL);
			imshow("New Sharped Image", imgResult);
			// Create binary image from source image
			Mat bw;
			cvtColor(imgResult, bw, COLOR_BGR2GRAY);
			threshold(bw, bw, 40, 255, THRESH_BINARY | THRESH_OTSU);
			// Supression du bruit
			namedWindow("Binary Image", WINDOW_NORMAL);
			GaussianBlur(bw, bw, Size(5, 5), 2);
			Mat element = Mat::ones(Size(3, 3), CV_8U);
			morphologyEx(bw, bw, MORPH_CLOSE, element);
			imshow("Binary Image", bw);
			// Perform the distance transform algorithm
			Mat dist;
			distanceTransform(bw, dist, DIST_L2, 3);
			// Normalize the distance image for range = {0.0, 1.0}
			// so we can visualize and threshold it
			normalize(dist, dist, 0, 1.0, NORM_MINMAX);
			namedWindow("Distance Transform Image", WINDOW_NORMAL);
			imshow("Distance Transform Image", dist);
			// Threshold to obtain the peaks
			// This will be the markers for the foreground objects
			threshold(dist, dist, 0.4, 1.0, THRESH_BINARY);
			// Dilate a bit the dist image
			Mat kernel1 = Mat::ones(3, 3, CV_8U);
			dilate(dist, dist, kernel1);
			namedWindow("Peaks", WINDOW_NORMAL);
			imshow("Peaks", dist);
			// Create the CV_8U version of the distance image
			// It is needed for findContours()
			Mat dist_8u;
			dist.convertTo(dist_8u, CV_8U);
			// Find total markers
			vector<vector<Point> > contours;
			findContours(dist_8u, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
			// Create the marker image for the watershed algorithm
			Mat markers = Mat::zeros(dist.size(), CV_32S);
			// Draw the foreground markers
			for (size_t i = 0; i < contours.size(); i++)
			{
				drawContours(markers, contours, static_cast<int>(i), Scalar(static_cast<int>(i) + 1), -1);
			}
			// Draw the background marker
			circle(markers, Point(5, 5), 3, Scalar(255), -1);
			namedWindow("Markers", WINDOW_NORMAL);
			imshow("Markers", markers * 10000);
			// Perform the watershed algorithm
			watershed(imgResult, markers);
			Mat mark;
			markers.convertTo(mark, CV_8U);
			bitwise_not(mark, mark);
			//    imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
			// image looks like at that point
			// Generate random colors
			vector<Vec3b> colors2;
			for (size_t i = 0; i < contours.size(); i++)
			{
				int b = theRNG().uniform(0, 256);
				int g = theRNG().uniform(0, 256);
				int r = theRNG().uniform(0, 256);
				colors2.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
			}
			// Create the result image
			Mat dst = Mat::zeros(markers.size(), CV_8UC3);
			// Fill labeled objects with random colors
			for (int i = 0; i < markers.rows; i++)
			{
				for (int j = 0; j < markers.cols; j++)
				{
					int index = markers.at<int>(i, j);
					if (index > 0 && index <= static_cast<int>(contours.size()))
					{
						dst.at<Vec3b>(i, j) = colors2[index - 1];
					}
				}
			}
			// Visualize the final image
			namedWindow("Final Result", WINDOW_NORMAL);
			imshow("Final Result", dst);
			imwrite("kmean.jpg", dst);

			Mat mask;
			cvtColor(dst, mask, CV_BGR2GRAY);
			threshold(mask, mask, 10, 255, THRESH_BINARY);
			cvtColor(mask, mask, CV_GRAY2BGR);
			bitwise_and(src, mask, dst);

			// Visualize the final image
			namedWindow("Final Result after mask", WINDOW_NORMAL);
			imshow("Final Result after mask", dst);*/

		}
		// Unseeded Region growth 
		if (key == 'u') {
			// On recupere l'image a traiter
			src = imread("Paquet 3/planks-color/plank-color-1/Arbre 1/p12.png");

			namedWindow("src", WINDOW_NORMAL);
			//imshow("src", src);
			// On creer l'image correspondant aux classes 
			Mat classes = Mat::zeros(src.size(), CV_64FC1);
			// On cree la structure de stockage des couleurs < cluster_id, <nb_elem, sumB, sumG, sumR>>
			std::unordered_map<int, vector<int>> clusters;
			// On note le dernier cluster creer 
			int last_cluster = 1;

			// On creer nos elements structurant de descente et de monte 
			Mat elem_descente = (Mat_<int>(3, 3) << 1, 1, 1, 1, 0, 0, 0, 0, 0);
			Mat elem_monte = (Mat_<int>(3, 3) << 0, 0, 0, 0, 0, 1, 1, 1, 1);

			// Initialisation taille de l'image 
			int w = src.size().width - 1;
			int h = src.size().height - 1;
			// Initialisation des tailles du masque
			int n = elem_descente.size().height; 		// Taille du masque
			int n2 = (n - 1) / 2; 				// Taille demi masque 

			// Premer parcours de l'image en descendant 
			for (int x = 0; x < h; x++) {
				for (int y = 0; y < w; y++) {
					//     cout << 1 << endl;
					int nb = n * n;
					int res = last_cluster;
					for (int i = 0; i < n; i++) {
						for (int j = 0; j < n; j++) {
							// On ignore les bord 
							if (x + i - n2 < 0 || x + i - n2 >= h || y + j - n2 < 0 || y + j - n2 >= w) {
								nb--;
							}
							else {
								//cout << n << " " << i << " " << j << endl;
								// Si on est dans l'element alors on regarde le cluster 
								if (elem_descente.at<int>(i, j) == 1) {
									//              cout << 2 << endl;
									double cluster = classes.at<double>((x + i - n2), (y + j - n2));
									//               cout << 3 << ", cluster = " << cluster << ", Clusters size = " << last_cluster << endl;
									int distance = sqrt(
										pow(src.at<cv::Vec3b>((x + i - n2), (y + j - n2))[0] - clusters[cluster][1] / clusters[cluster][0], 2) +
										pow(src.at<cv::Vec3b>((x + i - n2), (y + j - n2))[1] - clusters[cluster][2] / clusters[cluster][0], 2) +
										pow(src.at<cv::Vec3b>((x + i - n2), (y + j - n2))[2] - clusters[cluster][3] / clusters[cluster][0], 2)
										);
									//             cout << 4 << endl;
									if (distance < alpha_sliderURG) {
										res = cluster;
									}
									//                 cout << 5 << endl;
								}

							}
						}
					}
					//       cout << 6 << endl;
					if (nb > 0) {
						if (res == last_cluster) {
							//         cout << 7.1 << endl;
									 // On creer un nouveau cluster
							vector<int> v;
							v.push_back(1);
							v.push_back(src.at<cv::Vec3b>(x, y)[0]);
							v.push_back(src.at<cv::Vec3b>(x, y)[1]);
							v.push_back(src.at<cv::Vec3b>(x, y)[2]);
							clusters[last_cluster] = v;
							//        cout << 8.1 << endl;
							classes.at<double>(x, y) = last_cluster;
							last_cluster++;
						}
						else {
							//         cout << 7.2 << endl;
									 // On l'ajoute au cluster
							clusters[res][0]++;
							clusters[res][1] += src.at<cv::Vec3b>(x, y)[0];
							clusters[res][2] += src.at<cv::Vec3b>(x, y)[1];
							clusters[res][3] += src.at<cv::Vec3b>(x, y)[2];

							//         cout << 8.2 << endl;
							classes.at<double>(x, y) = res;
						}
						//      cout << 9 << endl;

					}
				}
			}
			// Second parcours de l'image en montant
			for (int x = h - 1; x >= 0; x--) {
				for (int y = w - 1; y >= 0; y--) {
					int nb = n * n;
					int res = last_cluster;
					double min_distance = alpha_slider_max;
					for (int i = 0; i < n; i++) {
						for (int j = 0; j < n; j++) {
							// On ignore les bord 
							if (x + i - n2 < 0 || x + i - n2 >= h || y + j - n2 < 0 || y + j - n2 >= w) {
								nb--;
							}
							else {
								//cout << n << " " << i << " " << j << endl;
								// Si on est dans l'element alors on regarde le cluster 
								if (elem_monte.at<int>(i, j) == 1) {
									double cluster = classes.at<double>((x + i - n2), (y + j - n2));
									double distance = sqrt(
										pow(src.at<cv::Vec3b>((x + i - n2), (y + j - n2))[0] - clusters[cluster][1] / clusters[cluster][0], 2) +
										pow(src.at<cv::Vec3b>((x + i - n2), (y + j - n2))[1] - clusters[cluster][2] / clusters[cluster][0], 2) +
										pow(src.at<cv::Vec3b>((x + i - n2), (y + j - n2))[2] - clusters[cluster][3] / clusters[cluster][0], 2)
										);
									if (distance < alpha_sliderURG && distance < min_distance) {
										res = cluster;
										min_distance = distance;
									}
								}

							}
						}
					}
					if (res == last_cluster) {
						// On creen un nouveau cluster
						vector<int> v;
						v.push_back(1);
						v.push_back(src.at<cv::Vec3b>(x, y)[0]);
						v.push_back(src.at<cv::Vec3b>(x, y)[1]);
						v.push_back(src.at<cv::Vec3b>(x, y)[2]);
						clusters[last_cluster] = v;
						last_cluster++;
					}
					else {
						double cluster = classes.at<double>(x, y);
						// On le retire de son ancien cluster
						clusters[res][0]--;
						clusters[res][1] -= src.at<cv::Vec3b>(x, y)[0];
						clusters[res][2] -= src.at<cv::Vec3b>(x, y)[1];
						clusters[res][3] -= src.at<cv::Vec3b>(x, y)[2];
						// On l'ajoute au nouveau cluster
						clusters[res][0]++;
						clusters[res][1] += src.at<cv::Vec3b>(x, y)[0];
						clusters[res][2] += src.at<cv::Vec3b>(x, y)[1];
						clusters[res][3] += src.at<cv::Vec3b>(x, y)[2];
					}
					classes.at<double>(x, y) = res;
				}
			}
			// On recreer la nouvelle image
			for (int x = 0; x < h; x++) {
				for (int y = 0; y < w; y++) {
					double cluster = classes.at<double>(x, y);
					src.at<cv::Vec3b>(x, y)[0] = clusters[cluster][1] / clusters[cluster][0];
					src.at<cv::Vec3b>(x, y)[1] = clusters[cluster][2] / clusters[cluster][0];
					src.at<cv::Vec3b>(x, y)[2] = clusters[cluster][3] / clusters[cluster][0];
				}
			}
			cout << clusters.size() << endl;

			namedWindow("Unseeded region growth", WINDOW_NORMAL);
			imshow("Unseeded region growth", src);
			imwrite("Unseeded region growth.jpg", src);
		}
		// Bounding Box
		if (key == 'e') {
			src = imread("kmean.jpg");

			cvtColor(src, src_gray, COLOR_BGR2GRAY);
			blur(src_gray, src_gray, Size(3, 3));

			Canny(src_gray, canny_output, alpha_sliderT1, alpha_sliderT2);

			vector<vector<Point> > contours;
			findContours(canny_output, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

			vector<vector<Point> > contours_poly(contours.size());
			vector<Rect> boundRect(contours.size());
			vector<Point2f>centers(contours.size());
			vector<float>radius(contours.size());
			for (size_t i = 0; i < contours.size(); i++)
			{
				approxPolyDP(contours[i], contours_poly[i], 3, true);
				boundRect[i] = boundingRect(contours_poly[i]);
				minEnclosingCircle(contours_poly[i], centers[i], radius[i]);
			}

			Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);

			int min_w = 300, max_w = 600, min_h = 100, max_h = 250;
			double avg_x = 0, avg_y = 0;
			int nb = 0;
			for (size_t i = 0; i < contours.size(); i++)
			{
				Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
				//drawContours(drawing, contours_poly, (int)i, color);
				int taille_x = sqrt((boundRect[i].tl().x - boundRect[i].br().x) * (boundRect[i].tl().x - boundRect[i].br().x));
				int taille_y = sqrt((boundRect[i].tl().y - boundRect[i].br().y) * (boundRect[i].tl().y - boundRect[i].br().y));
				//if (320 < taille_x && taille_x < 480 && 140 < taille_y && taille_y < 200) {  // Extraction 1 
				//if (taille_x / taille_y > 1.9 && taille_x / taille_y < 3.1 && 300 < taille_x && taille_x < 600 && (100 < taille_y && taille_y < 250) { // Extraction 2
				if (taille_x / taille_y > 1.9 && taille_x / taille_y < 3.3 && ((min_w < taille_x && taille_x < max_w) || (min_h < taille_y && taille_y < min_h))) { // Extraction 3
					avg_x += taille_x;
					avg_y += taille_y;
					nb++;
				}
				//circle(drawing, centers[i], (int)radius[i], color, 2);
			}
			avg_x = avg_x / nb;
			avg_y = avg_y / nb;
			int num_planche = 0;
			src = imread(images_number[current_image]);
			for (size_t i = 0; i < contours.size(); i++)
			{
				Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
				//drawContours(drawing, contours_poly, (int)i, color);
				double taille_x = sqrt((boundRect[i].tl().x - boundRect[i].br().x) * (boundRect[i].tl().x - boundRect[i].br().x));
				double taille_y = sqrt((boundRect[i].tl().y - boundRect[i].br().y) * (boundRect[i].tl().y - boundRect[i].br().y));
				//if (320 < taille_x && taille_x < 480 && 140 < taille_y && taille_y < 200) {
				if (taille_x / taille_y > 1.9 && taille_x / taille_y < 3.3 && ((min_w < taille_x && taille_x < max_w) || (min_h < taille_y && taille_y < min_h))) { // Extraction 3
					cout << taille_x / taille_y << endl;
					rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, -3);
					for (int a = 0; a < 8; a++) {
						rectangle(drawing, Point(boundRect[i].tl().x + a * avg_x, boundRect[i].tl().y), Point(boundRect[i].br().x + a * avg_x, boundRect[i].br().y), color, 3);
						rectangle(drawing, Point(boundRect[i].tl().x + a * -1 * avg_x, boundRect[i].tl().y), Point(boundRect[i].br().x + a * -1 * avg_x, boundRect[i].br().y), color, 3);
						if (boundRect[i].tl().x + a * avg_x > 0 && boundRect[i].tl().x + a * avg_x < src.size().width && boundRect[i].br().x + a * avg_x > 0 && boundRect[i].br().x + a * avg_x < src.size().width) {
							Rect roi(Point(boundRect[i].tl().x + a * avg_x, boundRect[i].tl().y), Point(boundRect[i].br().x + a * avg_x, boundRect[i].br().y));
							Mat planche = src(roi);
							imwrite("planches/la planche" + to_string(num_planche++) + ".jpg", planche);
						}
						if (boundRect[i].tl().x + a * -1 * avg_x > 0 && boundRect[i].tl().x + a * -1 * avg_x < src.size().width && boundRect[i].br().x + a * -1 * avg_x > 0 && boundRect[i].br().x + a * -1 * avg_x < src.size().width) {
							Rect roi(Point(boundRect[i].tl().x + a * -1 * avg_x, boundRect[i].tl().y), Point(boundRect[i].br().x + a * -1 * avg_x, boundRect[i].br().y));
							Mat planche = src(roi);
							imwrite("planches/la planche" + to_string(num_planche++) + ".jpg", planche);
						}
					}

				}
				else {
					//rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2);
				}
				cout << "Taille x = " << taille_x << " , Taille y = " << taille_y << endl;
				//circle(drawing, centers[i], (int)radius[i], color, 2);
			}


			namedWindow("Contours", WINDOW_NORMAL);
			imshow("Contours", drawing);
			imwrite("bbox.jpg", drawing);


			Mat mask;
			src = imread(images_number[current_image]);
			cvtColor(drawing, mask, CV_BGR2GRAY);
			threshold(mask, mask, 10, 255, THRESH_BINARY);
			cvtColor(mask, mask, CV_GRAY2BGR);
			bitwise_and(src, mask, drawing);

			// Visualize the final image
			namedWindow("Final Result after mask", WINDOW_NORMAL);
			imshow("Final Result after mask", drawing);
		}
		if (key == 'd') {
			src = imread("kmean.jpg");

			cvtColor(src, src_gray, COLOR_BGR2GRAY);
			blur(src_gray, src_gray, Size(3, 3));

			Canny(src_gray, canny_output, alpha_sliderT1, alpha_sliderT2);

			vector<vector<Point> > contours;
			findContours(canny_output, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

			vector<vector<Point> > contours_poly(contours.size());
			vector<Rect> boundRect(contours.size());
			vector<Point2f>centers(contours.size());
			vector<float>radius(contours.size());
			for (size_t i = 0; i < contours.size(); i++)
			{
				approxPolyDP(contours[i], contours_poly[i], 3, true);
				boundRect[i] = boundingRect(contours_poly[i]);
				minEnclosingCircle(contours_poly[i], centers[i], radius[i]);
			}

			Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);

			int min_w = 300, max_w = 450, min_h = 100, max_h = 250;
			double avg_x = 0, avg_y = 0;
			int nb = 0;
			for (size_t i = 0; i < contours.size(); i++)
			{
				Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
				//drawContours(drawing, contours_poly, (int)i, color);
				int taille_x = sqrt((boundRect[i].tl().x - boundRect[i].br().x) * (boundRect[i].tl().x - boundRect[i].br().x));
				int taille_y = sqrt((boundRect[i].tl().y - boundRect[i].br().y) * (boundRect[i].tl().y - boundRect[i].br().y));
				//if (320 < taille_x && taille_x < 480 && 140 < taille_y && taille_y < 200) {  // Extraction 1 
				//if (taille_x / taille_y > 1.9 && taille_x / taille_y < 3.1 && 300 < taille_x && taille_x < 600 && (100 < taille_y && taille_y < 250) { // Extraction 2
				if (taille_x / taille_y > 1.9 && taille_x / taille_y < 3.3 && ((min_w < taille_x && taille_x < max_w) || (min_h < taille_y && taille_y < min_h))) { // Extraction 3
					avg_x += taille_x;
					avg_y += taille_y;
					nb++;
				}
				//circle(drawing, centers[i], (int)radius[i], color, 2);
			}
			avg_x = avg_x / nb;
			avg_y = avg_y / nb;
			int num_planche = 0;
			src = imread(images_number[current_image]);
			for (size_t i = 0; i < contours.size(); i++)
			{
				Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
				//drawContours(drawing, contours_poly, (int)i, color);
				double taille_x = sqrt((boundRect[i].tl().x - boundRect[i].br().x) * (boundRect[i].tl().x - boundRect[i].br().x));
				double taille_y = sqrt((boundRect[i].tl().y - boundRect[i].br().y) * (boundRect[i].tl().y - boundRect[i].br().y));
				//if (320 < taille_x && taille_x < 480 && 140 < taille_y && taille_y < 200) {
				if (taille_x / taille_y > 1.9 && taille_x / taille_y < 3.3 && ((min_w < taille_x && taille_x < max_w) || (min_h < taille_y && taille_y < min_h))) { // Extraction 3
					cout << taille_x / taille_y << endl;
					rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, -3); //////////// ---------------------------------- PASSAGE EN BOITE VIDE 
					for (int a = 0; a < 8; a++) {
						//rectangle(drawing, Point(boundRect[i].tl().x + a * avg_x, boundRect[i].tl().y), Point(boundRect[i].br().x + a * avg_x, boundRect[i].br().y), color, 3);
						//rectangle(drawing, Point(boundRect[i].tl().x + a * -1 * avg_x, boundRect[i].tl().y), Point(boundRect[i].br().x + a * -1 * avg_x, boundRect[i].br().y), color, 3);
						if (boundRect[i].tl().x + a * avg_x > 0 && boundRect[i].tl().x + a * avg_x < src.size().width && boundRect[i].br().x + a * avg_x > 0 && boundRect[i].br().x + a * avg_x < src.size().width) {
							Rect roi(Point(boundRect[i].tl().x + a * avg_x, boundRect[i].tl().y), Point(boundRect[i].br().x + a * avg_x, boundRect[i].br().y));
							Mat planche = src(roi);
							imwrite("planches/la planche" + to_string(num_planche++) + ".jpg", planche);
						}
						if (boundRect[i].tl().x + a * -1 * avg_x > 0 && boundRect[i].tl().x + a * -1 * avg_x < src.size().width && boundRect[i].br().x + a * -1 * avg_x > 0 && boundRect[i].br().x + a * -1 * avg_x < src.size().width) {
							Rect roi(Point(boundRect[i].tl().x + a * -1 * avg_x, boundRect[i].tl().y), Point(boundRect[i].br().x + a * -1 * avg_x, boundRect[i].br().y));
							Mat planche = src(roi);
							imwrite("planches/la planche" + to_string(num_planche++) + ".jpg", planche);
						}
					}

				}
				else {
					//rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2);
				}
				cout << "Taille x = " << taille_x << " , Taille y = " << taille_y << endl;
				//circle(drawing, centers[i], (int)radius[i], color, 2);
			}


			namedWindow("Contours", WINDOW_NORMAL);
			imshow("Contours", drawing);
			imwrite("bbox.jpg", drawing);


			Mat mask;
			src = imread(images_number[current_image]);
			cvtColor(drawing, mask, CV_BGR2GRAY);
			threshold(mask, mask, 10, 255, THRESH_BINARY);
			cvtColor(mask, mask, CV_GRAY2BGR);
			bitwise_and(src, mask, drawing);

			// Visualize the final image
			namedWindow("Final Result after mask", WINDOW_NORMAL);
			imshow("Final Result after mask", drawing);
		}
		// Flou deplacement + bbox 
		if (key == 'z') {
			src = imread(images_number[current_image]);
			// ----------------------------------- On recupere les lignes ----------------------------------- //
			// Creation de la matrice contenant les lignes 
			Mat ligne;
			// On recupere l'image source en niveaux de gris 
			cvtColor(src, src_gray, CV_BGR2GRAY);
			// Creation du kernel pour le flou de deplacement lateral
			int size = 500;
			Mat kernel = Mat::zeros(size, size, CV_32FC1);
			kernel.row(int((size - 1) / 2)) = 1;
			kernel = kernel / size;
			// On applique le filtre a notre image en niveau de gris 
			filter2D(src_gray, ligne, -1, kernel);
			// Affichage de l'image transformee
			namedWindow("Apres filtre", WINDOW_NORMAL);
			imshow("Apres filtre", ligne);
			imwrite("lignes.jpg", ligne);
			// Seuillage automatique afin de binariser l'image et avoir les lignes en blanc 
			threshold(ligne, ligne, 1, 255, THRESH_BINARY | THRESH_OTSU);
			// Affichage de l'image seuillee
			namedWindow("Lignes", WINDOW_NORMAL);
			imshow("Lignes", ligne);
			// ---------------------------------------------------------------------------------------------- //

			src = imread("kmean.jpg");

			// ----------------------- On recupere les boites englobantes et imagettes ---------------------- //
			// On passe en image gris puis on reduis le bruti avec un flou 
			cvtColor(src, src_gray, COLOR_BGR2GRAY);
			blur(src_gray, src_gray, Size(3, 3));

			// On recupere les contours avec canny
			Canny(src_gray, canny_output, alpha_sliderT1, alpha_sliderT2);

			// On labelise et regroupe les contours 
			vector<vector<Point> > contours;
			findContours(canny_output, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
			// On les aproxime avec une boite englobante 
			vector<vector<Point> > contours_poly(contours.size());
			vector<Rect> boundRect(contours.size());
			vector<Point2f>centers(contours.size());
			vector<float>radius(contours.size());
			for (size_t i = 0; i < contours.size(); i++)
			{
				approxPolyDP(contours[i], contours_poly[i], 3, true);
				boundRect[i] = boundingRect(contours_poly[i]);
				minEnclosingCircle(contours_poly[i], centers[i], radius[i]);
			}
			// On creer l'image qui vas stocker notre detection de boite 
			Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);

			// On recupere des statistiques sur les boites detectes (lageur hauteur)
			int min_w = 300, max_w = 600, min_h = 100, max_h = 250;
			double avg_x = 0, avg_y = 0;
			int nb = 0;
			for (size_t i = 0; i < contours.size(); i++) {
				int taille_x = sqrt((boundRect[i].tl().x - boundRect[i].br().x) * (boundRect[i].tl().x - boundRect[i].br().x));
				int taille_y = sqrt((boundRect[i].tl().y - boundRect[i].br().y) * (boundRect[i].tl().y - boundRect[i].br().y));
				//if (320 < taille_x && taille_x < 480 && 140 < taille_y && taille_y < 200) {  // Extraction 1 
				//if (taille_x / taille_y > 1.9 && taille_x / taille_y < 3.1 && 300 < taille_x && taille_x < 600 && 100 < taille_y && taille_y < 250) { // Extraction 2
				if (taille_x / taille_y > 1.9 && taille_x / taille_y < 3.3 && ((min_w < taille_x && taille_x < max_w) || (min_h < taille_y && taille_y < min_h))) { // Extraction 3
					avg_x += taille_x;
					avg_y += taille_y;
					nb++;
				}
			}
			avg_x = avg_x / nb;
			avg_y = avg_y / nb;
			// Initialisation des compteur d'enregistrement de planches
			int num_planche = 0;
			int num_planche_voisine = 0;
			// On recharge l'image a traiter 
			src = imread(images_number[current_image]);
			// Pour chaque boite englobante trouvee 
			for (size_t i = 0; i < contours.size(); i++) {
				num_planche_voisine = 0;
				// On recupere une couleur aleatoire 
				Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
				// On recupere la taille de la boite actuel 
				double taille_x = sqrt((boundRect[i].tl().x - boundRect[i].br().x) * (boundRect[i].tl().x - boundRect[i].br().x));
				double taille_y = sqrt((boundRect[i].tl().y - boundRect[i].br().y) * (boundRect[i].tl().y - boundRect[i].br().y));
				// On regarde si on a affaire a une boite qui semble etre une planche dans l'image 
				//if (320 < taille_x && taille_x < 480 && 140 < taille_y && taille_y < 200) {  // Extraction 1 
				//if (taille_x / taille_y > 1.9 && taille_x / taille_y < 3.1 && 300 < taille_x && taille_x < 600 && (100 < taille_y && taille_y < 250) { // Extraction 2
				if (taille_x / taille_y > 1.9 && taille_x / taille_y < 3.3 && ((min_w < taille_x && taille_x < max_w) || (min_h < taille_y && taille_y < min_h))) { // Extraction 3
					Rect roi(boundRect[i].tl(), boundRect[i].br());
					Mat inligne = ligne(roi);
					// On regarde si la boite detectee se trouve bien sur une ligne 
					if (mean(inligne)[0] / 255 > 0.9) {
						rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, -3);
						Mat planche = src(roi);
						imwrite("planches/p" + to_string(++num_planche) + ".jpg", planche);
					}
					// On essaye d'ajouter les voisins a la recuperation de boite 
					for (int a = 1; a < 8; a++) {
						// Voisins a droite
						// On verifie que l'on ne depasse pas les limites de l'image
						if (boundRect[i].tl().x + a * avg_x > 0 && boundRect[i].tl().x + a * avg_x < src.size().width && boundRect[i].br().x + a * avg_x > 0 && boundRect[i].br().x + a * avg_x < src.size().width) {
							// On recupere la region a droite a 'a' pas sur le cote avec la taille moyenne comme decalage 
							Rect roi(Point(boundRect[i].tl().x + a * avg_x, boundRect[i].tl().y), Point(boundRect[i].br().x + a * avg_x, boundRect[i].br().y));
							Mat inligne = ligne(roi);
							// On regarde si on se trouve bien sur une ligne 
							if (mean(inligne)[0] / 255 > 0.9) {
								// On dessine le rectangle 
								rectangle(drawing, Point(boundRect[i].tl().x + a * avg_x, boundRect[i].tl().y), Point(boundRect[i].br().x + a * avg_x, boundRect[i].br().y), color, 3);
								Mat planche = src(roi);
								imwrite("planches/p" + to_string(num_planche) + "_" + to_string(num_planche_voisine++) + ".jpg", planche);
							}
							else {
								// On est pas sur une ligne donc on regarde s'il faut monter ou descendre pour etre sur la ligne 
								Rect roi(Point(boundRect[i].tl().x + a * avg_x, boundRect[i].tl().y + 1), Point(boundRect[i].br().x + a * avg_x, boundRect[i].br().y + 1));
								Mat inligne_dir = ligne(roi);
								// Si ce teste passe on doit decendre sinon on monte 
								if (mean(inligne)[0] / 255 < mean(inligne_dir)[0] / 255) { // Descente 
									// On descent pixel par pixel 
									double max = mean(inligne_dir)[0];
									int b = 2;
									for (b = 2; b < 100; b++) {
										// On stop si on depasse une limite 
										if (boundRect[i].tl().y + b < 0 || boundRect[i].br().y + b > src.size().height) break;
										Rect roi(Point(boundRect[i].tl().x + a * avg_x, boundRect[i].tl().y + b), Point(boundRect[i].br().x + a * avg_x, boundRect[i].br().y + b));
										Mat inligne_correct = ligne(roi);
										// On sauvegarde le max si on preogresse sinon on stop
										if (mean(inligne_correct)[0] >= max) {
											max = mean(inligne_correct)[0];
										}
										else {
											break;
										}
									}
									// On recupere la region la "+ dans la ligne" et on la sauvegarde 
									Rect roi(Point(boundRect[i].tl().x + a * avg_x, boundRect[i].tl().y + b - 1), Point(boundRect[i].br().x + a * avg_x, boundRect[i].br().y + b - 1));
									if (b < 99 || max > 0.9) {
										Mat planche = src(roi);
										rectangle(drawing, roi, color, 3);
										imwrite("planches/p" + to_string(num_planche) + "_" + to_string(num_planche_voisine++) + ".jpg", planche);
									}
								}
								else { // Monte
									// On monte pixel par pixel 
									double max = mean(inligne_dir)[0];
									int b = 1;
									for (b = 1; b < 100; b++) {
										// On stop si on depasse une limite 
										if (boundRect[i].tl().y - b < 0 || boundRect[i].br().y - b > src.size().height) break;
										Rect roi(Point(boundRect[i].tl().x + a * avg_x, boundRect[i].tl().y - b), Point(boundRect[i].br().x + a * avg_x, boundRect[i].br().y - b));
										Mat inligne_correct = ligne(roi);
										// On sauvegarde le max si on preogresse sinon on stop
										if (mean(inligne_correct)[0] >= max) {
											max = mean(inligne_correct)[0];
										}
										else {
											break;
										}
									}
									// On recupere la region la "+ dans la ligne" et on la sauvegarde 
									Rect roi(Point(boundRect[i].tl().x + a * avg_x, boundRect[i].tl().y - b + 1), Point(boundRect[i].br().x + a * avg_x, boundRect[i].br().y - b + 1));
									if (b < 99 || max > 0.9) {
										Mat planche = src(roi);
										rectangle(drawing, roi, color, 3);
										imwrite("planches/p" + to_string(num_planche) + "_" + to_string(num_planche_voisine++) + ".jpg", planche);
									}
								}
							}
						}
						// Voisins a gauche
						// On verifie que l'on ne depasse pas les limites de l'image
						if (boundRect[i].tl().x + a * -1 * avg_x > 0 && boundRect[i].tl().x + a * -1 * avg_x < src.size().width && boundRect[i].br().x + a * -1 * avg_x > 0 && boundRect[i].br().x + a * -1 * avg_x < src.size().width) {
							// On recupere la region a gauche a 'a' pas sur le cote avec la taille moyenne comme decalage 
							Rect roi(Point(boundRect[i].tl().x + a * -1 * avg_x, boundRect[i].tl().y), Point(boundRect[i].br().x + a * -1 * avg_x, boundRect[i].br().y));
							Mat inligne = ligne(roi);
							// On regarde si on se trouve bien sur une ligne 
							if (mean(inligne)[0] / 255 > 0.9) {
								// On dessine le rectangle 
								rectangle(drawing, Point(boundRect[i].tl().x + a * -1 * avg_x, boundRect[i].tl().y), Point(boundRect[i].br().x + a * -1 * avg_x, boundRect[i].br().y), color, 3);
								Mat planche = src(roi);
								imwrite("planches/p" + to_string(num_planche) + "_" + to_string(num_planche_voisine++) + ".jpg", planche);
							}
							else {
								// On est pas sur une ligne donc on regarde s'il faut monter ou descendre pour etre sur la ligne 
								Rect roi(Point(boundRect[i].tl().x + a * -1 * avg_x, boundRect[i].tl().y + 1), Point(boundRect[i].br().x + a * -1 * avg_x, boundRect[i].br().y + 1));
								Mat inligne_dir = ligne(roi);
								// Si ce teste passe on doit decendre sinon on monte 
								if (mean(inligne)[0] / 255 < mean(inligne_dir)[0] / 255) { // Descente
									// On descent pixel par pixel 
									double max = mean(inligne_dir)[0];
									int b = 2;
									for (b = 2; b < 100; b++) {
										// On stop si on depasse une limite 
										if (boundRect[i].tl().y + b < 0 || boundRect[i].br().y + b > src.size().height) break;
										Rect roi(Point(boundRect[i].tl().x + a * -1 * avg_x, boundRect[i].tl().y + b), Point(boundRect[i].br().x + a * -1 * avg_x, boundRect[i].br().y + b));
										Mat inligne_correct = ligne(roi);
										// On sauvegarde le max si on preogresse sinon on stop
										if (mean(inligne_correct)[0] >= max) {
											max = mean(inligne_correct)[0];
										}
										else {
											break;
										}
									}
									// On recupere la region la "+ dans la ligne" et on la sauvegarde 
									Rect roi(Point(boundRect[i].tl().x + a * -1 * avg_x, boundRect[i].tl().y + b - 1), Point(boundRect[i].br().x + a * -1 * avg_x, boundRect[i].br().y + b - 1));
									if (b < 99 || max > 0.9) {
										Mat planche = src(roi);
										rectangle(drawing, roi, color, 3);
										imwrite("planches/p" + to_string(num_planche) + "_" + to_string(num_planche_voisine++) + ".jpg", planche);
									}
								}
								else { // Monte
									// On monte pixel par pixel 
									double max = mean(inligne_dir)[0];
									int b = 1;
									for (b = 1; b < 100; b++) {
										// On stop si on depasse une limite 
										if (boundRect[i].tl().y - b < 0 || boundRect[i].br().y - b > src.size().height) break;
										Rect roi(Point(boundRect[i].tl().x + a * -1 * avg_x, boundRect[i].tl().y - b), Point(boundRect[i].br().x + a * -1 * avg_x, boundRect[i].br().y - b));
										Mat inligne_correct = ligne(roi);
										// On sauvegarde le max si on preogresse sinon on stop
										if (mean(inligne_correct)[0] >= max) {
											max = mean(inligne_correct)[0];
										}
										else {
											break;
										}
									}
									// On recupere la region la "+ dans la ligne" et on la sauvegarde 
									Rect roi(Point(boundRect[i].tl().x + a * -1 * avg_x, boundRect[i].tl().y - b + 1), Point(boundRect[i].br().x + a * -1 * avg_x, boundRect[i].br().y - b + 1));
									if (b < 99 || max > 0.9) {
										Mat planche = src(roi);
										rectangle(drawing, roi, color, 3);
										imwrite("planches/p" + to_string(num_planche) + "_" + to_string(num_planche_voisine++) + ".jpg", planche);
									}
								}
							}
						}
					}

				}
			}
			// ---------------------------------------------------------------------------------------------- //
			namedWindow("Contours", WINDOW_NORMAL);
			imshow("Contours", drawing);
			imwrite("bbox.jpg", drawing);


			Mat mask;
			src = imread(images_number[current_image]);
			cvtColor(drawing, mask, CV_BGR2GRAY);
			threshold(mask, mask, 10, 255, THRESH_BINARY);
			cvtColor(mask, mask, CV_GRAY2BGR);
			bitwise_and(src, mask, drawing);

			// Visualize the final image
			namedWindow("Final Result after mask", WINDOW_NORMAL);
			imshow("Final Result after mask", drawing);

		}
		// Test bbox contour
		if (key == 'a') {
			Mat canny = imread("CannyClean.jpg");

			int min_w = 300, max_w = 450, min_h = 95, max_h = 150;
			Mat vertical_sum, horizontal_sum;
			Mat horizontal_kernel = Mat::zeros(Size(max_w, max_h), CV_32FC1);
			for (int i = 0; i < max_h; i++)
				if (i < (max_h - min_h) / 2 || i > min_h + (max_h - min_h) / 2)
					horizontal_kernel.row(i) = 1;
			horizontal_kernel = horizontal_kernel / (max_w * (2 * (max_h - min_h)));
			Mat verical_kernel = Mat::zeros(Size(max_w, max_h), CV_32FC1);
			for (int i = 0; i < max_w; i++)
				if (i < (max_w - min_w) / 2 || i > min_w + (max_w - min_w) / 2)
					verical_kernel.col(i) = 1;
			verical_kernel = verical_kernel / (min_w * (2 * (max_w - min_w)));
			filter2D(canny, vertical_sum, CV_8UC1, horizontal_kernel);
			filter2D(canny, horizontal_sum, CV_8UC1, verical_kernel);

			normalize(vertical_sum, vertical_sum, 0, 255, NORM_MINMAX);
			normalize(horizontal_sum, horizontal_sum, 0, 255, NORM_MINMAX);

			namedWindow("big Image", WINDOW_NORMAL);
			imshow("big Image", vertical_sum);
			namedWindow("small Image", WINDOW_NORMAL);
			imshow("small Image", horizontal_sum);

			Mat res = vertical_sum + horizontal_sum;
			normalize(res, res, 0, 255, NORM_MINMAX);
			cvtColor(res, res, COLOR_BGR2GRAY);
			//threshold(res, res, 150, 255, THRESH_BINARY);

			namedWindow("marker Image", WINDOW_NORMAL);
			imshow("marker Image", res);
			imwrite("zztest.jpg", res);

			GaussianBlur(res, canny_output, Size(25, 25), 10, 2);

			namedWindow("cannyzz", WINDOW_NORMAL);
			imshow("cannyzz", canny_output);

		}
		// Recuperation couleur billon 
		if (key == 'w') {

			for (int i = 0; i < images_billon.size(); i++)
			{
				src = imread(images_billon[i]);

				namedWindow("Billon" + to_string(i), WINDOW_NORMAL);
				imshow("Billon" + to_string(i), src);

				// Initialisation K
				int K = alpha_sliderK;
				// Mise en forme de ligne de l'image
				int tailleLigne = src.rows * src.cols;
				Mat src_ligne = src.reshape(1, tailleLigne);
				src_ligne.convertTo(src_ligne, CV_32F);
				// K means 
				vector<int> labels;
				Mat1f colors;
				kmeans(src_ligne, K, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.), 2, KMEANS_PP_CENTERS, colors);
				cout << "couleurs de l'image " << i << "\n" << colors << endl;

				int taille_carre = 200;
				for (int j = 0; j < K; j++)
				{
					rectangle(src, Point(j * taille_carre, src.size().height - taille_carre), Point((j + 1) * taille_carre, src.size().height), Scalar(colors[j][0], colors[j][1], colors[j][2]), -1);

				}

				namedWindow("Billon" + to_string(i), WINDOW_NORMAL);
				imshow("Billon" + to_string(i), src);
				imwrite("BillonsColor/" + to_string(K) + "/Billon" + to_string(i) + ".jpg", src);
			}
		}
		// Association billon planches
		if (key == 'x') {
			// k3
			//vector<Scalar> billon_color = { Scalar(49, 53, 37), Scalar(217, 196, 182) ,Scalar(91, 145, 106) };
			// k4
			vector<Scalar> billon_color = { Scalar(76, 97, 61), Scalar(218, 196, 182), Scalar(95, 153, 114), Scalar(35, 35, 29) };
			// k5
			//vector<Scalar> billon_color = { Scalar(77, 96, 61), Scalar(229, 205, 190), Scalar(84, 155, 112), Scalar(35, 35, 29), Scalar(165, 150, 135) };
			// k6
			//vector<Scalar> billon_color = { Scalar(74, 118, 82), Scalar(229, 205, 190), Scalar(88, 164, 118), Scalar(30, 29, 23), Scalar(168, 148, 134), Scalar(70, 71, 44) };

			// Initialisation K
			int K = alpha_sliderK;
			// Mise en forme de ligne de l'image
			int tailleLigne = src.rows * src.cols;
			Mat src_ligne = src.reshape(1, tailleLigne);
			src_ligne.convertTo(src_ligne, CV_32F);
			// K means 
			vector<int> labels;
			Mat1f colors;
			kmeans(src_ligne, K, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.), 2, KMEANS_PP_CENTERS, colors);

			for (int j = 0; j < billon_color.size(); j++) {
				double min = 500;
				int imin = 0;
				for (int i = 0; i < K; i++) {
					double d = sqrt(pow(colors[i][0] - billon_color[j][0], 2) + pow(colors[i][1] - billon_color[j][1], 2) + pow(colors[i][2] - billon_color[j][2], 2));
					if (d < min) {
						min = d;
						imin = i;
					}
				}
				for (unsigned int l = 0; l < tailleLigne; l++) {
					if (labels[l] == imin) {
						src_ligne.at<float>(l, 0) = colors(labels[l], 0);
						src_ligne.at<float>(l, 1) = colors(labels[l], 1);
						src_ligne.at<float>(l, 2) = colors(labels[l], 2);
					}
					else {
						src_ligne.at<float>(l, 0) = 0;
						src_ligne.at<float>(l, 1) = 0;
						src_ligne.at<float>(l, 2) = 0;
					}
				}
				// Remise en forme image
				Mat outputImage = src_ligne.reshape(3, src.rows);
				outputImage.convertTo(outputImage, CV_8U);
				// Affichage
				namedWindow("kmeans", WINDOW_NORMAL);
				imshow("kmeans", outputImage);
				imwrite("kmean_color" + to_string(j) + ".jpg", outputImage);
			}
		}
		// test kmean autres espace de couleur 
		if (key == 'c') {
			src = imread(images_number[current_image]);
			char key2 = waitKey(0);
			if (key2 == 'a') cvtColor(src, src, COLOR_BGR2XYZ);
			if (key2 == 'z') cvtColor(src, src, COLOR_BGR2YCrCb);
			if (key2 == 'e') cvtColor(src, src, COLOR_BGR2HSV);
			if (key2 == 'r') cvtColor(src, src, COLOR_BGR2Lab);
			if (key2 == 't') cvtColor(src, src, COLOR_BGR2Luv);
			if (key2 == 'y') cvtColor(src, src, COLOR_BGR2HLS);
			if (key2 == 'u') cvtColor(src, src, COLOR_BGR2YUV);
			// Initialisation K
			int K = alpha_sliderK;
			// Mise en forme de ligne de l'image
			int tailleLigne = src.rows * src.cols;
			Mat src_ligne = src.reshape(1, tailleLigne);
			src_ligne.convertTo(src_ligne, CV_32F);
			// K means 
			vector<int> labels;
			Mat1f colors;
			kmeans(src_ligne, K, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.), 2, KMEANS_PP_CENTERS, colors);
			for (unsigned int i = 0; i < tailleLigne; i++) {
				src_ligne.at<float>(i, 0) = colors(labels[i], 0);
				src_ligne.at<float>(i, 1) = colors(labels[i], 1);
				src_ligne.at<float>(i, 2) = colors(labels[i], 2);
			}
			// Remise en forme image
			Mat outputImage = src_ligne.reshape(3, src.rows);
			outputImage.convertTo(outputImage, CV_8U);

			if (key2 == 'a') { cvtColor(outputImage, outputImage, COLOR_XYZ2BGR); imwrite("kspace/XYZ.jpg", outputImage); }
			if (key2 == 'z') { cvtColor(outputImage, outputImage, COLOR_YCrCb2BGR); imwrite("kspace/YCrCb.jpg", outputImage); }
			if (key2 == 'e') { cvtColor(outputImage, outputImage, COLOR_HSV2BGR); imwrite("kspace/HSV.jpg", outputImage); }
			if (key2 == 'r') { cvtColor(outputImage, outputImage, COLOR_Lab2BGR); imwrite("kspace/Lab.jpg", outputImage); }
			if (key2 == 't') { cvtColor(outputImage, outputImage, COLOR_Luv2BGR); imwrite("kspace/Luv.jpg", outputImage); }
			if (key2 == 'y') { cvtColor(outputImage, outputImage, COLOR_HLS2BGR); imwrite("kspace/HLS.jpg", outputImage); }
			if (key2 == 'u') { cvtColor(outputImage, outputImage, COLOR_YUV2BGR); imwrite("kspace/YUV.jpg", outputImage); }
			// Affichage
			namedWindow("kmeans", WINDOW_NORMAL);
			imshow("kmeans", outputImage);

		}
		// Detection planche avec les nombres
		if (key == 'v') {
			time_t start, end;

			// On initialise la taille theorique des planches
			//int box_w = 420, box_h = 190; // 3 
			int box_w = 380, box_h = 170; // 1 2 6 
			//int box_w = 400, box_h = 180; // 4 5 


			// --------------------------------------------------------------------------------------------- //
			// --------------------------------------- Kmeans ---------------------------------------------- //
			// Initialisation K
			int K = 3;
			// Mise en forme de ligne de l'image
			int tailleLigne = src.rows * src.cols;
			Mat src_ligne = src.reshape(1, tailleLigne);
			src_ligne.convertTo(src_ligne, CV_32F);
			// K means 
			vector<int> labels;
			Mat1f kcolors;
			kmeans(src_ligne, K, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.), 2, KMEANS_PP_CENTERS, kcolors);

			double k_color_min = 255, k_color_max = 0;
			int k_color_imin, k_color_imax;
			for (int i = 0; i < K; i++)
			{
				double gray_value = 0.2627 * kcolors[i][2] + 0.678 * kcolors[i][1] + 0.0593 * kcolors[i][0];
				if (gray_value < k_color_min) {
					k_color_min = gray_value;
					k_color_imin = i;
				}
				if (gray_value > k_color_max) {
					k_color_max = gray_value;
					k_color_imax = i;
				}
			}
			int kmean_plank_color;
			for (int i = 0; i < K; i++) {
				if (i != k_color_imin && i != k_color_imax)
					kmean_plank_color = i;
			}

			for (unsigned int i = 0; i < tailleLigne; i++) {
				//src_ligne.at<float>(i, 0) = kcolors(labels[i], 0);
				//src_ligne.at<float>(i, 1) = kcolors(labels[i], 1);
				//src_ligne.at<float>(i, 2) = kcolors(labels[i], 2);
				if (labels[i] == kmean_plank_color) {
					src_ligne.at<float>(i, 0) = 255;
					src_ligne.at<float>(i, 1) = 255;
					src_ligne.at<float>(i, 2) = 255;
				}
				else {
					src_ligne.at<float>(i, 0) = 0;
					src_ligne.at<float>(i, 1) = 0;
					src_ligne.at<float>(i, 2) = 0;
				}
			}
			// Remise en forme image
			Mat kmean = src_ligne.reshape(3, src.rows);
			//kmean.convertTo(kmean, CV_8U);

			// Affichage
			namedWindow("kmeans", WINDOW_NORMAL);
			imshow("kmeans", kmean);
			// --------------------------------------------------------------------------------------------- //


			// --------------------------------------------------------------------------------------------- //
			// ------------------------- Recuperation des contour de canny netoyes ------------------------- //
			time(&start);
			cout << "Recuperation de canny propre : ";
			// Conversion en niveau de gris 
			cvtColor(src, src_gray, CV_BGR2GRAY);
			namedWindow("Image gray", WINDOW_NORMAL);
			imshow("Image gray", src_gray);

			/// Detect edges using canny
			Canny(src_gray, canny_output, alpha_sliderT1, alpha_sliderT2, 3);
			namedWindow("Canny", WINDOW_NORMAL);
			imshow("Canny", canny_output);
			imwrite("Canny.jpg", canny_output);
			// Truc un peu magique 
			// On fait une fermeture sur canny avec un gros filtre 
			// On extrait les bord de la fermeture avec une egalite entre la dilatation des deux composante par un filtre 5x5 
			// On floute fortement ce qui n'est pas un bord puis on remet les bords non flouté dans l'image
			// On detecte a nouveau les bord avec Canny 

			Mat mask_canny, mask1, mask2;
			morphologyEx(canny_output, mask1, MORPH_CLOSE, getStructuringElement(CV_SHAPE_RECT, Size(alpha_sliderFS, alpha_sliderFS)));
			bitwise_not(mask1, mask2);

			dilate(mask1, mask1, getStructuringElement(CV_SHAPE_RECT, Size(5, 5)));
			dilate(mask2, mask2, getStructuringElement(CV_SHAPE_RECT, Size(5, 5)));

			bitwise_and(mask1, mask2, mask_canny);

			Mat test_b, test_o;
			GaussianBlur(src_gray, test_b, Size(9, 9), 2);
			bitwise_and(src_gray, mask_canny, test_o);
			bitwise_not(mask_canny, mask_canny);
			bitwise_and(test_b, mask_canny, test_b);
			mask_canny = test_b + test_o;

			Canny(mask_canny, canny_output, alpha_sliderT1, alpha_sliderT2, 3);
			namedWindow("CannyClean", WINDOW_NORMAL);
			imshow("CannyClean", canny_output);
			imwrite("CannyClean.jpg", canny_output);

			time(&end);
			cout << double(end - start) << "s" << endl;
			// --------------------------------------------------------------------------------------------- //


			// --------------------------------------------------------------------------------------------- //
			// ------------------ Recuperation de l'angle pour aligner la grille a l'image ------------------//
			time(&start);
			cout << "Recuperation de l'angle : ";
			Mat canny;
			src = imread("CannyClean.jpg");
			cvtColor(src, canny, COLOR_BGR2GRAY);

			vector<Vec2f> lines;
			HoughLines(canny, lines, 1, CV_PI / 180, 150);
			int nbh = 100;
			while (lines.size() > 10) {
				HoughLines(canny, lines, 1, CV_PI / 180, alpha_sliderTH + nbh);
				//cout << lines.size() << endl;
				if (lines.size() > 1000) nbh += 100;
				if (lines.size() > 100) nbh += 50;
				if (lines.size() > 10) nbh += 10;
			}
			vector<float> angles;
			for (int i = 0; i < lines.size(); i++) {
				angles.push_back(lines[i][1]);
				//cout << lines[i][1] << endl;
			}
			sort(angles.begin(), angles.end());
			float avgAngle = (angles[(int)(angles.size() / 2)] - CV_PI / 2);
			time(&end);
			cout << double(end - start) << "s" << endl;
			// --------------------------------------------------------------------------------------------- //

			// Grille et watershed 
			Mat imgresult;
			src = imread(images_number[current_image]);
			// --------------------------------------------------------------------------------------------- //
			// ------------------------- Rotation de l'image selon l'angle --------------------------------- //
			time(&start);
			cout << "Rotation de l'image : ";
			Point2f src_center(src.cols / 2.0F, src.rows / 2.0F);
			Mat rot_mat = getRotationMatrix2D(src_center, avgAngle * 180 / CV_PI, 1.0);
			warpAffine(src, src, rot_mat, src.size());

			// Selection de seed a la main
			namedWindow("Select dim", WINDOW_NORMAL);
			//setMouseCallback("Select dim", onClick);
			imshow("Select dim", src);
			//waitKey(0);
			//cout << pBL << pUR << endl;
			time(&end);
			cout << double(end - start) << "s" << endl;
			// --------------------------------------------------------------------------------------------- //

			// --------------------------------------------------------------------------------------------- //
			// --------- Preparation de l'image pour le watershed et pour la detection de grille ----------- //
			Mat bin;
			morphologyEx(src, imgresult, MORPH_CLOSE, getStructuringElement(CV_SHAPE_RECT, Size(15, 15)));
			morphologyEx(src, bin, MORPH_ERODE, getStructuringElement(CV_SHAPE_RECT, Size(15, 15)));
			cvtColor(bin, bin, COLOR_BGR2GRAY);

			for (int i = 0; i < bin.rows; i++) {
				for (int j = 0; j < bin.cols; j++) {
					int gray = (int)bin.at<uchar>(i, j);
					if (gray + 70 > 255)
						bin.at<uchar>(i, j) = (uchar)255;
					else
						bin.at<uchar>(i, j) = (uchar)gray + 70;
				}
			}

			threshold(bin, bin, 125, 255, THRESH_BINARY_INV);
			// --------------------------------------------------------------------------------------------- //

			// --------------------------------------------------------------------------------------------- //
			// ----- Recuperation des regions et Recoloriage en blanc des region non connexe au bords ------ //
			// ----- de l'image et pas trop grandes (surface inferieur a la surface de la planche) --------- //
			time(&start);
			cout << "Selection des regions : ";
			Mat image_regions;
			connectedComponents(bin, image_regions);
			double minimum, maximum;
			minMaxLoc(image_regions, &minimum, &maximum);
			// Calcul de l'histogramme de l'image pour compter la surface de chaque region 
			vector<int> hist(maximum + 1, 0);
			for (int i = 0; i < image_regions.rows; i++) {
				for (int j = 0; j < image_regions.cols; j++) {
					int index = image_regions.at<int>(i, j);
					hist[index]++;
				}
			}
			// On ne garde pas les regions trop grandes (les lignes entres les planches sont visees)
			vector<Vec3b> colorsC;
			for (size_t i = 0; i < maximum; i++) {
				if (hist[i + 1] < box_w * box_h)
					colorsC.push_back(Vec3b((uchar)255, (uchar)255, (uchar)255));
				else
					colorsC.push_back(Vec3b((uchar)0, (uchar)0, (uchar)0));
			}
			// On ne garde pes les regions connexe au bords/*
			for (int i = 0; i < image_regions.rows; i++) {
				int index = image_regions.at<int>(i, 0);
				if (index > 0 && index <= maximum) colorsC[index - 1] = Vec3b((uchar)0, (uchar)0, (uchar)0);
				index = image_regions.at<int>(i, image_regions.cols - 1);
				if (index > 0 && index <= maximum) colorsC[index - 1] = Vec3b((uchar)0, (uchar)0, (uchar)0);
			}
			for (int i = 0; i < image_regions.cols; i++) {
				int index = image_regions.at<int>(0, i);
				if (index > 0 && index <= maximum) colorsC[index - 1] = Vec3b((uchar)0, (uchar)0, (uchar)0);
				index = image_regions.at<int>(image_regions.rows - 1, i);
				if (index > 0 && index <= maximum) colorsC[index - 1] = Vec3b((uchar)0, (uchar)0, (uchar)0);
			}
			// Create the result image
			Mat image_regions_filtered = Mat::zeros(image_regions.size(), CV_8UC3);

			// On colorie les regions avec les couleurs assignes precedament 
			for (int i = 0; i < image_regions.rows; i++) {
				for (int j = 0; j < image_regions.cols; j++) {
					int index = image_regions.at<int>(i, j);
					if (index > 0 && index <= static_cast<int>(colorsC.size())) {
						image_regions_filtered.at<Vec3b>(i, j) = colorsC[index - 1];
					}
				}
			}
			cvtColor(image_regions_filtered, image_regions_filtered, COLOR_BGR2GRAY);
			namedWindow("Contours", WINDOW_NORMAL);
			imshow("Contours", image_regions_filtered);
			time(&end);
			cout << double(end - start) << "s" << endl;
			// --------------------------------------------------------------------------------------------- //

			// --------------------------------------------------------------------------------------------- //
			// -------------- Alignement de la grille a l'image avec l'image des regions ------------------- //
			time(&start); cout << "Alignement grille / regions : ";
			// On recupere l'image pour avoir la taille puis on la convertie en gris 
			Mat grille = imread(images_number[current_image]);
			cvtColor(grille, grille, COLOR_BGR2GRAY);
			// On set le minimum a la resolution de l'image 
			int min = src.rows * src.cols;
			int imin = 0, jmin = 0;
			// On initialise le nombre de ligne et de colonnes 
			int row = (int)((double)src.rows / (double)box_h), col = (int)((double)src.cols / (double)box_w);
			// On balaye avec la grille pour trouver le meilleur fiting des marqueurs 
			for (int i = 0; i < box_w; i += 10) {
				for (int j = 0; j < box_h; j += 10) {
					// On dessine notre grille :
					// On se decale de l'indice actuel et on trace un point tout les interval de la taille theorique des boites 
					grille.setTo(0);
					for (int x = 0; x < col; x++) {
						for (int y = 0; y < row; y++) {
							circle(grille, Point(i + x * box_w, j + y * box_h), 20, Scalar(255), -1);
						}
					}

					// On calcul al difference entre les deux images pour voir le nombre de pixels superposé sur la grille 
					Mat dif = grille - image_regions_filtered;
					int nbdif = countNonZero(dif);

					// On regarde si c'est le min 
					if (nbdif < min) {
						min = nbdif;
						imin = i;
						jmin = j;
					}
					//namedWindow("Dif", WINDOW_NORMAL);
					//imshow("Dif", dif);
				}
			}
			cout << " ( Min = " << min << " ) ";
			// On dessine la grille avec les meilleurs parametres 
			grille.setTo(0);
			int nbcolor = 0;
			for (int x = 0; x < col; x++) {
				for (int y = 0; y < row; y++) {
					circle(grille, Point(imin + x * box_w, jmin + y * box_h), 20, Scalar(++nbcolor), -1);
				}
			}

			// Decoupage de l'image 
			Mat test;
			bitwise_and(grille, image_regions_filtered, test);
			int minx = test.cols, miny = test.rows, maxx = 0, maxy = 0;
			for (int i = 0; i < test.rows; i++) {
				for (int j = 0; j < test.cols; j++) {
					int index = (int)test.at<uchar>(i, j);
					if (index > 0) {
						if (j < minx) minx = j;
						if (i < miny) miny = i;
						if (j > maxx) maxx = j;
						if (i > maxy) maxy = i;
					}
				}
			}
			minx = minx - box_w < 0 ? 0 : minx - box_w / 2;
			miny = miny - box_h < 0 ? 0 : miny - box_h / 2;
			maxx = maxx + box_w > src.cols ? src.cols : maxx + box_w / 2;
			maxy = maxy + box_h > src.rows ? src.rows : maxy + box_h / 2;
			rectangle(test, Point(minx, miny), Point(maxx, maxy), Scalar(150));
			namedWindow("les points en colisions", WINDOW_NORMAL);
			imshow("les points en colisions", test);


			namedWindow("Grilleav", WINDOW_NORMAL);
			imshow("Grilleav", grille);

			imgresult = imgresult(Rect(Point(minx, miny), Point(maxx, maxy)));
			grille = grille(Rect(Point(minx, miny), Point(maxx, maxy)));
			src = src(Rect(Point(minx, miny), Point(maxx, maxy)));
			kmean = kmean(Rect(Point(minx, miny), Point(maxx, maxy)));

			imwrite("image_cut.jpg", src);

			namedWindow("Select dim", WINDOW_NORMAL);
			imshow("Select dim", src);

			namedWindow("Grille", WINDOW_NORMAL);
			imshow("Grille", grille);
			time(&end); cout << double(end - start) << "s" << endl;
			// --------------------------------------------------------------------------------------------- //


			// --------------------------------------------------------------------------------------------- //
			// ------------------------------ Application du watershed ------------------------------------- //
			time(&start); cout << "Application du watershed : ";
			// Conversion en image 32bit single channel pour le watersheding
			Mat tgrille;
			grille.convertTo(tgrille, CV_32S);
			grille = tgrille;
			// On applique l'algorithm watershed
			watershed(imgresult, tgrille);
			// On convertie le resultat du watershed en image 8bit 
			Mat mark;
			grille.convertTo(mark, CV_8U);
			bitwise_not(mark, mark);
			// Generate random colors
			vector<Vec3b> colors;
			for (size_t i = 0; i < nbcolor; i++) {
				int b = theRNG().uniform(0, 256), g = theRNG().uniform(0, 256), r = theRNG().uniform(0, 256);
				colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
			}
			// Create the result image
			Mat watershed_result = Mat::zeros(grille.size(), CV_8UC3);
			// Fill labeled objects with random colors
			for (int i = 0; i < grille.rows; i++) {
				for (int j = 0; j < grille.cols; j++) {
					int index = grille.at<int>(i, j);
					if (index > 0 && index <= static_cast<int>(nbcolor)) {
						watershed_result.at<Vec3b>(i, j) = colors[index - 1];
					}
				}
			}
			// Visualize the final image
			namedWindow("Final Result", WINDOW_NORMAL);
			imshow("Final Result", watershed_result);
			imwrite("kmean.jpg", watershed_result);
			time(&end); cout << double(end - start) << "s" << endl;
			// --------------------------------------------------------------------------------------------- //


			// --------------------------------------------------------------------------------------------- //
			// ------------------------------------ Bounding box ------------------------------------------- //
			/*
			cvtColor(watershed_result, src_gray, COLOR_BGR2GRAY);
			blur(src_gray, src_gray, Size(3, 3));

			Canny(src_gray, canny_output, alpha_sliderT1, alpha_sliderT2);

			vector<vector<Point> > contours;
			findContours(canny_output, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

			vector<vector<Point> > contours_poly(contours.size());
			vector<Rect> boundRect(contours.size());
			vector<Point2f>centers(contours.size());
			vector<float>radius(contours.size());
			for (size_t i = 0; i < contours.size(); i++)
			{
				approxPolyDP(contours[i], contours_poly[i], 3, true);
				boundRect[i] = boundingRect(contours_poly[i]);
				minEnclosingCircle(contours_poly[i], centers[i], radius[i]);
			}

			Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
			Mat drawing2 = Mat::zeros(canny_output.size(), CV_8UC3);

			int min_w = box_w - 40, max_w = box_w + 40, min_h = box_h - 40, max_h = box_h + 40;
			double avg_x = 0, avg_y = 0;
			double avgdcx = 0, avgdcy = 0; // average decalage des centre en x et y
			int nb = 0;
			for (size_t i = 0; i < contours.size(); i++)
			{
				Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
				//drawContours(drawing, contours_poly, (int)i, color);
				double taille_x = sqrt((boundRect[i].tl().x - boundRect[i].br().x) * (boundRect[i].tl().x - boundRect[i].br().x));
				double taille_y = sqrt((boundRect[i].tl().y - boundRect[i].br().y) * (boundRect[i].tl().y - boundRect[i].br().y));
				//if (320 < taille_x && taille_x < 480 && 140 < taille_y && taille_y < 200) {  // Extraction 1
				//if (taille_x / taille_y > 1.9 && taille_x / taille_y < 3.1 && 300 < taille_x && taille_x < 600 && (100 < taille_y && taille_y < 250) { // Extraction 2
				if (taille_x / taille_y > 2.5 && taille_x / taille_y < 3.3 && ((min_w < taille_x && taille_x < max_w) || (min_h < taille_y && taille_y < min_h))) { // Extraction 3
					avg_x += taille_x;
					avg_y += taille_y;
					nb++;
					avgdcx += centers[i].x - (int)(centers[i].x / box_w) * box_w - imin;
					avgdcy += centers[i].y - (int)(centers[i].y / box_h) * box_h - jmin;
					//circle(drawing, centers[i], (int)radius[i], color, 2);
				}
			}
			avg_x = avg_x / nb;
			avg_y = avg_y / nb;
			avgdcx = avgdcx / nb;
			avgdcy = avgdcy / nb;
			int num_planche = 0;
			for (size_t i = 0; i < contours.size(); i++)
			{
				Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
				//drawContours(drawing, contours_poly, (int)i, color);
				double taille_x = sqrt((boundRect[i].tl().x - boundRect[i].br().x) * (boundRect[i].tl().x - boundRect[i].br().x));
				double taille_y = sqrt((boundRect[i].tl().y - boundRect[i].br().y) * (boundRect[i].tl().y - boundRect[i].br().y));
				//if (320 < taille_x && taille_x < 480 && 140 < taille_y && taille_y < 200) {
				if (taille_x / taille_y > 2.5 && taille_x / taille_y < 3.3 && ((min_w < taille_x && taille_x < max_w) || (min_h < taille_y && taille_y < min_h))) { // Extraction 3
					//cout << taille_x / taille_y << endl;
					rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, -3);
					for (int a = 0; a < 8; a++) {
						rectangle(drawing, Point(boundRect[i].tl().x + a * avg_x, boundRect[i].tl().y), Point(boundRect[i].br().x + a * avg_x, boundRect[i].br().y), color, 3);
						rectangle(drawing, Point(boundRect[i].tl().x + a * -1 * avg_x, boundRect[i].tl().y), Point(boundRect[i].br().x + a * -1 * avg_x, boundRect[i].br().y), color, 3);
						if (boundRect[i].tl().x + a * avg_x > 0 && boundRect[i].tl().x + a * avg_x < src.size().width && boundRect[i].br().x + a * avg_x > 0 && boundRect[i].br().x + a * avg_x < src.size().width) {
							Rect roi(Point(boundRect[i].tl().x + a * avg_x, boundRect[i].tl().y), Point(boundRect[i].br().x + a * avg_x, boundRect[i].br().y));
							Mat planche = src(roi);
							imwrite("planches/la planche" + to_string(num_planche++) + ".jpg", planche);
						}
						if (boundRect[i].tl().x + a * -1 * avg_x > 0 && boundRect[i].tl().x + a * -1 * avg_x < src.size().width && boundRect[i].br().x + a * -1 * avg_x > 0 && boundRect[i].br().x + a * -1 * avg_x < src.size().width) {
							Rect roi(Point(boundRect[i].tl().x + a * -1 * avg_x, boundRect[i].tl().y), Point(boundRect[i].br().x + a * -1 * avg_x, boundRect[i].br().y));
							Mat planche = src(roi);
							imwrite("planches/la planche" + to_string(num_planche++) + ".jpg", planche);
						}
					}

				}
				else {
					//rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2);
				}
				//cout << "Taille x = " << taille_x << " , Taille y = " << taille_y << endl;
				//circle(drawing, centers[i], (int)radius[i], color, 2);
			}
			for (int x = 0; x < col; x++) {
				for (int y = 0; y < row; y++) {
					Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
					Point centre = Point(imin + x * box_w + avgdcx, jmin + y * box_h + avgdcy);
					rectangle(drawing2, Point(centre.x - avg_x / 2, centre.y - avg_y / 2), Point(centre.x + avg_x / 2, centre.y + avg_y / 2), color, 3);
					if (0 < centre.x - avg_x / 2 && centre.x - avg_x / 2 < src.cols && 0 < centre.x + avg_x / 2 && centre.x + avg_x / 2 < src.cols && 0 < centre.y - avg_y / 2 && centre.y - avg_y / 2 < src.rows && 0 < centre.y + avg_y / 2 && centre.y + avg_y / 2 < src.rows) {
						Rect roi(Point(centre.x - avg_x / 2, centre.y - avg_y / 2), Point(centre.x + avg_x / 2, centre.y + avg_y / 2));
						Mat planche = src(roi);
						imwrite("planches/p_" + to_string(num_planche++) + ".jpg", planche);
					}
				}
			}

			namedWindow("Contours", WINDOW_NORMAL);
			imshow("Contours", drawing);
			namedWindow("Contours2", WINDOW_NORMAL);
			imshow("Contours2", drawing2);
			imwrite("bbox.jpg", drawing);


			Mat mask;
			cvtColor(drawing, mask_canny, CV_BGR2GRAY);
			threshold(mask_canny, mask_canny, 10, 255, THRESH_BINARY);
			cvtColor(mask_canny, mask_canny, CV_GRAY2BGR);
			bitwise_and(src, mask_canny, drawing);

			// Visualize the final image
			namedWindow("Final Result after mask", WINDOW_NORMAL);
			imshow("Final Result after mask", drawing);
			*/
			// --------------------------------------------------------------------------------------------- //


			// --------------------------------------------------------------------------------------------- //
			// --------------------------- Decalade des boites via watershed ------------------------------- 
			time(&start);
			cout << "Bounding boxing : ";
			// A recuperer automatiquement 
			double taille_mire_image = 553; // 1 
			//double taille_mire_image = 549; // 2 
			//double taille_mire_image = 590; // 3
			//double taille_mire_image = 602; // 4 
			//double taille_mire_image = 593; // 5
			double taille_mire = 250;
			// A donner en param 
			double largeur_planche = 160, hauteur_planche = 55;
			// Calcul de la taille dans l'image 
			double real_box_w_numbers = taille_mire_image * largeur_planche / taille_mire;
			double real_box_h_numbers = taille_mire_image * hauteur_planche / taille_mire;
			// Initialisation de l'image contenant les boites
			Mat drawing = Mat::zeros(grille.size(), CV_8UC3);

			// Recalage des points de la grille apres decoupe de l'image 
			if (minx > 0)
				imin = box_w - ((minx - imin) % box_w);
			if (miny > 0)
				jmin = box_h - ((miny - jmin) % box_h);
			row = (int)((double)src.rows / (double)box_h);
			col = (int)((double)src.cols / (double)box_w);

			vector<vector<Point>> listeBoites(col, vector<Point>(row, Point(0, 0)));
			vector<vector<int>> listePoids(col, vector<int>(row, 0));
			// Pour chaque marqueur on lui associe une boite 
			for (int x = 0; x < col; x++) {
				for (int y = 0; y < row; y++) {
					// On prend le centre du marqueur de la planche 
					Point centre(imin + x * box_w, jmin + y * box_h);
					// On recupere le label de la zone 
					int index = grille.at<int>(centre);
					// On calcule les bornes bix (borne inf x), biy (borne inf y), bsx (borne sup x) et bsy (borne sup y)
					// de la zone dans laquel on cherchera a aligner notre gabari 
					int bix = (centre.x - real_box_w_numbers < 0) ? 0 : centre.x - real_box_w_numbers;
					int biy = (centre.y - real_box_h_numbers < 0) ? 0 : centre.y - real_box_h_numbers;
					int bsx = (centre.x + real_box_w_numbers >= grille.cols) ? grille.cols - 1 : centre.x + real_box_w_numbers;
					int bsy = (centre.y + real_box_h_numbers >= grille.rows) ? grille.rows - 1 : centre.y + real_box_h_numbers;
					// On recupere avec les bornes les positions possible pour le coins haut gauche
					// bix et biy ne change pas mais on soustrait la taille de la boite aux bornes bsx et bsy
					bsx -= real_box_w_numbers;
					bsy -= real_box_h_numbers;
					// Initialisation des param a recuperer et a maxomiser 
					int bestx = bix;
					int besty = biy;
					int maxpixel = 0;
					int maxpixel_bord = 0;
					// On regarde pour chaque position le nombre de pixel appartenant au label 
					// on gardera la position pour laquelle on a le plus de pixels dans notre gabari 
					for (int i = bix + 1; i < bsx - 1; i += 1) {
						for (int j = biy + 1; j < bsy - 1; j += 1) {
							// On compte le nombre de pixel dans la zone du gabari avec pour coin haut gauche le point (i,j)
							int nbpixel = 0;
							//cout << "(" << i + real_box_w << "," << j + real_box_h << ")" << endl;
							int pix_val = grille.at<int>(j + real_box_h_numbers, i + real_box_w_numbers);

							for (size_t gx = 0; gx < real_box_w_numbers; gx += 10) {
								for (size_t gy = 0; gy < real_box_h_numbers; gy += 10) {
									if (grille.at<int>(j + gy, i + gx) == index)
										nbpixel++;
								}
							}

							// On compare le nombre de pixels obtenu 
							// S'il y a autent de pixel alors re compare le nombre ligne du partage des eaux au bord du gabari
							// S'il y a plus de pixel on le garde automatiquement
							if (nbpixel >= maxpixel) {
								// On compte les pixels apparteannt au partage des eaux sur les bord 
								int nbpixel_bord = 0;
								for (size_t gx = 0; gx < real_box_w_numbers; gx += 1) {
									if (grille.at<int>(j, i + gx) < 0)
										nbpixel_bord++;
								}
								for (size_t gy = 0; gy < real_box_h_numbers; gy += 1) {
									if (grille.at<int>(j + gy, i) < 0)
										nbpixel_bord++;
								}
								if (nbpixel == maxpixel) { // Autent de pixel donc on compare les bords 
									if (nbpixel_bord > maxpixel_bord) {
										bestx = i;
										besty = j;
										maxpixel = nbpixel;
										maxpixel_bord = nbpixel_bord;
									}
								}
								else { // Plus de pixel donc on retiens le point 
									bestx = i;
									besty = j;
									maxpixel = nbpixel;
									maxpixel_bord = nbpixel_bord;
								}
							}

						}
					}

					// On stock le meilleur rectangle 
					listeBoites[x][y] = Point(bestx, besty);
					listePoids[x][y] = maxpixel_bord;

					circle(drawing, Point(bestx, besty), 20, Scalar(255, 255, 255), -1);

				}
			}
			// Hough
			cvtColor(drawing, canny_output, CV_BGR2GRAY);
			HoughLines(canny_output, lines, 1, CV_PI / 180, alpha_sliderTH);
			src_gray = Mat::zeros(drawing.size(), CV_8UC1);
			cvtColor(src_gray, src_gray, COLOR_GRAY2RGB);
			for (size_t i = 0; i < lines.size(); i++)
			{
				float rho = lines[i][0];
				float theta = lines[i][1];
				double a = cos(theta), b = sin(theta);
				double x0 = a * rho, y0 = b * rho;
				Point pt1(cvRound(x0 + 10000 * (-b)),
					cvRound(y0 + 10000 * (a)));
				Point pt2(cvRound(x0 - 10000 * (-b)),
					cvRound(y0 - 10000 * (a)));
				if ((theta > 0 - CV_PI / 32 && theta < 0 + CV_PI / 32) || (theta > CV_PI / 2 - CV_PI / 32 && theta < CV_PI / 2 + CV_PI / 32))
					line(src_gray, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
			}

			namedWindow("Res hough", WINDOW_NORMAL);
			imshow("Res hough", src_gray);
			cvtColor(src_gray, src_gray, COLOR_BGR2GRAY);
			drawing = Mat::zeros(grille.size(), CV_8UC3);
			String change = "0", prev_change = "-1";
			while (change != prev_change) {
				prev_change = change;
				change = "";

				for (int x = 0; x < col; x++) {
					for (int y = 0; y < row; y++) {
						Point centre(imin + x * box_w, jmin + y * box_h);
						int index = grille.at<int>(centre);

						if (x > 0) {
							//cout << "Point (" << x << "," << y << ") x = " << listeBoites[x][y].x << " , boite de gauche x = " << listeBoites[x - 1][y].x + real_box_w << endl;
							// distance entre deux coins
							int dist = listeBoites[x][y].x - (listeBoites[x - 1][y].x + real_box_w_numbers);
							if (dist < 0) { // Superposition de boite donc on tente un decalage 
								// On note une superposition entre les deux planches 
								change += to_string(x) + " " + to_string(y) + " r ";
								circle(drawing, centre, 20, Scalar(0, 0, 255), -1);
								//cout << "Superposition " << x << "," << y << " : ";
								// On cherche la premiere ligne croisee dans notre boite pour savoir laquel colle le moins avec l'alignement 
								Mat boite = src_gray(Rect(
									listeBoites[x][y].x < 0 ? 0 : listeBoites[x][y].x,
									listeBoites[x][y].y < 0 ? 0 : listeBoites[x][y].y,
									listeBoites[x][y].x + real_box_w_numbers >= src_gray.cols ? src_gray.cols - listeBoites[x][y].x : real_box_w_numbers,
									listeBoites[x][y].y + real_box_h_numbers >= src_gray.rows ? src_gray.rows - listeBoites[x][y].y : real_box_h_numbers));
								Mat colision = src_gray(Rect(
									listeBoites[x - 1][y].x < 0 ? 0 : listeBoites[x - 1][y].x,
									listeBoites[x - 1][y].y < 0 ? 0 : listeBoites[x - 1][y].y,
									listeBoites[x - 1][y].x + real_box_w_numbers >= src_gray.cols ? src_gray.cols - listeBoites[x - 1][y].x : real_box_w_numbers,
									listeBoites[x - 1][y].y + real_box_h_numbers >= src_gray.rows ? src_gray.rows - listeBoites[x - 1][y].y : real_box_h_numbers));
								int firstLigne_boite = -1, firstligne_colision = -1;
								for (int i = 0; i < real_box_w_numbers - 1; i++)
								{
									// Si on trouve une ligne on la note
									if (firstLigne_boite == -1 && boite.at<uchar>(real_box_h_numbers / 2, i) != 0)
										firstLigne_boite = i;
									if (firstligne_colision == -1 && colision.at<uchar>(real_box_h_numbers / 2, i) != 0)
										firstligne_colision = i;
								}
								//cout << firstLigne_boite << " - " << firstligne_colision << endl;
								// On decale la boite qui est le moins bien alignee aux autres 
								if (firstLigne_boite > firstligne_colision && !(firstligne_colision > real_box_w_numbers / 2))
								{
									//cout << "decalage moi meme" << endl;
									listeBoites[x][y].x = listeBoites[x - 1][y].x + real_box_w_numbers;
								}
								else
								{
									//cout << "decalage adjacent" << endl;
									listeBoites[x - 1][y].x = listeBoites[x][y].x - real_box_w_numbers;
								}
							}
							else if (dist > 0.1 * real_box_w_numbers) { // Il y a un trou entre deux boite 
							 // On note la detection d'un trou entre les deux boites 
								change += to_string(x) + " " + to_string(y) + " b ";
								Mat colision = src_gray(Rect(
									listeBoites[x - 1][y].x < 0 ? 0 : listeBoites[x - 1][y].x,
									listeBoites[x - 1][y].y < 0 ? 0 : listeBoites[x - 1][y].y,
									listeBoites[x - 1][y].x + real_box_w_numbers >= src_gray.cols ? src_gray.cols - listeBoites[x - 1][y].x : real_box_w_numbers,
									listeBoites[x - 1][y].y + real_box_h_numbers >= src_gray.rows ? src_gray.rows - listeBoites[x - 1][y].y : real_box_h_numbers));
								int firstligne_colision = -1, lastligne_colision = -1;
								for (int i = 0; i < real_box_w_numbers - 1; i++)
								{
									// Si on trouve une ligne on la note
									if (firstligne_colision == -1 && colision.at<uchar>(real_box_h_numbers / 2, i) != 0)
										firstligne_colision = i;
									if (lastligne_colision == -1 && colision.at<uchar>(real_box_h_numbers / 2, real_box_w_numbers - 1 - i) != 0)
										lastligne_colision = real_box_w_numbers - 1 - i;
								}
								// On regarde si on peut aligner au autres boites de la colonne sans creer de superposition 
								if (firstligne_colision < dist) {
									listeBoites[x - 1][y].x = listeBoites[x - 1][y].x + firstligne_colision;
									circle(drawing, centre, 20, Scalar(255, 0, 0), -1);
								}
								if (lastligne_colision < real_box_w_numbers / 2) {
									listeBoites[x - 1][y].x = listeBoites[x - 1][y].x + dist - firstligne_colision;
									circle(drawing, centre, 20, Scalar(0, 255, 0), -1);
								}
							}
							else {
								circle(drawing, centre, 20, Scalar(255, 255, 255), -1);
							}
						}
						else {
							circle(drawing, centre, 20, Scalar(255, 255, 255), -1);
						}

						//circle(drawing, centre, 20, Scalar(255, 255, 255), -1);
					}
				}
			}
			for (int x = 0; x < col; x++) {
				for (int y = 0; y < row; y++) {
					Point centre(imin + x * box_w, jmin + y * box_h);
					int index = grille.at<int>(centre);

					Mat isplanche = kmean(
						Rect(
							Point(
								listeBoites[x][y].x < 0 ? 0 : (listeBoites[x][y].x >= kmean.cols ? kmean.cols - 1 : listeBoites[x][y].x),
								listeBoites[x][y].y < 0 ? 0 : (listeBoites[x][y].y >= kmean.rows ? kmean.rows - 1 : listeBoites[x][y].y)),
							Point(
								listeBoites[x][y].x + real_box_w_numbers >= kmean.cols ? kmean.cols - 1 : listeBoites[x][y].x + real_box_w_numbers,
								listeBoites[x][y].y + real_box_h_numbers >= kmean.rows ? kmean.rows - 1 : listeBoites[x][y].y + real_box_h_numbers)
							)
						);
					int nbpixel = 0;
					for (int i = 0; i < isplanche.cols; i += 1) {
						for (int j = 0; j < isplanche.rows; j += 1) {
							if (isplanche.at<Vec3f>(j, i)[0] != 0)
								nbpixel++;
						}
					}
					if (nbpixel > isplanche.cols * isplanche.rows * 0.4) {
						rectangle(drawing, listeBoites[x][y], Point(listeBoites[x][y].x + real_box_w_numbers, listeBoites[x][y].y + real_box_h_numbers), colors[index - 1], 5);
						rectangle(src, listeBoites[x][y], Point(listeBoites[x][y].x + real_box_w_numbers, listeBoites[x][y].y + real_box_h_numbers), colors[index - 1], 5);
						cout << listeBoites[x][y] << endl;
					}
				}
			}

			namedWindow("Alignement boite reel", WINDOW_NORMAL);
			imshow("Alignement boite reel", drawing);

			namedWindow("Alignement boite reel source", WINDOW_NORMAL);
			imshow("Alignement boite reel source", src);

			time(&end);
			cout << double(end - start) << "s" << endl;
			// --------------------------------------------------------------------------------------------- //




			cout << "Fin" << endl;


		}
		// Passage de l'autre coté 
		if (key == 'b') {
			src = imread(images_color[current_image]);
			cvtColor(src, src, COLOR_BGR2GRAY);
			cvtColor(src, src, COLOR_GRAY2BGR);
			//morphologyEx(src, src, MORPH_ERODE, getStructuringElement(CV_SHAPE_RECT, Size(alpha_sliderFS, alpha_sliderFS)));
			//src = imread("image_cut.jpg");
			// Initialisation K
			int K = alpha_sliderK;
			// Mise en forme de ligne de l'image
			int tailleLigne = src.rows * src.cols;
			Mat src_ligne = src.reshape(1, tailleLigne);
			src_ligne.convertTo(src_ligne, CV_32F);
			// K means 
			vector<int> labels;
			Mat1f colors;
			kmeans(src_ligne, K, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.), 2, KMEANS_PP_CENTERS, colors);
			for (unsigned int i = 0; i < tailleLigne; i++) {
				src_ligne.at<float>(i, 0) = colors(labels[i], 0);
				src_ligne.at<float>(i, 1) = colors(labels[i], 1);
				src_ligne.at<float>(i, 2) = colors(labels[i], 2);
			}
			// Remise en forme image
			Mat outputImage = src_ligne.reshape(3, src.rows);
			outputImage.convertTo(outputImage, CV_8U);
			// Affichage
			namedWindow("kmeans", WINDOW_NORMAL);
			imshow("kmeans", outputImage);
			imwrite("kmean9.jpg", outputImage);

			src = imread(images_color[current_image]);
			// Conversion en niveau de gris 
			cvtColor(src, src_gray, CV_BGR2GRAY);
			namedWindow("Image gray", WINDOW_NORMAL);
			imshow("Image gray", src_gray);
			/*
			// On diminue la texture de bois
			//bilateralFilter(src_gray, src_gray_filtered, 9, 200, 200);
			GaussianBlur(src_gray, src_gray_filtered, Size(9, 9), 2);
			imwrite("filtered.jpg", src_gray_filtered);
			namedWindow("Image gray filtered", WINDOW_NORMAL);
			imshow("Image gray filtered", src_gray_filtered);
			*/
			/// Detect edges using canny
			Canny(src_gray, canny_output, alpha_sliderT1, alpha_sliderT2, 3);
			namedWindow("Canny", WINDOW_NORMAL);
			imshow("Canny", canny_output);
			imwrite("Canny.jpg", canny_output);
			// Truc un peu magique 
			// On fait une fermeture sur canny avec un gros filtre 
			// On extrait les bord de la fermeture avec une egalite entre la dilatation des deux composante par un filtre 5x5 
			// On floute fortement ce qui n'est pas un bord puis on remet les bords non flouté dans l'image
			// On detecte a nouveau les bord avec Canny 

			Mat mask, mask1, mask2;
			morphologyEx(canny_output, mask1, MORPH_CLOSE, getStructuringElement(CV_SHAPE_RECT, Size(alpha_sliderFS, alpha_sliderFS)));
			bitwise_not(mask1, mask2);

			dilate(mask1, mask1, getStructuringElement(CV_SHAPE_RECT, Size(5, 5)));
			dilate(mask2, mask2, getStructuringElement(CV_SHAPE_RECT, Size(5, 5)));

			bitwise_and(mask1, mask2, mask);

			Mat test_b, test_o;
			GaussianBlur(src_gray, test_b, Size(9, 9), 2);
			bitwise_and(src_gray, mask, test_o);
			bitwise_not(mask, mask);
			bitwise_and(test_b, mask, test_b);
			mask = test_b + test_o;

			Canny(mask, canny_output, alpha_sliderT1, alpha_sliderT2, 3);
			namedWindow("CannyClean", WINDOW_NORMAL);
			imshow("CannyClean", canny_output);
			imwrite("CannyClean.jpg", canny_output);

			vector<Vec2f> lines;
			HoughLines(canny_output, lines, 1, CV_PI / 180, alpha_sliderTH);
			Mat drawing = Mat::zeros(src.size(), CV_8UC3);
			Mat kmean9 = imread("kmean9.jpg");
			cvtColor(kmean9, kmean9, COLOR_BGR2GRAY);
			int bestx = src.cols, besty = src.rows;
			for (size_t i = 0; i < lines.size(); i++)
			{
				float rho = lines[i][0];
				float theta = lines[i][1];
				double a = cos(theta), b = sin(theta);
				double x0 = a * rho, y0 = b * rho;
				Point pt1(cvRound(x0 + 10000 * (-b)),
					cvRound(y0 + 10000 * (a)));
				Point pt2(cvRound(x0 - 10000 * (-b)),
					cvRound(y0 - 10000 * (a)));
				if (theta > CV_PI / 2 - CV_PI / 64 && theta < CV_PI / 2 + CV_PI / 64) {
					//line(drawing, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
					for (int i = 350; i > 0; i--)
					{
						Point p_act = Point(cvRound(x0 - i * (-b)), cvRound(y0 - i * (a)));
						//cout << p_act << endl;
						int count_right = 0, count_left = 0;
						for (int j = 0; j < 5; j++)
						{
							if (p_act.x - j > 0) {
								count_left += kmean9.at<uchar>(p_act.y + 50, p_act.x - j);
							}
							if (p_act.x + j > 0) {
								count_right += kmean9.at<uchar>(p_act.y + 50, p_act.x + j);
							}
						}
						//cout << count_left << " _ " << count_right << " _ " << abs(count_left - count_right) << endl;
						if (abs(count_left - count_right) > 200) {
							//circle(drawing, Point(p_act.x, p_act.y +50), 20, Scalar(255, 0, 0), -1);
							if (p_act.y + 50 < besty) {
								bestx = p_act.x;
								besty = p_act.y + 50;
							}
						}
						else {
							//circle(drawing, Point(p_act.x, p_act.y +50), 2, Scalar(0, 255, 0), -1);
						}
					}
				}

				if (theta > 0 - CV_PI / 64 && theta < 0 + CV_PI / 64) {
					//line(drawing, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
					//circle(drawing, Point(x0, y0+50), 2, Scalar(0, 255, 0), -1);
				}
			}
			// On dessine le coins de la pilme de planche 
			//circle(drawing, Point(bestx,besty), 20, Scalar(255, 0, 0), -1);

			// On dessine les rectangle du coté nombre retourné 
			//1
			//vector<Point> box_numbers = { Point(198, 39),Point(127, 230),Point(130, 396),Point(180, 568),Point(129, 737),Point(109, 910),Point(171, 1073),Point(162, 1247),Point(115, 1411),Point(119, 1574),Point(152, 1735),Point(131, 1873),Point(552, 2),Point(513, 227),Point(506, 393),Point(534, 568),Point(595, 733),Point(498, 904),Point(525, 1028),Point(516, 1235),Point(507, 1406),Point(495, 1567),Point(506, 1737),Point(906, 29),Point(967, 227),Point(878, 398),Point(892, 567),Point(949, 732),Point(871, 894),Point(878, 1063),Point(902, 1190),Point(861, 1402),Point(908, 1561),Point(860, 1734),Point(1260, 98),Point(1321, 231),Point(1255, 395),Point(1289, 564),Point(1303, 703),Point(1257, 897),Point(1292, 1063),Point(1256, 1206),Point(1294, 1356),Point(1266, 1558),Point(1302, 1718),Point(1206, 1901),Point(1620, 76),Point(1675, 228),Point(1621, 400),Point(1643, 567),Point(1657, 730),Point(1623, 872),Point(1646, 1056),Point(1657, 1217),Point(1648, 1392),Point(1626, 1541),Point(1656, 1709),Point(1590, 1854),Point(2029, 231),Point(1989, 400),Point(1997, 566),Point(2011, 734),Point(2024, 928),Point(2000, 1058),Point(2001, 1200),Point(1995, 1382),Point(1979, 1527),Point(2010, 1705),Point(1962, 1887),Point(1971, 2258),Point(2365, 401),Point(2380, 572),Point(2365, 699),Point(2378, 889),Point(2354, 1056),Point(2355, 1218),Point(2349, 1345),Point(2355, 1526),Point(2364, 1704),Point(2315, 1892) };
			//4
			vector<Point> box_numbers = { Point(92, 1300),Point(95, 1484),Point(121, 1666),Point(121, 1845),Point(96, 1993),Point(97, 2201),Point(112, 2380),Point(110, 2586),Point(122, 2749),Point(84, 2923),Point(124, 3111),Point(478, 1302),Point(490, 1483),Point(507, 1664),Point(507, 1841),Point(482, 2020),Point(499, 2201),Point(498, 2374),Point(496, 2553),Point(508, 2687),Point(470, 2917),Point(510, 3098),Point(864, 1303),Point(886, 1486),Point(893, 1664),Point(893, 1840),Point(892, 2018),Point(894, 2197),Point(884, 2373),Point(882, 2545),Point(894, 2725),Point(902, 2891),Point(896, 3067),Point(1262, 1308),Point(1279, 1493),Point(1279, 1661),Point(1280, 1841),Point(1296, 2019),Point(1280, 2197),Point(1287, 2370),Point(1268, 2533),Point(1280, 2708),Point(1287, 2895),Point(1282, 3032),Point(1657, 1345),Point(1656, 1487),Point(1665, 1663),Point(1665, 1840),Point(1682, 2021),Point(1666, 2157),Point(1674, 2370),Point(1629, 2526),Point(1666, 2722),Point(1636, 2859),Point(1668, 3061),Point(2075, 1310),Point(2042, 1487),Point(2055, 1665),Point(2051, 1844),Point(2068, 2016),Point(2055, 2190),Point(2060, 2369),Point(2015, 2507),Point(2052, 2692),Point(2022, 2893),Point(2054, 3060),Point(2529, 1313),Point(2427, 1489),Point(2444, 1669),Point(2439, 1847),Point(2453, 2016),Point(2444, 2207),Point(2446, 2342),Point(2401, 2535),Point(2442, 2725),Point(2408, 2850),Point(2440, 3071) };

			int average_lastBox_x = 2440;
			int min_LastBox_y = 1300;

			for (int i = 0; i < box_numbers.size(); i++) {

			}

			for (int i = 0; i < box_numbers.size(); i++)
			{

				int real_box_w = 377, real_box_h = 130; //4
				//int real_box_w = 395, real_box_h = 136; //1
				double ratio_echelle = 0.98;
				// Coin du paquet de planche + mirroir des boite cote nombre 
				Point coin_haut_gauche = Point(bestx + (average_lastBox_x - box_numbers[i].x) * ratio_echelle, besty + (box_numbers[i].y - min_LastBox_y) * ratio_echelle);
				Point coin_bas_droite = Point(bestx + (average_lastBox_x - box_numbers[i].x) * ratio_echelle + real_box_w, besty + (box_numbers[i].y - min_LastBox_y) * ratio_echelle + real_box_h);
				Mat planche = src(Rect(coin_haut_gauche, coin_bas_droite));
				Scalar avgcolor = mean(planche);
				rectangle(drawing, coin_haut_gauche, coin_bas_droite, avgcolor, 5);
				imwrite("planches/planche_" + to_string(i) + ".jpg", planche);
			}

			namedWindow("Res hough", WINDOW_NORMAL);
			imshow("Res hough", drawing);
			//imwrite("test_class.jpg", drawing);

			namedWindow("src", WINDOW_NORMAL);
			imshow("src", src);
		}
		// Detection lignes puis verticales ---------------------------
		if (key == 's') {
			// Recuperation du billon 
			string pilePlanchesName = "planks-color-1";
			cout << "Planches a segmenter : ";
			cin >> pilePlanchesName;
			Mat src;
			src = imread("PilePlanches/" + pilePlanchesName);
			namedWindow("la source pour le screen", WINDOW_NORMAL);
			imshow("la source pour le screen", src);

			// Recuperation des planches 
			
			// Les taille des planches en moyenne dans les images planks-color-x.jpeg 
			int real_box_w = 395, real_box_h = 136; //1
			//int real_box_w = 349, real_box_h = 120; //2
			//int real_box_w = 330, real_box_h = 113; //3
			//int real_box_w = 377, real_box_h = 130; //4
			//int real_box_w = 383, real_box_h = 131; //5

			cout << "Taille moyenne des planches en pixels :" << endl;
			cout << "Largeur : "; cin >> real_box_w;
			cout << "Hauteur : "; cin >> real_box_h;


			time_t start, end;
			time(&start);
			int num_planche = 0;

			//src = imread(images_color[current_image]);
			// Conversion en niveau de gris 
			cvtColor(src, src_gray, CV_BGR2GRAY);
			namedWindow("Image gray", WINDOW_NORMAL);
			imshow("Image gray", src_gray);
			/*
			// On diminue la texture de bois
			//bilateralFilter(src_gray, src_gray_filtered, 9, 200, 200);
			GaussianBlur(src_gray, src_gray_filtered, Size(9, 9), 2);
			imwrite("filtered.jpg", src_gray_filtered);
			namedWindow("Image gray filtered", WINDOW_NORMAL);
			imshow("Image gray filtered", src_gray_filtered);
			*/
			/// Detect edges using canny
			Canny(src_gray, canny_output, alpha_sliderT1, alpha_sliderT2, 3);
			namedWindow("Canny", WINDOW_NORMAL);
			imshow("Canny", canny_output);
			imwrite("Canny.jpg", canny_output);
			// Truc un peu magique 
			// On fait une fermeture sur canny avec un gros filtre 
			// On extrait les bord de la fermeture avec une egalite entre la dilatation des deux composante par un filtre 5x5 
			// On floute fortement ce qui n'est pas un bord puis on remet les bords non flouté dans l'image
			// On detecte a nouveau les bord avec Canny 

			Mat mask, mask1, mask2;
			morphologyEx(canny_output, mask1, MORPH_CLOSE, getStructuringElement(CV_SHAPE_RECT, Size(alpha_sliderFS, alpha_sliderFS)));
			bitwise_not(mask1, mask2);

			dilate(mask1, mask1, getStructuringElement(CV_SHAPE_RECT, Size(5, 5)));
			dilate(mask2, mask2, getStructuringElement(CV_SHAPE_RECT, Size(5, 5)));

			bitwise_and(mask1, mask2, mask);

			Mat test_b, test_o;
			GaussianBlur(src_gray, test_b, Size(9, 9), 2);
			bitwise_and(src_gray, mask, test_o);
			bitwise_not(mask, mask);
			bitwise_and(test_b, mask, test_b);
			mask = test_b + test_o;

			Canny(mask, canny_output, alpha_sliderT1, alpha_sliderT2, 3);
			namedWindow("CannyClean", WINDOW_NORMAL);
			imshow("CannyClean", canny_output);
			imwrite("CannyClean.jpg", canny_output);


			
			//int real_box_w = 380, real_box_h = 120; //2 no color

			vector<Vec2f> lines;
			vector<vector<double>> points_moyen_accumulateur;
			HoughLines(canny_output, lines, 1, CV_PI / 180, alpha_sliderTH);
			Mat drawing = Mat::zeros(src.size(), CV_8UC3);
			src.copyTo(drawing);
			int bestx = src.cols, besty = src.rows;
			for (size_t i = 0; i < lines.size(); i++)
			{
				float rho = lines[i][0];
				float theta = lines[i][1];
				double a = cos(theta), b = sin(theta);
				double x0 = a * rho, y0 = b * rho;
				if (theta > CV_PI / 2 - CV_PI / 64 && theta < CV_PI / 2 + CV_PI / 64) {
					//line(drawing, Point(x0 - (src.cols / 2) * (-b) -5000 , y0 - (src.cols / 2) * (a)), Point(x0 - (src.cols / 2) * (-b) + 5000, y0 - (src.cols / 2) * (a)), Scalar(0, 0, 255), 3, LINE_AA);
					Point centre_ligne = Point(x0 - (src.cols / 2) * (-b), y0 - (src.cols / 2) * (a));
					circle(drawing, centre_ligne, 2, Scalar(0, 255, 0), -1);

					double min_dist = 2 * real_box_h;
					int i_min_dist = 0;
					for (int j = 0; j < points_moyen_accumulateur.size(); j++) {
						double dist = abs(centre_ligne.y - (points_moyen_accumulateur[j][0] / points_moyen_accumulateur[j][1]));
						if (dist < min_dist) {
							min_dist = dist;
							i_min_dist = j;
						}
					}
					if (min_dist > real_box_h * 0.9) {
						vector<double> new_point = { 1.0 * centre_ligne.y, 1, theta };
						points_moyen_accumulateur.push_back(new_point);
					}
					else {
						points_moyen_accumulateur[i_min_dist][0] += centre_ligne.y;
						points_moyen_accumulateur[i_min_dist][1] ++;
						points_moyen_accumulateur[i_min_dist][2] += theta;
					}
				}
			}
			vector<double> points_moyen;
			for (int j = 0; j < points_moyen_accumulateur.size(); j++)
			{
				points_moyen.push_back(points_moyen_accumulateur[j][0] / points_moyen_accumulateur[j][1]);
				//line(drawing, Point(0, points_moyen_accumulateur[j][0] / points_moyen_accumulateur[j][1]), Point(5000, points_moyen_accumulateur[j][0] / points_moyen_accumulateur[j][1]), Scalar(255, 0, 255), 3, LINE_AA);
			}
			sort(points_moyen.begin(), points_moyen.end());

			//imwrite("hough.png", drawing);

			for (int j = 0; j < points_moyen.size(); j++) {
				int point_moyen_y = points_moyen[j];

				//line(drawing, Point(0,point_moyen_y), Point(5000, point_moyen_y), Scalar(255, 0, 255), 3, LINE_AA);
				Mat ligne = src(Rect(Point(0, point_moyen_y - real_box_h), Point(src.cols, point_moyen_y)));
				Mat canny_ligne = canny_output(Rect(Point(0, point_moyen_y - real_box_h), Point(src.cols, point_moyen_y)));

				namedWindow("Res canny ligne", WINDOW_NORMAL);
				imshow("Res canny ligne", canny_output);

				vector<int> bords_ligne;

				vector<Vec2f> verticals;
				HoughLines(canny_ligne, verticals, 1, CV_PI / 180, real_box_h / 3);
				for (size_t i = 0; i < verticals.size(); i++)
				{
					float rho = verticals[i][0];
					float theta = verticals[i][1];
					double a = cos(theta), b = sin(theta);
					double x0 = a * rho, y0 = b * rho;
					if (theta > 0 - CV_PI / 64 && theta < 0 + CV_PI / 64) {
						//line(drawing, Point(x0, point_moyen_y - real_box_h), Point(x0, point_moyen_y), Scalar(0, 0, 255), 3, LINE_AA);
						bords_ligne.push_back(x0);
					}
				}
				bords_ligne.push_back(ligne.cols - 1);
				bords_ligne.push_back(1);
				sort(bords_ligne.begin(), bords_ligne.end());
				for (int i = bords_ligne.size() / 2; i < bords_ligne.size() - 1; i++)
				{
					//line(drawing, Point(bords_ligne[i], point_moyen_y - real_box_h), Point(bords_ligne[i], point_moyen_y), Scalar(0, 0, 255), 3, LINE_AA);
					if (bords_ligne[i + 1] - bords_ligne[i] > real_box_w * 1.2) {

						//line(drawing, Point(bords_ligne[i] + real_box_w, point_moyen_y - real_box_h), Point(bords_ligne[i] + real_box_w, point_moyen_y), Scalar(0, 0, 255), 3, LINE_AA);
						bords_ligne.insert(bords_ligne.begin() + i + 1, bords_ligne[i] + real_box_w);
					}
				}
				for (int i = bords_ligne.size() / 2; i > 0; i--)
				{
					//line(drawing, Point(bords_ligne[i], point_moyen_y - real_box_h), Point(bords_ligne[i], point_moyen_y), Scalar(0, 0, 255), 3, LINE_AA);
					if (bords_ligne[i] - bords_ligne[i - 1] > real_box_w * 1.2) {

						//line(drawing, Point(bords_ligne[i] - real_box_w, point_moyen_y - real_box_h), Point(bords_ligne[i] - real_box_w, point_moyen_y), Scalar(0, 0, 255), 3, LINE_AA);
						bords_ligne.insert(bords_ligne.begin() + i, bords_ligne[i] - real_box_w);
						i++;
					}
				}

				for (int i = 0; i < bords_ligne.size() - 1; i++)
				{
					if (bords_ligne[i + 1] - bords_ligne[i] > real_box_w * 0.8 && bords_ligne[i + 1] - bords_ligne[i] < real_box_w * 1.2) {
						//rectangle(drawing, Rect(Point(bords_ligne[i], point_moyen_y - real_box_h), Point(bords_ligne[i+1], point_moyen_y)), Scalar(0, 255, 0), 5);
						Mat planche = src(Rect(Point(bords_ligne[i], point_moyen_y - real_box_h), Point(bords_ligne[i + 1], point_moyen_y)));
						Mat canny_planche = canny_output(Rect(Point(bords_ligne[i], point_moyen_y - real_box_h), Point(bords_ligne[i + 1], point_moyen_y)));

						vector<Vec2f> horizontales;
						HoughLines(canny_planche, horizontales, 1, CV_PI / 180, real_box_w / 4);
						int max_decalage = 0, nb_max_decalage = 0, min_decalage = 0, nb_min_decalage = 0;
						for (size_t j = 0; j < horizontales.size(); j++)
						{
							float rho = horizontales[j][0];
							float theta = horizontales[j][1];
							double a = cos(theta), b = sin(theta);
							double x0 = a * rho, y0 = b * rho;
							if (theta > CV_PI / 2 - CV_PI / 64 && theta < CV_PI / 2 + CV_PI / 64) {
								//line(drawing, Point(bords_ligne[i], point_moyen_y - real_box_h + y0), Point(bords_ligne[i + 1], point_moyen_y - real_box_h + y0), Scalar(255, 0, 0), 3, LINE_AA);
								//line(planche, Point(0, y0), Point(planche.cols, y0), Scalar(255, 0, 0), 3, LINE_AA);

								if (y0 > real_box_h / 2 && y0 > max_decalage) {
									max_decalage += y0;
									nb_max_decalage++;
									//cout << max_decalage << endl;
								}
								if (y0 < real_box_h / 2 && y0 < min_decalage) {
									min_decalage += y0;
									nb_min_decalage++;
								}

								namedWindow("planche", WINDOW_NORMAL);
								imshow("planche", planche);
								//waitKey(0);
							}
						}
						if (nb_max_decalage > 0 && nb_max_decalage > nb_min_decalage) {
							max_decalage = max_decalage / nb_max_decalage;
							rectangle(drawing, Rect(Point(bords_ligne[i], point_moyen_y - real_box_h - (real_box_h - max_decalage)), Point(bords_ligne[i + 1], point_moyen_y - (real_box_h - max_decalage))), Scalar(0, 255, 0), 5);

							planche = src(Rect(Point(bords_ligne[i], point_moyen_y - real_box_h - (real_box_h - max_decalage)), Point(bords_ligne[i + 1], point_moyen_y - (real_box_h - max_decalage))));
						}
						else if (nb_min_decalage > 0) {
							min_decalage = min_decalage / nb_min_decalage;
							cout << min_decalage << endl;
							rectangle(drawing, Rect(Point(bords_ligne[i], point_moyen_y - real_box_h + min_decalage), Point(bords_ligne[i + 1], point_moyen_y + min_decalage)), Scalar(0, 255, 0), 5);

							planche = src(Rect(Point(bords_ligne[i], point_moyen_y - real_box_h + min_decalage), Point(bords_ligne[i + 1], point_moyen_y + min_decalage)));
						}
						else {
							rectangle(drawing, Rect(Point(bords_ligne[i], point_moyen_y - real_box_h), Point(bords_ligne[i + 1], point_moyen_y)), Scalar(0, 255, 0), 5);

							planche = src(Rect(Point(bords_ligne[i], point_moyen_y - real_box_h), Point(bords_ligne[i + 1], point_moyen_y)));
						}

						imwrite("planches/p" + to_string(num_planche++) + ".png", planche);
					}
				}
				namedWindow("Res hough", WINDOW_NORMAL);
				imshow("Res hough", drawing);
			}

			namedWindow("Res hough", WINDOW_NORMAL);
			imshow("Res hough", drawing);
			//imwrite("test_class.jpg", drawing);

			namedWindow("src", WINDOW_NORMAL);
			imshow("src", src);

			time(&end);
			cout << "Temps : " << double(end - start) << "s" << endl;
		}
		// Transformée de fourrier 
		if (key == 'f') {

			//src = imread("Paquet 3/planks-color/plank-color-1/Arbre 5/p" + to_string(9) + ".png");
			//src = imread("planches/p" + to_string(65) + ".png");
			src = imread(images_billon[1]);
			morphologyEx(src, src, MORPH_GRADIENT, getStructuringElement(CV_SHAPE_RECT, Size(3, 3)));
			//threshold(src, src, 30, 255, CV_THRESH_BINARY);
			namedWindow("Src fft", WINDOW_NORMAL);
			imshow("Src fft", src);    // Show the result
			cvtColor(src, src, COLOR_BGR2GRAY);

			Mat padded;                            //expand input image to optimal size
			int m = getOptimalDFTSize(src.rows);
			int n = getOptimalDFTSize(src.cols); // on the border add zero values
			copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, BORDER_CONSTANT, Scalar::all(0));

			Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
			Mat complexI;
			merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

			dft(complexI, complexI);            // this way the result may fit in the source matrix

			// compute the magnitude and switch to logarithmic scale
			// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
			split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
			magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
			Mat magI = planes[0];

			magI += Scalar::all(1);                    // switch to logarithmic scale
			log(magI, magI);

			// crop the spectrum, if it has an odd number of rows or columns
			magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

			// rearrange the quadrants of Fourier image  so that the origin is at the image center
			int cx = magI.cols / 2;
			int cy = magI.rows / 2;

			Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
			Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
			Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
			Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

			Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
			q0.copyTo(tmp);
			q3.copyTo(q0);
			tmp.copyTo(q3);

			q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
			q2.copyTo(q1);
			tmp.copyTo(q2);

			normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
													// viewable image form (float between values 0 and 1).

			namedWindow("Src fft", WINDOW_NORMAL);
			//imshow("Src fft", src);    // Show the result
			namedWindow("fft result", WINDOW_NORMAL);
			imshow("fft result", magI);
			waitKey();
		}
		// surf
		if (key == 'g') {
			struct feature {
				// Position de la feature
				Point pos;
				// Octave de la feature
				int s;
				// Descripteur de la feature
				double sum_dx, sum_dx_abs, sum_dy, sum_dy_abs;
			};
			// Recuperation des images 
			Mat planche = imread("Paquet 3/planks-color/plank-color-1/Arbre 5/p" + to_string(50) + ".png");
			//Mat planche = imread("planches/p" + to_string(65) + ".png");
			//Mat billon = imread(images_billon[1]);
			Mat billon = imread("A01c.jpeg");
			//morphologyEx(billon, billon, MORPH_GRADIENT, getStructuringElement(CV_SHAPE_RECT, Size(3, 3)));
			//morphologyEx(planche, planche, MORPH_GRADIENT, getStructuringElement(CV_SHAPE_RECT, Size(3, 3)));
			// Passage en niveau de gris 
			cvtColor(billon, billon, COLOR_BGR2GRAY);
			cvtColor(planche, planche, COLOR_BGR2GRAY);

			// ------------------------------------------------------------------------------------------------------- //
			// ------------------------------------------- Detect features ------------------------------------------- //
			// Calcul des images integrales 
			Mat integral_billon(billon.size(), CV_32F), integral_planche(planche.size(), CV_32F);
			integral(billon, integral_billon, CV_32F);
			integral(planche, integral_planche, CV_32F);
			cout << "Images integrales fini" << endl;

			// Recuperation des features 
			vector<int> firstOctave_filtersSize = { 9,15,21,27 };
			//vector<int> firstOctave_filtersSize = { 15,27,39,51 };
			//vector<int> firstOctave_filtersSize = { 27,51,75,99 };
			vector<Mat> firstOctave_blob_billon, firstOctave_blob_planche;
			vector<Mat> firstOctave_maxLoc_billon, firstOctave_maxLoc_planche;
			for (size_t indice_filterSize = 0; indice_filterSize < firstOctave_filtersSize.size(); indice_filterSize++)
			{
				int t = firstOctave_filtersSize[indice_filterSize];
				Mat surf_hessian_billon = Mat::zeros(billon.size(), CV_32F), surf_hessian_planche = Mat::zeros(planche.size(), CV_32F);

				// Calcul pour l'image du billon
				for (size_t i = (t - 1) / 2; i < billon.rows - (t - 1) / 2; i++)
				{
					for (size_t j = (t - 1) / 2; j < billon.cols - (t - 1) / 2; j++)
					{
						double lxx = 0
							+ 1 * (integral_billon.at<float>(i + (t + 3) / 6, j - (t + 3) / 6)  // D
								+ integral_billon.at<float>(i - (t + 3) / 6, j - (t - 1) / 2)   // A
								- integral_billon.at<float>(i + (t + 3) / 6, j - (t - 1) / 2)   // B 
								- integral_billon.at<float>(i - (t + 3) / 6, j - (t + 3) / 6))  // C
							+ -2 * (integral_billon.at<float>(i + (t + 3) / 6, j + (t - 3) / 6) // H
								+ integral_billon.at<float>(i - (t + 3) / 6, j - (t - 3) / 6)   // E
								- integral_billon.at<float>(i + (t + 3) / 6, j - (t - 3) / 6)   // F
								- integral_billon.at<float>(i - (t + 3) / 6, j + (t - 3) / 6))  // G
							+ 1 * (integral_billon.at<float>(i + (t + 3) / 6, j + (t - 1) / 2)  // L
								+ integral_billon.at<float>(i - (t + 3) / 6, j + (t + 3) / 6)   // I
								- integral_billon.at<float>(i + (t + 3) / 6, j + (t + 3) / 6)   // J
								- integral_billon.at<float>(i - (t + 3) / 6, j + (t - 1) / 2)); // K
						double lyy = 0
							+ 1 * (integral_billon.at<float>(i - (t + 3) / 6, j + (t + 3) / 6)  // D 
								+ integral_billon.at<float>(i - (t - 1) / 2, j - (t + 3) / 6)   // A 
								- integral_billon.at<float>(i - (t - 1) / 2, j + (t + 3) / 6)   // B 
								- integral_billon.at<float>(i - (t + 3) / 6, j - (t + 3) / 6))  // C 
							+ -2 * (integral_billon.at<float>(i + (t - 3) / 6, j + (t + 3) / 6) // H 
								+ integral_billon.at<float>(i - (t - 3) / 6, j - (t + 3) / 6)   // E 
								- integral_billon.at<float>(i - (t - 3) / 6, j + (t + 3) / 6)   // F 
								- integral_billon.at<float>(i + (t - 3) / 6, j - (t + 3) / 6))  // G 
							+ 1 * (integral_billon.at<float>(i + (t - 1) / 2, j + (t + 3) / 6)  // L 
								+ integral_billon.at<float>(i + (t + 3) / 6, j - (t + 3) / 6)   // I 
								- integral_billon.at<float>(i + (t + 3) / 6, j + (t + 3) / 6)   // J 
								- integral_billon.at<float>(i + (t - 1) / 2, j - (t + 3) / 6)); // K 
						double lxy = 0
							+ 1 * (integral_billon.at<float>(i - 1, j - 1)                      // D 
								+ integral_billon.at<float>(i - t / 3, j - t / 3)               // A 
								- integral_billon.at<float>(i - 1, j - t / 3)                   // B 
								- integral_billon.at<float>(i - t / 3, j - 1))                  // C 
							+ -1 * (integral_billon.at<float>(i + t / 3, j - 1)                 // H 
								+ integral_billon.at<float>(i + 1, j - t / 3)                   // E 
								- integral_billon.at<float>(i + t / 3, j - t / 3)               // F 
								- integral_billon.at<float>(i + 1, j - 1))                      // G 
							+ 1 * (integral_billon.at<float>(i - 1, j + t / 3)                  // L 
								+ integral_billon.at<float>(i - t / 3, j + 1)                   // I 
								- integral_billon.at<float>(i - 1, j + 1)                       // J 
								- integral_billon.at<float>(i - t / 3, j + t / 3))              // K 
							+ -1 * (integral_billon.at<float>(i + t / 3, j + t / 3)             // P 
								+ integral_billon.at<float>(i + 1, j + 1)                       // M 
								- integral_billon.at<float>(i + t / 3, j + 1)                   // N
								- integral_billon.at<float>(i + 1, j + t / 3));                 // O 
						surf_hessian_billon.at<float>(i, j) = lxx * lyy - lxy * lxy;
						//cout << lxx << " , " << lyy << " , " << lxy << " , " << lxx * lyy - lxy * lxy << " , " << integral_billon.at<float>(i + 2, j - 2) << endl;
					}
				}

				// Calcul pour l'image de la planche
				for (size_t i = (t - 1) / 2; i < planche.rows - (t - 1) / 2; i++)
				{
					for (size_t j = (t - 1) / 2; j < planche.cols - (t - 1) / 2; j++)
					{
						double lxx = 0
							+ 1 * (integral_planche.at<float>(i + (t + 3) / 6, j - (t + 3) / 6)  // D
								+ integral_planche.at<float>(i - (t + 3) / 6, j - (t - 1) / 2)   // A
								- integral_planche.at<float>(i + (t + 3) / 6, j - (t - 1) / 2)   // B 
								- integral_planche.at<float>(i - (t + 3) / 6, j - (t + 3) / 6))  // C
							+ -2 * (integral_planche.at<float>(i + (t + 3) / 6, j + (t - 3) / 6) // H
								+ integral_planche.at<float>(i - (t + 3) / 6, j - (t - 3) / 6)   // E
								- integral_planche.at<float>(i + (t + 3) / 6, j - (t - 3) / 6)   // F
								- integral_planche.at<float>(i - (t + 3) / 6, j + (t - 3) / 6))  // G
							+ 1 * (integral_planche.at<float>(i + (t + 3) / 6, j + (t - 1) / 2)  // L
								+ integral_planche.at<float>(i - (t + 3) / 6, j + (t + 3) / 6)   // I
								- integral_planche.at<float>(i + (t + 3) / 6, j + (t + 3) / 6)   // J
								- integral_planche.at<float>(i - (t + 3) / 6, j + (t - 1) / 2)); // K
						double lyy = 0
							+ 1 * (integral_planche.at<float>(i - (t + 3) / 6, j + (t + 3) / 6)  // D 
								+ integral_planche.at<float>(i - (t - 1) / 2, j - (t + 3) / 6)   // A 
								- integral_planche.at<float>(i - (t - 1) / 2, j + (t + 3) / 6)   // B 
								- integral_planche.at<float>(i - (t + 3) / 6, j - (t + 3) / 6))  // C 
							+ -2 * (integral_planche.at<float>(i + (t - 3) / 6, j + (t + 3) / 6) // H 
								+ integral_planche.at<float>(i - (t - 3) / 6, j - (t + 3) / 6)   // E 
								- integral_planche.at<float>(i - (t - 3) / 6, j + (t + 3) / 6)   // F 
								- integral_planche.at<float>(i + (t - 3) / 6, j - (t + 3) / 6))  // G 
							+ 1 * (integral_planche.at<float>(i + (t - 1) / 2, j + (t + 3) / 6)  // L 
								+ integral_planche.at<float>(i + (t + 3) / 6, j - (t + 3) / 6)   // I 
								- integral_planche.at<float>(i + (t + 3) / 6, j + (t + 3) / 6)   // J 
								- integral_planche.at<float>(i + (t - 1) / 2, j - (t + 3) / 6)); // K 
						double lxy = 0
							+ 1 * (integral_planche.at<float>(i - 1, j - 1)                      // D 
								+ integral_planche.at<float>(i - t / 3, j - t / 3)               // A 
								- integral_planche.at<float>(i - 1, j - t / 3)                   // B 
								- integral_planche.at<float>(i - t / 3, j - 1))                  // C 
							+ -1 * (integral_planche.at<float>(i + t / 3, j - 1)                 // H 
								+ integral_planche.at<float>(i + 1, j - t / 3)                   // E 
								- integral_planche.at<float>(i + t / 3, j - t / 3)               // F 
								- integral_planche.at<float>(i + 1, j - 1))                      // G 
							+ 1 * (integral_planche.at<float>(i - 1, j + t / 3)                  // L 
								+ integral_planche.at<float>(i - t / 3, j + 1)                   // I 
								- integral_planche.at<float>(i - 1, j + 1)                       // J 
								- integral_planche.at<float>(i - t / 3, j + t / 3))              // K 
							+ -1 * (integral_planche.at<float>(i + t / 3, j + t / 3)             // P 
								+ integral_planche.at<float>(i + 1, j + 1)                       // M 
								- integral_planche.at<float>(i + t / 3, j + 1)                   // N
								- integral_planche.at<float>(i + 1, j + t / 3));                 // O 
						surf_hessian_planche.at<float>(i, j) = lxx * lyy - lxy * lxy;
						//cout << lxx << " , " << lyy << " , " << lxy << " , " << lxx * lyy - lxy * lxy << " , " << integral_billon.at<float>(i + 2, j - 2) << endl;
					}
				}

				firstOctave_blob_billon.push_back(surf_hessian_billon);
				firstOctave_blob_planche.push_back(surf_hessian_planche);

				cout << t << endl;
			}
			cout << "Calcul des blobs fini" << endl;
			// Recuperations des differences de gaussiennes 
			vector<Mat> DoGs_billon, DoGs_planche;
			for (size_t i = 1; i < firstOctave_filtersSize.size(); i++)
			{
				Mat DoG_billon, DoG_planche;
				subtract(firstOctave_blob_billon[i - 1], firstOctave_blob_billon[i], DoG_billon);
				DoGs_billon.push_back(DoG_billon);
				subtract(firstOctave_blob_planche[i - 1], firstOctave_blob_planche[i], DoG_planche);
				DoGs_planche.push_back(DoG_planche);
			}
			// Initialisation des images pour aficher les features 
			Mat draw_surf_features_billon = Mat::zeros(billon.size(), CV_8U);
			Mat draw_surf_features_planche = Mat::zeros(planche.size(), CV_8U);
			// Recuperation des features du billon 
			vector<feature> surf_features_billon;
			double min_billon, max_billon;
			minMaxLoc(DoGs_billon[1], &min_billon, &max_billon);
			for (int i = 1; i < DoGs_billon[1].rows - 1; i++)
			{
				for (int j = 1; j < DoGs_billon[1].cols - 1; j++)
				{
					float centre = DoGs_billon[1].at<float>(i, j);
					if (centre <= 0) continue;
					// On test des voisin du meme plan 
					if (centre <= DoGs_billon[1].at<float>(i - 1, j - 1)) continue;
					if (centre <= DoGs_billon[1].at<float>(i - 1, j)) continue;
					if (centre <= DoGs_billon[1].at<float>(i - 1, j + 1)) continue;
					if (centre <= DoGs_billon[1].at<float>(i, j - 1)) continue;
					if (centre <= DoGs_billon[1].at<float>(i, j + 1)) continue;
					if (centre <= DoGs_billon[1].at<float>(i + 1, j - 1)) continue;
					if (centre <= DoGs_billon[1].at<float>(i + 1, j)) continue;
					if (centre <= DoGs_billon[1].at<float>(i + 1, j + 1)) continue;
					// Les voisins du dessus 
					if (centre <= DoGs_billon[2].at<float>(i - 1, j - 1)) continue;
					if (centre <= DoGs_billon[2].at<float>(i - 1, j)) continue;
					if (centre <= DoGs_billon[2].at<float>(i - 1, j + 1)) continue;
					if (centre <= DoGs_billon[2].at<float>(i, j - 1)) continue;
					if (centre <= DoGs_billon[2].at<float>(i, j)) continue;
					if (centre <= DoGs_billon[2].at<float>(i, j + 1)) continue;
					if (centre <= DoGs_billon[2].at<float>(i + 1, j - 1)) continue;
					if (centre <= DoGs_billon[2].at<float>(i + 1, j)) continue;
					if (centre <= DoGs_billon[2].at<float>(i + 1, j + 1)) continue;
					// Les voisins du dessous 
					if (centre <= DoGs_billon[0].at<float>(i - 1, j - 1)) continue;
					if (centre <= DoGs_billon[0].at<float>(i - 1, j)) continue;
					if (centre <= DoGs_billon[0].at<float>(i - 1, j + 1)) continue;
					if (centre <= DoGs_billon[0].at<float>(i, j - 1)) continue;
					if (centre <= DoGs_billon[0].at<float>(i, j)) continue;
					if (centre <= DoGs_billon[0].at<float>(i, j + 1)) continue;
					if (centre <= DoGs_billon[0].at<float>(i + 1, j - 1)) continue;
					if (centre <= DoGs_billon[0].at<float>(i + 1, j)) continue;
					if (centre <= DoGs_billon[0].at<float>(i + 1, j + 1)) continue;
					feature f;
					f.pos = Point(i, j);
					f.s = 1;
					surf_features_billon.push_back(f);
					//cout << centre / max * 255 << endl;
					circle(draw_surf_features_billon, Point(j, i), (int)(centre / max_billon * 255), 255);
				}
			}
			cout << "Features Billon fini" << endl;
			// Recuperation des features de la planche
			vector<feature> surf_features_planche;
			double min_planche, max_planche;
			minMaxLoc(DoGs_planche[1], &min_planche, &max_planche);
			for (int i = 1; i < DoGs_planche[1].rows - 1; i++)
			{
				for (int j = 1; j < DoGs_planche[1].cols - 1; j++)
				{
					float centre = DoGs_planche[1].at<float>(i, j);
					if (centre <= 0) continue;
					// On test des voisin du meme plan 
					if (centre <= DoGs_planche[1].at<float>(i - 1, j - 1)) continue;
					if (centre <= DoGs_planche[1].at<float>(i - 1, j)) continue;
					if (centre <= DoGs_planche[1].at<float>(i - 1, j + 1)) continue;
					if (centre <= DoGs_planche[1].at<float>(i, j - 1)) continue;
					if (centre <= DoGs_planche[1].at<float>(i, j + 1)) continue;
					if (centre <= DoGs_planche[1].at<float>(i + 1, j - 1)) continue;
					if (centre <= DoGs_planche[1].at<float>(i + 1, j)) continue;
					if (centre <= DoGs_planche[1].at<float>(i + 1, j + 1)) continue;
					// Les voisins du dessus 
					if (centre <= DoGs_planche[2].at<float>(i - 1, j - 1)) continue;
					if (centre <= DoGs_planche[2].at<float>(i - 1, j)) continue;
					if (centre <= DoGs_planche[2].at<float>(i - 1, j + 1)) continue;
					if (centre <= DoGs_planche[2].at<float>(i, j - 1)) continue;
					if (centre <= DoGs_planche[2].at<float>(i, j)) continue;
					if (centre <= DoGs_planche[2].at<float>(i, j + 1)) continue;
					if (centre <= DoGs_planche[2].at<float>(i + 1, j - 1)) continue;
					if (centre <= DoGs_planche[2].at<float>(i + 1, j)) continue;
					if (centre <= DoGs_planche[2].at<float>(i + 1, j + 1)) continue;
					// Les voisins du dessous 
					if (centre <= DoGs_planche[0].at<float>(i - 1, j - 1)) continue;
					if (centre <= DoGs_planche[0].at<float>(i - 1, j)) continue;
					if (centre <= DoGs_planche[0].at<float>(i - 1, j + 1)) continue;
					if (centre <= DoGs_planche[0].at<float>(i, j - 1)) continue;
					if (centre <= DoGs_planche[0].at<float>(i, j)) continue;
					if (centre <= DoGs_planche[0].at<float>(i, j + 1)) continue;
					if (centre <= DoGs_planche[0].at<float>(i + 1, j - 1)) continue;
					if (centre <= DoGs_planche[0].at<float>(i + 1, j)) continue;
					if (centre <= DoGs_planche[0].at<float>(i + 1, j + 1)) continue;
					feature f;
					f.pos = Point(i, j);
					f.s = 1;
					surf_features_planche.push_back(f);
					//cout << centre << endl;
					circle(draw_surf_features_planche, Point(j, i), (int)(centre / max_planche * 25), 255);
				}
			}
			cout << "Features planche fini" << endl;

			cout << surf_features_billon.size() << " , " << surf_features_planche.size() << endl;

			// ------------------------------------------------------------------------------------------------------- //


			// ------------------------------------------------------------------------------------------------------- //
			// -------------------------- Recuperation des descripteurs des features --------------------------------- //
			// On itere sur chaque features 
			for (int indice_feature = 0; indice_feature < surf_features_planche.size(); indice_feature++)
			{
				feature f = surf_features_planche[indice_feature];
				Point p = f.pos;
				cout << " ------------ p :" << p << endl;
				// Initialisation de l'histogram des gradient oriente 
				vector<double> HOG;
				for (int i = 0; i < 36; i++)
				{
					HOG.push_back(0);
				}
				// Calcul de l'histogram des gradient oriente 
				for (int i = p.x - 10; i < p.x + 10; i++)
				{
					for (int j = p.y - 10; j < p.y + 10; j++)
					{
						// Si on est dans le cercle de taille 6s (s taille de detection de la feature) on ajoute a l'histogramme sinon on passe au point suivant 
						if (sqrt((p.x - i) * (p.x - i) + (p.y - j) * (p.y - j) > 6)) continue;
						double haar_x =
							integral_planche.at<float>(i + 3, j + 3)
							+ integral_planche.at<float>(i + 2, j)
							- integral_planche.at<float>(i + 2, j + 3)
							- integral_planche.at<float>(i + 3, j)
							- (
								integral_planche.at<float>(i + 1, j + 3)
								+ integral_planche.at<float>(i, j)
								- integral_planche.at<float>(i, j + 3)
								- integral_planche.at<float>(i + 1, j));
						double haar_y =
							integral_planche.at<float>(i + 3, j + 1)
							+ integral_planche.at<float>(i, j)
							- integral_planche.at<float>(i + 3, j)
							- integral_planche.at<float>(i, j + 1)
							- (
								integral_planche.at<float>(i + 3, j + 3)
								+ integral_planche.at<float>(i, j + 2)
								- integral_planche.at<float>(i + 3, j + 2)
								- integral_planche.at<float>(i, j + 3));


						//cout << "angle : " << atan2(haar_x, haar_y) * 180 / CV_PI << endl;
						// Ajout de l'angle a l'histogramme 
						int angle = (int)((atan2(haar_x, haar_y) * 180 / CV_PI) + 360) % 360;
						HOG[(int)(angle / 10)] += sqrt(haar_x * haar_x + haar_y * haar_y);
					}
				}
				// Recuperation du maximum de l'histogramme pour avoir la meilleure orientation pour notre feature 
				int max = 0, imax = 0;
				for (int i = 0; i < HOG.size(); i++)
				{
					//cout << HOG[i] << " , ";
					if (HOG[i] > max) {
						max = HOG[1];
						imax = i;
					}
				}
				//cout << "   Max = " << (imax * CV_PI / 36) * 180 / CV_PI << endl;

				// Recuperation de la region orientee pour recuperar le descripteur de la feature 
				RotatedRect rect(Point2f(p.y, p.x), Size(20, 20), (imax * CV_PI / 36) * 180 / CV_PI);
				Mat M, rotated, voisinage_rotated;
				float angle = rect.angle;
				Size rect_size = rect.size;
				if (rect.angle < -45.) {
					angle += 90.0;
					swap(rect_size.width, rect_size.height);
				}
				M = getRotationMatrix2D(rect.center, angle, 1.0);
				warpAffine(planche, rotated, M, planche.size(), INTER_CUBIC);
				getRectSubPix(rotated, rect_size, rect.center, voisinage_rotated);

				// Recuperation de la feature 
				Mat integral_voisinage(planche.size(), CV_32F);
				integral(voisinage_rotated, integral_voisinage, CV_32F);
				double sum_haar_x = 0, sum_haar_y = 0, sum_haar_x_abs = 0, sum_haar_y_abs = 0;
				// Division du voisinage en 16 patch
				for (int i = 0; i < voisinage_rotated.rows; i += 5)
				{
					for (int j = 0; j < voisinage_rotated.cols; j += 5)
					{
						// Calcul des reponse au vaguelettes de haar dans le patch
						for (int i2 = 0; i2 < 2; i2++)
						{
							for (int j2 = 0; j2 < 2; j2++)
							{
								double haar_x =
									integral_voisinage.at<float>(i + i2 + 3, j + j2 + 3)
									+ integral_voisinage.at<float>(i + i2 + 2, j + j2)
									- integral_voisinage.at<float>(i + i2 + 2, j + j2 + 3)
									- integral_voisinage.at<float>(i + i2 + 3, j + j2)
									- (
										integral_voisinage.at<float>(i + i2 + 1, j + j2 + 3)
										+ integral_voisinage.at<float>(i, j + j2)
										- integral_voisinage.at<float>(i, j + j2 + 3)
										- integral_voisinage.at<float>(i + i2 + 1, j + j2));
								double haar_y =
									integral_voisinage.at<float>(i + i2 + 3, j + j2 + 1)
									+ integral_voisinage.at<float>(i, j + j2)
									- integral_voisinage.at<float>(i + i2 + 3, j + j2)
									- integral_voisinage.at<float>(i, j + j2 + 1)
									- (
										integral_voisinage.at<float>(i + i2 + 3, j + j2 + 3)
										+ integral_voisinage.at<float>(i, j + j2 + 2)
										- integral_voisinage.at<float>(i + i2 + 3, j + j2 + 2)
										- integral_voisinage.at<float>(i, j + j2 + 3));
								sum_haar_x += haar_x;
								sum_haar_y += haar_y;
								sum_haar_x_abs += abs(haar_x);
								sum_haar_y_abs += abs(haar_y);
							}
						}
					}
				}
				f.sum_dx = sum_haar_x;
				f.sum_dx_abs = sum_haar_x_abs;
				f.sum_dy = sum_haar_y;
				f.sum_dy_abs = sum_haar_y_abs;
				cout << "Sdx = " << sum_haar_x << " , S|dx| = " << sum_haar_x_abs << " , Sdy = " << sum_haar_y << " , S|dy| = " << sum_haar_y_abs << endl;

			}

			namedWindow("sfb", WINDOW_NORMAL);
			imshow("sfb", draw_surf_features_billon);
			imwrite("sfb.jpg", draw_surf_features_billon);
			namedWindow("sfp", WINDOW_NORMAL);
			imshow("sfp", draw_surf_features_planche);
			imwrite("sfp.jpg", draw_surf_features_planche);

		}
		// Matching rotation planche mask best matching choix
		if (key == ',') {
			bool use_mask;
			Mat img; Mat templ; Mat mask; Mat result;
			const char* image_window = "Source Image";
			const char* result_window = "Result window";
			int match_method = CV_TM_SQDIFF;
			int max_Trackbar = 5;
			//img = imread("Paquet 3/planks-color/plank-color-1/Arbre 1/A05a.jpeg");
			img = imread("Paquet 3/planks-color/plank-color-1/Arbre 5/A01c2.jpeg");
			//templ = imread("Paquet 3/planks-color/plank-color-1/Arbre 6/p48.png");
			//img = imread("Paquet 3/planks-color/plank-color-1/Arbre 6/A05c.jpeg");
			// Copie de l'image pour l'affichage du resultat 
			Mat img_display;
			img.copyTo(img_display);

			double start_scale_resolution = 0.25, scale_resolution = start_scale_resolution;

			for (int numplanche = 49; numplanche < 59; numplanche++)
			{
				double minOfMin_val = DBL_MAX;
				Point minOfMin_loc;
				int minOfMin_angle = 0;
				Mat minOfMin_templ = templ;
				Mat minOfMin_mask = Mat::zeros(templ.size(), CV_8U);
				Mat minOfMin_result = result;
				for (int indice_canal_RGB = 0; indice_canal_RGB < 1; indice_canal_RGB++)
				{
					scale_resolution = start_scale_resolution;
					vector<tuple<int, int>> liste_angles;
					for (int i = 0; i < 360; i += 15)
					{
						liste_angles.push_back(make_tuple(0, i));
					}
					for (int scale_step = 0; scale_step < 3; scale_step++)
					{
						int min_nb_pixel = INT_MAX, max_nb_pixel = 0;
						//img = imread("Paquet 3/planks-color/plank-color-1/Arbre 1/A05a.jpeg");
						img = imread("Paquet 3/planks-color/plank-color-1/Arbre 5/A01c2.jpeg");
						//img = imread("Paquet 3/planks-color/plank-color-1/Arbre 6/A05c.jpeg");
						// Diminution de la resolution de l'image source 
						resize(img, img, Size(0, 0), scale_resolution, scale_resolution);

						//morphologyEx(img, img, MORPH_GRADIENT, getStructuringElement(CV_SHAPE_RECT, Size(3, 3)));
						//threshold(img, img, 20, 255, CV_THRESH_BINARY);
						namedWindow("la source pour le screen", WINDOW_NORMAL);
						imshow("la source pour le screen", img);

						//templ = imread("Paquet 3/planks-color/plank-color-1/Arbre 1/p" + to_string(numplanche) + ".png");
						templ = imread("Paquet 3/planks-color/plank-color-1/Arbre 5/p" + to_string(numplanche) + ".png");
						//templ = imread("Paquet 3/planks-color/plank-color-1/Arbre 6/p" + to_string(numplanche) + ".png");
						// Mise a l'echelle planche billon 
						//resize(templ, templ, Size(0, 0), 2.16, 2.16); // arbre 1
						resize(templ, templ, Size(0, 0), 2.38, 2.38); // arbre 5
						//resize(templ, templ, Size(0, 0), 3.14, 3.14); // arbre 6

						// Diminution de la resolution de la planche 
						resize(templ, templ, Size(0, 0), scale_resolution, scale_resolution);

						//cout << scale_resolution << endl;
						//morphologyEx(templ, templ, MORPH_GRADIENT, getStructuringElement(CV_SHAPE_RECT, Size(3, 3)));
						//threshold(templ, templ, 20, 255, CV_THRESH_BINARY);


						cvtColor(img, img, COLOR_BGR2Lab);
						cvtColor(templ, templ, COLOR_BGR2Lab);
						vector<Mat> templ_planes;
						split(templ, templ_planes);
						vector<Mat> img_planes;
						split(img, img_planes);
						// LAB 
						templ_planes[0] = 0;
						img_planes[0] = 0;
						merge(templ_planes, templ);
						merge(img_planes, img);
						// R G B 
						//templ = templ_planes[indice_canal_RGB];
						//img = img_planes[indice_canal_RGB];

						namedWindow("plan", WINDOW_NORMAL);
						namedWindow("img", WINDOW_NORMAL);
						imshow("plan", templ);
						imshow("img", img);

						//waitKey(0);
						//matchTemplate(img_planes[indice_canal_RGB], templ_planes[indice_canal_RGB], result, match_method);
						matchTemplate(img, templ, result, match_method);

						//normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

						double minVal; double maxVal; Point minLoc; Point maxLoc;
						Point matchLoc;
						minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

						matchLoc = minLoc;
						//cout << minVal << endl;

						double bestMatch_val = minVal;
						Point bestMatch_loc = matchLoc;
						int bestMatch_angle = 0;
						Mat bestMatch_templ = templ;
						Mat bestMatch_mask = Mat::zeros(templ.size(), CV_8U);
						Mat bestMatch_result = result;

						for (int indice_angle = 0; indice_angle < liste_angles.size(); indice_angle++)
						{
							int angle = get<1>(liste_angles[indice_angle]);
							Mat rot_templ;
							Point2f templ_center(templ.cols / 2.0F, templ.rows / 2.0F);
							Mat rot_mat = getRotationMatrix2D(templ_center, angle, 1.0);
							Rect2f bbox = cv::RotatedRect(cv::Point2f(), templ.size(), angle).boundingRect2f();
							// adjust transformation matrix
							rot_mat.at<double>(0, 2) += bbox.width / 2.0 - templ.cols / 2.0;
							rot_mat.at<double>(1, 2) += bbox.height / 2.0 - templ.rows / 2.0;
							warpAffine(templ, rot_templ, rot_mat, bbox.size());

							namedWindow("rot templ", WINDOW_NORMAL);
							imshow("rot templ", rot_templ);

							threshold(rot_templ, mask, 1, 255, THRESH_BINARY);

							namedWindow("rot templ mask", WINDOW_NORMAL);
							imshow("rot templ mask", mask);
							imwrite("rot_templ.jpg", rot_templ);
							imwrite("rot_templ_mask.jpg", mask);
							//cvtColor(mask, mask, COLOR_BGR2GRAY);
							//threshold(mask, mask, 1, 255, CV_THRESH_BINARY);
							//cvtColor(mask, mask, COLOR_GRAY2BGR);

							matchTemplate(img, rot_templ, result, match_method, mask);

							//normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

							double minVal; double maxVal; Point minLoc; Point maxLoc;
							Point matchLoc;
							minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

							matchLoc = minLoc;
							get<0>(liste_angles[indice_angle]) = minVal;
							cout << minVal << endl;
							cvtColor(mask, mask, COLOR_BGR2GRAY);
							minVal = minVal * ((countNonZero(mask) - 277134) / (279679 - 277134));
							cvtColor(mask, mask, COLOR_GRAY2BGR);
							if (minVal < bestMatch_val) {
								bestMatch_val = minVal;
								bestMatch_loc = matchLoc;
								bestMatch_angle = angle;
								bestMatch_templ = rot_templ;
								bestMatch_mask = mask;
								normalize(result, bestMatch_result, 0, 1, NORM_MINMAX, -1, Mat());
								//cout << "Min val = " << angle << endl;
							}
							if (scale_resolution == 1) {

								cvtColor(mask, mask, COLOR_BGR2GRAY);
								cout << angle << " : " << minVal << " nbn0 : " << countNonZero(mask) << " ratio : " << ((double)(templ.cols * templ.rows) / countNonZero(mask)) << endl;
								cvtColor(mask, mask, COLOR_GRAY2BGR);
								//rot_templ.copyTo(img_display.rowRange(matchLoc.y, matchLoc.y + rot_templ.rows).colRange(matchLoc.x, matchLoc.x + rot_templ.cols), mask);
							}
						}

						sort(liste_angles.begin(), liste_angles.end());
						liste_angles.erase(liste_angles.begin() + liste_angles.size() / 4, liste_angles.end());

						//cout << bestMatch_val << endl;
						//cout << numplanche << endl;

						if (scale_resolution == 1) {
							cout << "Canal " << indice_canal_RGB << " Planche " << numplanche << " Val = " << bestMatch_val << " Angle = " << bestMatch_angle << endl;
							//Mat matchInSrc = img_display(Rect(bestMatch_loc, Point(bestMatch_loc.x + bestMatch_templ.cols, bestMatch_loc.y + bestMatch_templ.rows)));
							//bitwise_and(matchInSrc, bestMatch_mask, matchInSrc);
							//imwrite("p7.png", matchInSrc);
							//imwrite("p8.png", bestMatch_templ);

							//bestMatch_templ.copyTo(img_display.rowRange(bestMatch_loc.y, bestMatch_loc.y + bestMatch_templ.rows).colRange(bestMatch_loc.x, bestMatch_loc.x + bestMatch_templ.cols), bestMatch_mask);



							if (bestMatch_val < minOfMin_val) {
								minOfMin_val = bestMatch_val;
								minOfMin_loc = bestMatch_loc;
								minOfMin_angle = bestMatch_angle;
								minOfMin_templ = bestMatch_templ;
								minOfMin_mask = bestMatch_mask;
								minOfMin_result = bestMatch_result;

							}

							//waitKey(0);
						}



						scale_resolution *= 2;
					}
					// Min du min 

					cout << "Min of min : Canal " << indice_canal_RGB << " Planche " << numplanche << " Val = " << minOfMin_val << " Angle = " << minOfMin_angle << endl;

					if (indice_canal_RGB == 0) {
						//cvtColor(minOfMin_templ,minOfMin_templ,COLOR_GRAY2BGR);
						//if (minOfMin_val < 2000)
						minOfMin_templ.copyTo(img_display.rowRange(minOfMin_loc.y, minOfMin_loc.y + minOfMin_templ.rows).colRange(minOfMin_loc.x, minOfMin_loc.x + minOfMin_templ.cols), minOfMin_mask);

						rectangle(img_display, minOfMin_loc, Point(minOfMin_loc.x + minOfMin_templ.cols, minOfMin_loc.y + minOfMin_templ.rows), Scalar::all(0), 2, 8, 0);
						rectangle(minOfMin_result, minOfMin_loc, Point(minOfMin_loc.x + minOfMin_templ.cols, minOfMin_loc.y + minOfMin_templ.rows), Scalar::all(0), 2, 8, 0);
						namedWindow(image_window, WINDOW_NORMAL);
						namedWindow(result_window, WINDOW_NORMAL);
						imshow(image_window, img_display);
						imshow(result_window, minOfMin_result);
					}
					//waitKey(0);
				}
			}


		}
		// Rotation billon matching -----------------------------------
		if (key == 'r') {

			// ---------------------------------- Initialisation des parametres ---------------------------------- //

			// Recuperation du billon 
			string billon_name = "A05a";
			cout << "Billon a traiter : ";
			cin >> billon_name;
			Mat billon;
			billon = imread("Billons/" + billon_name + ".jpeg");
			namedWindow("la source pour le screen", WINDOW_NORMAL);
			imshow("la source pour le screen", billon);

			// Recuperation des planches 
			int start_planche = 0, end_planche = 1;
			cout << "Numero des planches a traiter :" << endl; 
			cout << "De : "; cin >> start_planche;
			cout << " A : "; cin >> end_planche;

			// Recuperation de l'echelle
			double scale_planche_billon = 2;
			cout << "Scaling planche/billon :";
			cin >> scale_planche_billon;

			// Recuperation du pas pour les angles
			int pas_angle = 15;
			cout << "Pas de la recherche d'angle (15 par default) :";
			cin >> pas_angle;

			time_t start, end;
			time(&start);

			// Initialisation resolution scale 
			double scale_resolution = 0.25;
			// Initialisation seuil superposition 
			double superposition_threshold = 0.50;

			Mat templ; Mat result; Mat mask_final;
			//billon = imread("Paquet 3/planks-color/plank-color-1/Arbre 1n/A05a2.jpeg");

			//billon = imread("Paquet 3/planks-color/plank-color-1/Arbre 1/A05a2.jpeg");
			//billon = imread("Paquet 3/planks-color/plank-color-1/Arbre 2/A03a2.jpeg");
			//billon = imread("Paquet 3/planks-color/plank-color-1/Arbre 3/A04a2.jpeg");
			//billon = imread("Paquet 3/planks-color/plank-color-1/Arbre 4/A03c2.jpeg");
			//billon = imread("Paquet 3/planks-color/plank-color-1/Arbre 5/A01c2.jpeg");
			//billon = imread("Paquet 3/planks-color/plank-color-1/Arbre 6/A05c2.jpeg");
			//billon = imread("Paquet 3/planks-color/plank-color-1/Arbre 7/A01a2.jpeg");
			//billon = imread("Paquet 3/planks-color/plank-color-1/Arbre 8/A04c2.jpeg");

			//billon = imread("Paquet 3/planks-color/plank-color-2/Arbre 1/B10c.jpeg");
			//billon = imread("Paquet 3/planks-color/plank-color-2/Arbre 2/B01a.jpeg");
			//billon = imread("Paquet 3/planks-color/plank-color-2/Arbre 3/B10a.jpeg"); -------- 
			//billon = imread("Paquet 3/planks-color/plank-color-2/Arbre 4/B09c.jpeg");
			//billon = imread("Paquet 3/planks-color/plank-color-2/Arbre 5/B08c.jpeg");
			//billon = imread("Paquet 3/planks-color/plank-color-2/Arbre 6/B08a.jpeg");
			//billon = imread("Paquet 3/planks-color/plank-color-2/Arbre 7/B09a.jpeg");
			//billon = imread("Paquet 3/planks-color/plank-color-2/Arbre 8/B01c.jpeg");

			//billon = imread("Paquet 3/planks-color/plank-color-3/Arbre 1/C04c.jpeg");
			//billon = imread("Paquet 3/planks-color/plank-color-3/Arbre 2/C13c.jpeg");
			//billon = imread("Paquet 3/planks-color/plank-color-3/Arbre 3/C08c.jpeg");
			//billon = imread("Paquet 3/planks-color/plank-color-3/Arbre 4/C02c.jpeg");
			//billon = imread("Paquet 3/planks-color/plank-color-3/Arbre 5/C08a.jpeg");
			//billon = imread("Paquet 3/planks-color/plank-color-3/Arbre 6/C04a.jpeg"); 
			//billon = imread("Paquet 3/planks-color/plank-color-3/Arbre 7/C02a.jpeg"); 

			//billon = imread("Paquet 3/planks-color/plank-color-4/Arbre 1/D09c.jpeg");
			//billon = imread("Paquet 3/planks-color/plank-color-4/Arbre 2/D03c.jpeg");
			//billon = imread("Paquet 3/planks-color/plank-color-4/Arbre 3/D02a.jpeg");
			//billon = imread("Paquet 3/planks-color/plank-color-4/Arbre 4/D03a.jpeg");
			//billon = imread("Paquet 3/planks-color/plank-color-4/Arbre 5/C13a.jpeg");

			//billon = imread("Paquet 3/planks-color/plank-color-5/Arbre 1/D09a.jpeg");
			//billon = imread("Paquet 3/planks-color/plank-color-5/Arbre 2/D12c.jpeg");
			//billon = imread("Paquet 3/planks-color/plank-color-5/Arbre 3/D12a.jpeg");
			//billon = imread("Paquet 3/planks-color/plank-color-5/Arbre 4/D02c.jpeg");

			// Decoupage (supression des bords noirs)
			Mat cut_billon;
			cvtColor(billon, cut_billon, COLOR_BGR2GRAY);
			threshold(cut_billon, cut_billon, 1, 255, THRESH_BINARY);
			findNonZero(cut_billon, cut_billon);
			Rect first_bords = boundingRect(cut_billon);

			// On recupere tout les angles puis on les parcours 
			vector<tuple<double, int>> liste_angles;
			for (int i = 0; i < 360; i += pas_angle)
			{
				liste_angles.push_back(make_tuple(0, i));
			}

			// Reduction de dimension pour accelerer  
				// Diminution de la resolution de l'image source 
			resize(billon, billon, Size(0, 0), scale_resolution, scale_resolution);

			Mat img_display;
			billon.copyTo(img_display);

			namedWindow("la source pour le screen", WINDOW_NORMAL);
			imshow("la source pour le screen", billon);
			// ----------------------------- Creation de la liste de resultat ----------------------------- //
			struct bestMatch_planche
			{
				int numplanche;
				double scale;
				double rescale = 1;
				double score;
				int angle;
				Point loc;
				Mat image;
				vector<tuple<int, Point, double>> liste_angleLocScore;
			};
			vector<bestMatch_planche> bestMatch_planches;
			for (int numplanche = start_planche; numplanche < end_planche + 1; numplanche++) {
				bestMatch_planche p;
				p.numplanche = numplanche;
				p.score = DBL_MAX;
				p.angle = 0;
				p.loc = Point(0, 0);
				// Recuperation de l'image de la planche
				//p.image = imread("Paquet 3/planks-color/plank-color-1/Arbre 1/p" + to_string(numplanche) + ".png"); // 4 - 16
				//p.image = imread("Paquet 3/planks-color/plank-color-1/Arbre 2/p" + to_string(numplanche) + ".png"); // 17 - 26
				//p.image = imread("Paquet 3/planks-color/plank-color-1/Arbre 3/p" + to_string(numplanche) + ".png"); // 27 - 43 
				//p.image = imread("Paquet 3/planks-color/plank-color-1/Arbre 4/p" + to_string(numplanche) + ".png"); // 44 - 48
				//p.image = imread("Paquet 3/planks-color/plank-color-1/Arbre 5/p" + to_string(numplanche) + ".png"); // 49 - 58 
				//p.image = imread("Paquet 3/planks-color/plank-color-1/Arbre 6/p" + to_string(numplanche) + ".png"); // 59 - 63 
				//p.image = imread("Paquet 3/planks-color/plank-color-1/Arbre 7/p" + to_string(numplanche) + ".png"); // 64 - 75
				//p.image = imread("Paquet 3/planks-color/plank-color-1/Arbre 8/p" + to_string(numplanche) + ".png"); // 76 - 86

				//p.image = imread("Paquet 3/planks-color/plank-color-2/Arbre 1/p" + to_string(numplanche) + ".png"); // 6 - 11 
				//p.image = imread("Paquet 3/planks-color/plank-color-2/Arbre 2/p" + to_string(numplanche) + ".png"); // 12 - 24
				//p.image = imread("Paquet 3/planks-color/plank-color-2/Arbre 3/p" + to_string(numplanche) + ".png"); // 25 - 32
				//p.image = imread("Paquet 3/planks-color/plank-color-2/Arbre 4/p" + to_string(numplanche) + ".png"); // 33 - 39
				//p.image = imread("Paquet 3/planks-color/plank-color-2/Arbre 5/p" + to_string(numplanche) + ".png"); // 40 - 45
				//p.image = imread("Paquet 3/planks-color/plank-color-2/Arbre 6/p" + to_string(numplanche) + ".png"); // 46 - 56
				//p.image = imread("Paquet 3/planks-color/plank-color-2/Arbre 7/p" + to_string(numplanche) + ".png"); // 57 - 67
				//p.image = imread("Paquet 3/planks-color/plank-color-2/Arbre 8/p" + to_string(numplanche) + ".png"); // 68 - 75

				//p.image = imread("Paquet 3/planks-color/plank-color-3/Arbre 1/p" + to_string(numplanche) + ".png"); // 5 - 14 
				//p.image = imread("Paquet 3/planks-color/plank-color-3/Arbre 2/p" + to_string(numplanche) + ".png"); // 15 - 20
				//p.image = imread("Paquet 3/planks-color/plank-color-3/Arbre 3/p" + to_string(numplanche) + ".png"); // 21 - 30
				//p.image = imread("Paquet 3/planks-color/plank-color-3/Arbre 4/p" + to_string(numplanche) + ".png"); // 31 - 40
				//p.image = imread("Paquet 3/planks-color/plank-color-3/Arbre 5/p" + to_string(numplanche) + ".png"); // 41 - 57
				//p.image = imread("Paquet 3/planks-color/plank-color-3/Arbre 6/p" + to_string(numplanche) + ".png"); // 58 - 72
				//p.image = imread("Paquet 3/planks-color/plank-color-3/Arbre 7/p" + to_string(numplanche) + ".png"); // 73 - 91

				//p.image = imread("Paquet 3/planks-color/plank-color-4/Arbre 1/p" + to_string(numplanche) + ".png"); // 6 - 14 
				//p.image = imread("Paquet 3/planks-color/plank-color-4/Arbre 2/p" + to_string(numplanche) + ".png"); // 15 - 33
				//p.image = imread("Paquet 3/planks-color/plank-color-4/Arbre 3/p" + to_string(numplanche) + ".png"); // 34 - 48
				//p.image = imread("Paquet 3/planks-color/plank-color-4/Arbre 4/p" + to_string(numplanche) + ".png"); // 49 - 75
				//p.image = imread("Paquet 3/planks-color/plank-color-4/Arbre 5/p" + to_string(numplanche) + ".png"); // 76 - 82

				//p.image = imread("Paquet 3/planks-color/plank-color-5/Arbre 1/p" + to_string(numplanche) + ".png"); // 20 - 34 
				//p.image = imread("Paquet 3/planks-color/plank-color-5/Arbre 2/p" + to_string(numplanche) + ".png"); // 35 - 41
				//p.image = imread("Paquet 3/planks-color/plank-color-5/Arbre 3/p" + to_string(numplanche) + ".png"); // 42 - 51
				//p.image = imread("Paquet 3/planks-color/plank-color-5/Arbre 4/p" + to_string(numplanche) + ".png"); // 52 - 61

				stringstream ss;
				String leNumPlanche;
				ss << /*setw(3) << setfill('0') << */ numplanche;
				leNumPlanche = ss.str();

				//p.image = imread("Planche/p" + leNumPlanche + ".tif"); //  71 -  80
				p.image = imread("planches/p" + leNumPlanche + ".png"); //  71 -  80
				cout << "planches/p" + leNumPlanche + ".png" << endl;
				//p.image = imread("Paquet 3/planks-color/plank-color-1/Arbre 1n/pn" + leNumPlanche + ".tif"); //  71 -  80

				//p.image = imread("Paquet 3/planks-color/plank-color-1/Arbre 1/p" + leNumPlanche + ".tif"); //  71 -  80
				//p.image = imread("Paquet 3/planks-color/plank-color-1/Arbre 2/p" + leNumPlanche + ".tif"); //  61 -  70
				//p.image = imread("Paquet 3/planks-color/plank-color-1/Arbre 3/p" + leNumPlanche + ".tif"); //  44 -  60
				//p.image = imread("Paquet 3/planks-color/plank-color-1/Arbre 4/p" + leNumPlanche + ".tif"); //  39 -  43
				//p.image = imread("Paquet 3/planks-color/plank-color-1/Arbre 5/p" + leNumPlanche + ".tif"); //  29 -  38
				//p.image = imread("Paquet 3/planks-color/plank-color-1/Arbre 6/p" + leNumPlanche + ".tif"); //  24 -  28
				//p.image = imread("Paquet 3/planks-color/plank-color-1/Arbre 7/p" + leNumPlanche + ".tif"); //  12 -  23
				//p.image = imread("Paquet 3/planks-color/plank-color-1/Arbre 8/p" + leNumPlanche + ".tif"); //   1 -  11

				//p.image = imread("Paquet 3/planks-color/plank-color-2/Arbre 1/p" + leNumPlanche + ".tif"); // 143 - 147
				//p.image = imread("Paquet 3/planks-color/plank-color-2/Arbre 2/p" + leNumPlanche + ".tif"); // 130 - 142
				//p.image = imread("Paquet 3/planks-color/plank-color-2/Arbre 3/p" + leNumPlanche + ".tif"); // 122 - 129
				//p.image = imread("Paquet 3/planks-color/plank-color-2/Arbre 4/p" + leNumPlanche + ".tif"); // 115 - 121
				//p.image = imread("Paquet 3/planks-color/plank-color-2/Arbre 5/p" + leNumPlanche + ".tif"); // 109 - 114
				//p.image = imread("Paquet 3/planks-color/plank-color-2/Arbre 6/p" + leNumPlanche + ".tif"); //  99 - 108
				//p.image = imread("Paquet 3/planks-color/plank-color-2/Arbre 7/p" + leNumPlanche + ".tif"); //  89 -  98
				//p.image = imread("Paquet 3/planks-color/plank-color-2/Arbre 8/p" + leNumPlanche + ".tif"); //  81 -  88

				//p.image = imread("Paquet 3/planks-color/plank-color-3/Arbre 1/p" + leNumPlanche + ".tif"); // 220 - 228
				//p.image = imread("Paquet 3/planks-color/plank-color-3/Arbre 2/p" + leNumPlanche + ".tif"); // 215 - 219
				//p.image = imread("Paquet 3/planks-color/plank-color-3/Arbre 3/p" + leNumPlanche + ".tif"); // 205 - 214
				//p.image = imread("Paquet 3/planks-color/plank-color-3/Arbre 4/p" + leNumPlanche + ".tif"); // 195 - 204
				//p.image = imread("Paquet 3/planks-color/plank-color-3/Arbre 5/p" + leNumPlanche + ".tif"); // 178 - 194
				//p.image = imread("Paquet 3/planks-color/plank-color-3/Arbre 6/p" + leNumPlanche + ".tif"); // 164 - 177
				//p.image = imread("Paquet 3/planks-color/plank-color-3/Arbre 7/p" + leNumPlanche + ".tif"); // 148 - 163

				//p.image = imread("Paquet 3/planks-color/plank-color-4/Arbre 1/p" + leNumPlanche + ".tif"); // 297 - 305
				//p.image = imread("Paquet 3/planks-color/plank-color-4/Arbre 2/p" + leNumPlanche + ".tif"); // 278 - 296
				//p.image = imread("Paquet 3/planks-color/plank-color-4/Arbre 3/p" + leNumPlanche + ".tif"); // 263 - 277
				//p.image = imread("Paquet 3/planks-color/plank-color-4/Arbre 4/p" + leNumPlanche + ".tif"); // 236 - 262
				//p.image = imread("Paquet 3/planks-color/plank-color-4/Arbre 5/p" + leNumPlanche + ".tif"); // 229 - 235

				//p.image = imread("Paquet 3/planks-color/plank-color-5/Arbre 1/p" + leNumPlanche + ".tif"); // 333 - 346
				//p.image = imread("Paquet 3/planks-color/plank-color-5/Arbre 2/p" + leNumPlanche + ".tif"); // 326 - 332
				//p.image = imread("Paquet 3/planks-color/plank-color-5/Arbre 3/p" + leNumPlanche + ".tif"); // 316 - 325
				//p.image = imread("Paquet 3/planks-color/plank-color-5/Arbre 4/p" + leNumPlanche + ".tif"); // 306 - 315

				// On test si il y a une planche 
				//cout << (p.image.size()) << " | " << (p.image.size) << " | " << (p.image.size() == Size(0,0)) << endl;
				if (p.image.size() == Size(0, 0)) continue;
				// Mise a l'echelle planche billon 
				p.scale = scale_planche_billon;

				//p.scale = 1.04; // pc1 arbre 1n

				//p.scale = 2.16; // pc1 arbre 1
				//p.scale = 2.30; // pc1 arbre 2
				//p.scale = 2.06; // pc1 arbre 3
				//p.scale = 2.25; // pc1 arbre 4
				//p.scale = 2.38; // pc1 arbre 5
				//p.scale = 3.14; // pc1 arbre 6
				//p.scale = 2.45; // pc1 arbre 7
				//p.scale = 2.12; // pc1 arbre 8

				//p.scale = 3.57; // pc2 arbre 1
				//p.scale = 2.40; // pc2 arbre 2 (mesured scale)
				//p.scale = 2.00; // pc2 arbre 2 (best scale)
				//p.scale = 3.18; // pc2 arbre 3
				//p.scale = 3.01; // pc2 arbre 4
				//p.scale = 2.81; // pc2 arbre 5
				//p.scale = 2.75; // pc2 arbre 6
				//p.scale = 2.80; // pc2 arbre 7
				//p.scale = 2.75; // pc2 arbre 8

				//p.scale = 3.03; // pc3 arbre 1
				//p.scale = 4.67; // pc3 arbre 2 (mesured scale)
				//p.scale = 3.96; // pc3 arbre 2 (best scale)
				//p.scale = 3.04; // pc3 arbre 3
				//p.scale = 2.86; // pc3 arbre 4
				//p.scale = 2.36; // pc3 arbre 5
				//p.scale = 2.97; // pc3 arbre 6 (mesured scale)
				//p.scale = 2.67; // pc3 arbre 6 (best scale)
				//p.scale = 2.41; // pc3 arbre 7

				//p.scale = 2.65; // pc4 arbre 1 (mesured scale)
				//p.scale = 2.25; // pc4 arbre 1 (best scale)
				//p.scale = 2.08; // pc4 arbre 2
				//p.scale = 2.33; // pc4 arbre 3
				//p.scale = 1.83; // pc4 arbre 4
				//p.scale = 3.71; // pc4 arbre 5

				//p.scale = 2.36; // pc5 arbre 1
				//p.scale = 3.13; // pc5 arbre 2
				//p.scale = 2.88; // pc5 arbre 3
				//p.scale = 2.58; // pc5 arbre 4

				resize(p.image, p.image, Size(0, 0), p.scale, p.scale);

				// Diminution de la resolution de la planche 
				resize(p.image, p.image, Size(0, 0), scale_resolution, scale_resolution);
				bestMatch_planches.push_back(p);
			}
			// ----------------------------- Recuperation de l'angle ----------------------------- //
			for (int indice_angle = 0; indice_angle < liste_angles.size(); indice_angle++)
			{

				// Rotation du billon 
				int angle = get<1>(liste_angles[indice_angle]);
				Mat rot_billon;
				Point2f billon_center(billon.cols / 2.0F, billon.rows / 2.0F);
				Mat rot_mat = getRotationMatrix2D(billon_center, angle, 1.0);
				Rect2f bbox = cv::RotatedRect(cv::Point2f(), billon.size(), angle).boundingRect2f();
				rot_mat.at<double>(0, 2) += bbox.width / 2.0 - billon.cols / 2.0;
				rot_mat.at<double>(1, 2) += bbox.height / 2.0 - billon.rows / 2.0;
				warpAffine(billon, rot_billon, rot_mat, bbox.size());

				// Decoupage (supression des bords noirs)
				Mat cut_billon;
				cvtColor(rot_billon, cut_billon, COLOR_BGR2GRAY);
				threshold(cut_billon, cut_billon, 1, 255, THRESH_BINARY);
				findNonZero(cut_billon, cut_billon);
				Rect bords = boundingRect(cut_billon);
				rot_billon = rot_billon(bords);

				//namedWindow("rot billon", WINDOW_NORMAL);
				//imshow("rot billon", rot_billon);

				// Extraction ab de Lab 
				// Conversion BGR a Lab 
				cvtColor(rot_billon, rot_billon, COLOR_BGR2Lab);
				
				// Supression du cannal L de Lab 
				vector<Mat> billon_planes;
				split(rot_billon, billon_planes);
				billon_planes[0] = 0;
				merge(billon_planes, rot_billon);
				
				// Affichage du rendu de ab 
				//namedWindow("billon", WINDOW_NORMAL);
				//imshow("billon", rot_billon);

				// Initialisation de la liste des score par angle
				vector<double> liste_score_angle;
				double somm_score_angle = 0;
				for (int indice_planche = 0; indice_planche < bestMatch_planches.size(); indice_planche++)
				{
					// On recupere l'image de la planche 
					bestMatch_planches[indice_planche].image.copyTo(templ);

					// Extraction ab de Lab dans la planche 
					// Conversion BGR a Lab 
					cvtColor(templ, templ, COLOR_BGR2Lab);
					
					// Supression du cannal L de Lab 
					vector<Mat> templ_planes;
					split(templ, templ_planes);
					templ_planes[0] = 0;
					merge(templ_planes, templ);
					
					//namedWindow("plan", WINDOW_NORMAL);
					//imshow("plan", templ);

					// Matching 
					matchTemplate(rot_billon, templ, result, CV_TM_CCOEFF_NORMED);

					// Recuperation de la position ou il y a matching 
					double minVal; double maxVal; Point minLoc; Point maxLoc;
					Point matchLoc;
					minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

					matchLoc = maxLoc;

					cout << "Planche " << bestMatch_planches[indice_planche].numplanche << " Angle : " << angle << " Val : " << maxVal << endl;
					// Ajout du score a la liste d'angle 
					liste_score_angle.push_back(maxVal);
					somm_score_angle += maxVal;

					//imwrite("rot_billon.jpg", rot_billon);
					//imwrite("rot_templ.jpg", templ);

					// Si score max alors on le garde , c'est le meilleur match 
					if (maxVal > bestMatch_planches[indice_planche].score) {
						bestMatch_planches[indice_planche].score = maxVal;
						bestMatch_planches[indice_planche].loc = matchLoc;
						bestMatch_planches[indice_planche].angle = angle;
						//cout << "Min val = " << angle << endl;
					}

					bestMatch_planches[indice_planche].liste_angleLocScore.push_back(make_tuple(angle, matchLoc, maxVal));
				}
				// On trie la liste des score d'angle pour garder que les deux quartile du centre 
				sort(liste_score_angle.begin(), liste_score_angle.end());
				double somm_score_angle_quartile = 0;
				for (int i = liste_score_angle.size() * 1 / 4; i < liste_score_angle.size() * 3 / 4; i++)
				{
					somm_score_angle_quartile += liste_score_angle[i];
				}
				cout << "\tAngle : " << angle << " Score angle : " << somm_score_angle << endl;
				// On stock le core de l'angle 
				get<0>(liste_angles[indice_angle]) = somm_score_angle_quartile;
			}
			// Cumul des score des angles oppose pour choisir le meilleur angle 
			vector<tuple<double, int>> liste_angles_pair;
			for (int i = 0; i < 180; i += pas_angle)
			{
				liste_angles_pair.push_back(make_tuple(0, i));
			}
			sort(liste_angles.begin(), liste_angles.end());
			for (int i = 0; i < liste_angles.size(); i++)
			{
				cout << "Angle : " << get<1>(liste_angles[i]) << " \tScore : " << get<0>(liste_angles[i]) << endl;
				get<0>(liste_angles_pair[(get<1>(liste_angles[i]) % 180) / pas_angle]) += get<0>(liste_angles[i]);
			}
			sort(liste_angles_pair.begin(), liste_angles_pair.end(), greater<>());
			//liste_angles_pair.erase(liste_angles_pair.begin(), liste_angles_pair.begin() + 4);
			for (int i = 0; i < liste_angles_pair.size(); i++)
			{
				cout << "Pair Angle : " << get<1>(liste_angles_pair[i]) << " \tScore : " << get<0>(liste_angles_pair[i]) << endl;
			}


			// ----------------------------- Recuperation meilleur matching pour l'angle ----------------------------- //
			for (int i = 0; i < bestMatch_planches.size(); i++)
			{
				cout << "Planche " << bestMatch_planches[i].numplanche /*<< " : Score = " << bestMatch_planches[i].score << " , Angle = " << bestMatch_planches[i].angle*/ << endl;

				bestMatch_planches[i].score = 0;
				// On parcours la liste angleLocScore de la planche pour trouver le meilleur matching pour l'angle 
				for (int i_angle = 0; i_angle < bestMatch_planches[i].liste_angleLocScore.size(); i_angle++)
				{
					if (get<0>(bestMatch_planches[i].liste_angleLocScore[i_angle]) % 90 == get<1>(liste_angles_pair[0]) % 90) {
						if (get<2>(bestMatch_planches[i].liste_angleLocScore[i_angle]) > bestMatch_planches[i].score) {
							bestMatch_planches[i].angle = get<0>(bestMatch_planches[i].liste_angleLocScore[i_angle]);
							bestMatch_planches[i].loc = get<1>(bestMatch_planches[i].liste_angleLocScore[i_angle]);
							bestMatch_planches[i].score = get<2>(bestMatch_planches[i].liste_angleLocScore[i_angle]);
						}
					}
				}
			}
			// ----------------------------- Test de superposition ----------------------------- //
			double superposition_globale = 0;
			for (int i = 0; i < bestMatch_planches.size(); i++)
			{
				Mat mask_superposition, billon_rot_forMask;
				img_display.copyTo(billon_rot_forMask);
				img_display.copyTo(mask_superposition);
				cvtColor(mask_superposition, mask_superposition, COLOR_BGR2GRAY);
				threshold(mask_superposition, mask_superposition, 1, 255, THRESH_BINARY_INV);
				//mask_superposition.setTo(0);
				// On remplis le masque pour determiner s'il y a superposition 
				for (int k = 0; k < 4; k++)
				{
					int angle = get<1>(liste_angles_pair[0]) + k * 90;

					angle = angle; // -----------------------------------------------------------------------------------------
					Point2f billon_center = Point2f(billon_rot_forMask.cols / 2.0F, billon_rot_forMask.rows / 2.0F);
					Mat rot_mat = getRotationMatrix2D(billon_center, angle, 1.0);
					Rect bbox = RotatedRect(Point2f(), billon_rot_forMask.size(), angle).boundingRect2f();
					// Ajustement de la matrice de rotation a la taile de l'image
					rot_mat.at<double>(0, 2) += bbox.width / 2.0 - billon_rot_forMask.cols / 2.0;
					rot_mat.at<double>(1, 2) += bbox.height / 2.0 - billon_rot_forMask.rows / 2.0;
					warpAffine(billon_rot_forMask, billon_rot_forMask, rot_mat, bbox.size());
					warpAffine(mask_superposition, mask_superposition, rot_mat, bbox.size());

					Mat cut_billon;
					cvtColor(billon_rot_forMask, cut_billon, COLOR_BGR2GRAY);
					threshold(cut_billon, cut_billon, 1, 255, THRESH_BINARY);
					findNonZero(cut_billon, cut_billon);
					Rect bords = boundingRect(cut_billon);
					billon_rot_forMask = billon_rot_forMask(bords);
					mask_superposition = mask_superposition(bords);

					for (int l = 0; l < bestMatch_planches.size(); l++)
					{
						if (l != i && bestMatch_planches[l].angle == angle) {
							bestMatch_planches[l].image.copyTo(templ);
							templ.copyTo(billon_rot_forMask.rowRange(bestMatch_planches[l].loc.y, bestMatch_planches[l].loc.y + templ.rows).colRange(bestMatch_planches[l].loc.x, bestMatch_planches[l].loc.x + templ.cols));
							cvtColor(templ, templ, COLOR_BGR2GRAY);
							templ.setTo(255);
							templ.copyTo(mask_superposition.rowRange(bestMatch_planches[l].loc.y, bestMatch_planches[l].loc.y + templ.rows).colRange(bestMatch_planches[l].loc.x, bestMatch_planches[l].loc.x + templ.cols));
						}
					}

					angle = 360 - angle; // -----------------------------------------------------------------------------------------
					billon_center = Point2f(billon_rot_forMask.cols / 2.0F, billon_rot_forMask.rows / 2.0F);
					rot_mat = getRotationMatrix2D(billon_center, angle, 1.0);
					bbox = RotatedRect(Point2f(), billon_rot_forMask.size(), angle).boundingRect2f();
					// Ajustement de la matrice de rotation a la taile de l'image
					rot_mat.at<double>(0, 2) += bbox.width / 2.0 - billon_rot_forMask.cols / 2.0;
					rot_mat.at<double>(1, 2) += bbox.height / 2.0 - billon_rot_forMask.rows / 2.0;
					warpAffine(billon_rot_forMask, billon_rot_forMask, rot_mat, bbox.size());
					warpAffine(mask_superposition, mask_superposition, rot_mat, bbox.size());

					cut_billon;
					cvtColor(billon_rot_forMask, cut_billon, COLOR_BGR2GRAY);
					threshold(cut_billon, cut_billon, 1, 255, THRESH_BINARY);
					findNonZero(cut_billon, cut_billon);
					bords = boundingRect(cut_billon);
					billon_rot_forMask = billon_rot_forMask(bords);
					mask_superposition = mask_superposition(bords);

					//namedWindow("rot billon", WINDOW_NORMAL);
					//imshow("rot billon", mask_superposition);
				}

				// On compte le nombre de pixels dans le masque a l'endroit ou on devrait poser la planche 
				int angle = bestMatch_planches[i].angle;

				// Rotation 
				angle = angle; // -----------------------------------------------------------------------------------------
				Point2f billon_center = Point2f(billon_rot_forMask.cols / 2.0F, billon_rot_forMask.rows / 2.0F);
				Mat rot_mat = getRotationMatrix2D(billon_center, angle, 1.0);
				Rect bbox = RotatedRect(Point2f(), billon_rot_forMask.size(), angle).boundingRect2f();
				// Ajustement de la matrice de rotation a la taile de l'image
				rot_mat.at<double>(0, 2) += bbox.width / 2.0 - billon_rot_forMask.cols / 2.0;
				rot_mat.at<double>(1, 2) += bbox.height / 2.0 - billon_rot_forMask.rows / 2.0;
				warpAffine(billon_rot_forMask, billon_rot_forMask, rot_mat, bbox.size());
				warpAffine(mask_superposition, mask_superposition, rot_mat, bbox.size());

				// decoupage
				Mat cut_billon;
				cvtColor(billon_rot_forMask, cut_billon, COLOR_BGR2GRAY);
				threshold(cut_billon, cut_billon, 1, 255, THRESH_BINARY);
				findNonZero(cut_billon, cut_billon);
				Rect bords = boundingRect(cut_billon);
				billon_rot_forMask = billon_rot_forMask(bords);
				mask_superposition = mask_superposition(bords);

				Mat region_planche = mask_superposition(Rect(Point(bestMatch_planches[i].loc.x, bestMatch_planches[i].loc.y), Point((bestMatch_planches[i].loc.x + templ.cols > mask_superposition.cols ? mask_superposition.cols : bestMatch_planches[i].loc.x + templ.cols), (bestMatch_planches[i].loc.y + templ.rows > mask_superposition.rows ? mask_superposition.rows : bestMatch_planches[i].loc.y + templ.rows))));
				cout << "Planche " << bestMatch_planches[i].numplanche << " : " << countNonZero(region_planche) << " / " << templ.rows * templ.cols << " = " << (double)countNonZero(region_planche) / (double)(templ.rows * templ.cols) << endl;
				superposition_globale += (double)countNonZero(region_planche) / (double)(templ.rows * templ.cols);
				// A partir d'un certail seuil on force la planche a changer de position 
				if ((double)countNonZero(region_planche) / (double)(templ.rows * templ.cols) > superposition_threshold) {
					bestMatch_planches[i].score = 0;
					for (int i_angle = 0; i_angle < bestMatch_planches[i].liste_angleLocScore.size(); i_angle++)
					{
						if (get<0>(bestMatch_planches[i].liste_angleLocScore[i_angle]) % 90 == get<1>(liste_angles_pair[0]) % 90 && get<0>(bestMatch_planches[i].liste_angleLocScore[i_angle]) != angle) {
							if (get<2>(bestMatch_planches[i].liste_angleLocScore[i_angle]) > bestMatch_planches[i].score) {
								bestMatch_planches[i].angle = get<0>(bestMatch_planches[i].liste_angleLocScore[i_angle]);
								bestMatch_planches[i].loc = get<1>(bestMatch_planches[i].liste_angleLocScore[i_angle]);
								bestMatch_planches[i].score = get<2>(bestMatch_planches[i].liste_angleLocScore[i_angle]);
							}
						}
					}
				}

				//namedWindow("rot billon", WINDOW_NORMAL);
				//imshow("rot billon", region_planche);
				//waitKey(0);

				// Rotation retour a l'origine 
				angle = 360 - angle; // -----------------------------------------------------------------------------------------
				billon_center = Point2f(billon_rot_forMask.cols / 2.0F, billon_rot_forMask.rows / 2.0F);
				rot_mat = getRotationMatrix2D(billon_center, angle, 1.0);
				bbox = RotatedRect(Point2f(), billon_rot_forMask.size(), angle).boundingRect2f();
				// Ajustement de la matrice de rotation a la taile de l'image
				rot_mat.at<double>(0, 2) += bbox.width / 2.0 - billon_rot_forMask.cols / 2.0;
				rot_mat.at<double>(1, 2) += bbox.height / 2.0 - billon_rot_forMask.rows / 2.0;
				warpAffine(billon_rot_forMask, billon_rot_forMask, rot_mat, bbox.size());
				warpAffine(mask_superposition, mask_superposition, rot_mat, bbox.size());

				// Decoupage
				cut_billon;
				cvtColor(billon_rot_forMask, cut_billon, COLOR_BGR2GRAY);
				threshold(cut_billon, cut_billon, 1, 255, THRESH_BINARY);
				findNonZero(cut_billon, cut_billon);
				bords = boundingRect(cut_billon);
				billon_rot_forMask = billon_rot_forMask(bords);
				mask_superposition = mask_superposition(bords);

				//namedWindow("rot billon", WINDOW_NORMAL);
				//imshow("rot billon", mask_superposition);
			}
			superposition_globale = superposition_globale / bestMatch_planches.size();
			cout << "Superposition globale = " << superposition_globale << endl;

			// ----------------------------- Calcul du score moyen des planches ----------------------------- // 
			double final_score = 0;
			for (int i = 0; i < 4; i++)
			{
				int angle = (get<1>(liste_angles_pair[0]) + i * 90) % 360;
				for (int j = 0; j < bestMatch_planches.size(); j++)
				{
					if (bestMatch_planches[j].angle == angle) {
						final_score += bestMatch_planches[j].score;
					}
				}
			}

			final_score = final_score / bestMatch_planches.size();
			cout << "Score final : " << final_score << endl;
			//waitKey(0);
			// --------------------------- On tente une dminution de scaling --------------------------- //
			if (superposition_globale > 0.15 && final_score < 0.80) {
				double rescale_factor = 0.90;
				if (superposition_globale > 0.3 && final_score < 0.70) rescale_factor = 0.85;
				double score_prec = final_score;
				for (double sf = rescale_factor; sf < 1.15; sf+=0.01)
				{
					rescale_factor = sf;
					billon.copyTo(img_display);
					cout << "Test avec un scaling different : scale x " << rescale_factor << endl;
					for (int i = 0; i < 4; i++)
					{

						// Rotation du billon 
						int angle = (get<1>(liste_angles_pair[0]) + i * 90) % 360;
						Mat rot_billon;
						Point2f billon_center(billon.cols / 2.0F, billon.rows / 2.0F);
						Mat rot_mat = getRotationMatrix2D(billon_center, angle, 1.0);
						Rect2f bbox = cv::RotatedRect(cv::Point2f(), billon.size(), angle).boundingRect2f();
						rot_mat.at<double>(0, 2) += bbox.width / 2.0 - billon.cols / 2.0;
						rot_mat.at<double>(1, 2) += bbox.height / 2.0 - billon.rows / 2.0;
						warpAffine(billon, rot_billon, rot_mat, bbox.size());

						// Decoupage (supression des bords noirs)
						Mat cut_billon;
						cvtColor(rot_billon, cut_billon, COLOR_BGR2GRAY);
						threshold(cut_billon, cut_billon, 1, 255, THRESH_BINARY);
						findNonZero(cut_billon, cut_billon);
						Rect bords = boundingRect(cut_billon);
						rot_billon = rot_billon(bords);

						//namedWindow("rot billon", WINDOW_NORMAL);
						//imshow("rot billon", rot_billon);

						// Extraction ab de Lab 
						// Conversion BGR a Lab 
						cvtColor(rot_billon, rot_billon, COLOR_BGR2Lab);

						// Supression du cannal L de Lab 
						vector<Mat> billon_planes;
						split(rot_billon, billon_planes);
						billon_planes[0] = 0;
						merge(billon_planes, rot_billon);

						// Affichage du rendu de ab 
						//namedWindow("billon", WINDOW_NORMAL);
						//imshow("billon", rot_billon);

						// Initialisation de la liste des score par angle
						for (int indice_planche = 0; indice_planche < bestMatch_planches.size(); indice_planche++)
						{
							// On recupere l'image de la planche 
							bestMatch_planches[indice_planche].image.copyTo(templ);
							double rescaling = (1 / bestMatch_planches[indice_planche].scale) * (rescale_factor * bestMatch_planches[indice_planche].scale);
							resize(templ, templ, Size(0, 0), rescaling, rescaling);

							// Extraction ab de Lab dans la planche 
							// Conversion BGR a Lab 
							cvtColor(templ, templ, COLOR_BGR2Lab);

							// Supression du cannal L de Lab 
							vector<Mat> templ_planes;
							split(templ, templ_planes);
							templ_planes[0] = 0;
							merge(templ_planes, templ);

							//namedWindow("plan", WINDOW_NORMAL);
							//imshow("plan", templ);

							// Matching 
							matchTemplate(rot_billon, templ, result, CV_TM_CCOEFF_NORMED);

							// Recuperation de la position ou il y a matching 
							double minVal; double maxVal; Point minLoc; Point maxLoc;
							Point matchLoc;
							minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

							matchLoc = maxLoc;

							//cout << "Planche " << bestMatch_planches[indice_planche].numplanche << " Angle : " << angle << " Val : " << maxVal << endl;

							// Si score max alors on le garde , c'est le meilleur match 
							if (maxVal > bestMatch_planches[indice_planche].score) {
								bestMatch_planches[indice_planche].score = maxVal;
								bestMatch_planches[indice_planche].loc = matchLoc;
								bestMatch_planches[indice_planche].angle = angle;
								bestMatch_planches[indice_planche].rescale = rescale_factor;
								//cout << "Min val = " << angle << endl;
							}
						}
					}

					// Calcul du score global 
					final_score = 0;
					for (int i = 0; i < 4; i++)
					{
						int angle = (get<1>(liste_angles_pair[0]) + i * 90) % 360;

						for (int j = 0; j < bestMatch_planches.size(); j++)
						{
							if (bestMatch_planches[j].angle == angle) {
								final_score += bestMatch_planches[j].score;
							}
						}
					}

					final_score = final_score / bestMatch_planches.size();
					cout << "Score final : " << final_score << endl;
					if (score_prec == final_score) {
						break;
					}
					else {
						score_prec = final_score;
					}
					//waitKey(0);
				}
				
			}

			// ----------------------------- On affiche toutes les planches dans le billon ----------------------------- // 
			img_display.copyTo(mask_final);
			mask_final.setTo(0);
			final_score = 0;
			for (int i = 0; i < 4; i++)
			{
				int angle = (get<1>(liste_angles_pair[0]) + i * 90) % 360;

				// Rotation 
				angle = angle; // -----------------------------------------------------------------------------------------
				Point2f billon_center = Point2f(img_display.cols / 2.0F, img_display.rows / 2.0F);
				Mat rot_mat = getRotationMatrix2D(billon_center, angle, 1.0);
				Rect bbox = RotatedRect(Point2f(), img_display.size(), angle).boundingRect2f();
				// Ajustement de la matrice de rotation a la taile de l'image
				rot_mat.at<double>(0, 2) += bbox.width / 2.0 - img_display.cols / 2.0;
				rot_mat.at<double>(1, 2) += bbox.height / 2.0 - img_display.rows / 2.0;
				warpAffine(img_display, img_display, rot_mat, bbox.size());
				warpAffine(mask_final, mask_final, rot_mat, bbox.size());

				// Decoupage
				Mat cut_billon;
				cvtColor(img_display, cut_billon, COLOR_BGR2GRAY);
				threshold(cut_billon, cut_billon, 1, 255, THRESH_BINARY);
				findNonZero(cut_billon, cut_billon);
				Rect bords = boundingRect(cut_billon);
				img_display = img_display(bords);
				mask_final = mask_final(bords);

				for (int j = 0; j < bestMatch_planches.size(); j++)
				{
					if (bestMatch_planches[j].angle == angle) {
						bestMatch_planches[j].image.copyTo(templ);
						double rescaling = (1 / bestMatch_planches[j].scale) * (bestMatch_planches[j].rescale * bestMatch_planches[j].scale);
						resize(templ, templ, Size(0, 0), rescaling, rescaling);
						templ.copyTo(img_display.rowRange(bestMatch_planches[j].loc.y, bestMatch_planches[j].loc.y + templ.rows).colRange(bestMatch_planches[j].loc.x, bestMatch_planches[j].loc.x + templ.cols));
						final_score += bestMatch_planches[j].score;

						cout << "Planche " << bestMatch_planches[j].numplanche << " : " << bestMatch_planches[j].score << endl;

						templ.setTo(255);
						templ.copyTo(mask_final.rowRange(bestMatch_planches[j].loc.y, bestMatch_planches[j].loc.y + templ.rows).colRange(bestMatch_planches[j].loc.x, bestMatch_planches[j].loc.x + templ.cols));
					}
				}

				// Rotation retour a l'origine 
				angle = 360 - angle; // -----------------------------------------------------------------------------------------
				billon_center = Point2f(img_display.cols / 2.0F, img_display.rows / 2.0F);
				rot_mat = getRotationMatrix2D(billon_center, angle, 1.0);
				bbox = RotatedRect(Point2f(), img_display.size(), angle).boundingRect2f();
				// Ajustement de la matrice de rotation a la taile de l'image
				rot_mat.at<double>(0, 2) += bbox.width / 2.0 - img_display.cols / 2.0;
				rot_mat.at<double>(1, 2) += bbox.height / 2.0 - img_display.rows / 2.0;
				warpAffine(img_display, img_display, rot_mat, bbox.size());
				warpAffine(mask_final, mask_final, rot_mat, bbox.size());

				// Decoupe
				cut_billon;
				cvtColor(img_display, cut_billon, COLOR_BGR2GRAY);
				threshold(cut_billon, cut_billon, 1, 255, THRESH_BINARY);
				findNonZero(cut_billon, cut_billon);
				bords = boundingRect(cut_billon);
				img_display = img_display(bords);
				mask_final = mask_final(bords);

				namedWindow("rot billon", WINDOW_NORMAL);
				imshow("rot billon", img_display);
			}
			imwrite("Reconstruction" + billon_name + ".png", img_display);
			//rot_templ.copyTo(img_display.rowRange(matchLoc.y, matchLoc.y + rot_templ.rows).colRange(matchLoc.x, matchLoc.x + rot_templ.cols), mask);

			final_score = final_score / bestMatch_planches.size();
			cout << "Score final : " << final_score << endl;


			namedWindow("juste pour voir le resultat final", WINDOW_NORMAL);
			imshow("juste pour voir le resultat final", img_display);
			// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
			/*
			Mat masked_billon,mask_for_maskedbillon;
			bitwise_not(mask_final, mask_for_maskedbillon);
			bitwise_and(img_display, mask_for_maskedbillon, masked_billon);


			namedWindow("for Masked billon rot", WINDOW_NORMAL);
			imshow("for Masked billon rot", billon);
			namedWindow("for Masked billon", WINDOW_NORMAL);
			imshow("for Masked billon", mask_for_maskedbillon);
			waitKey(0);
			for (int i = 0; i < 4; i++)
			{
				Mat mask_for_maskedbillon_crop;
				mask_for_maskedbillon.copyTo(mask_for_maskedbillon_crop);

				// Decoupage (supression des bords noirs)
				cut_billon;
				cvtColor(billon, cut_billon, COLOR_BGR2GRAY);
				threshold(cut_billon, cut_billon, 1, 255, THRESH_BINARY);
				findNonZero(cut_billon, cut_billon);
				Rect bords = boundingRect(cut_billon);
				cut_billon = billon(bords);

				// Rotation du billon 
				int angle = (get<1>(liste_angles_pair[0]) + i * 90) % 360;
				Mat rot_billon;
				Point2f billon_center(cut_billon.cols / 2.0F, cut_billon.rows / 2.0F);
				Mat rot_mat = getRotationMatrix2D(billon_center, angle, 1.0);
				Rect2f bbox = cv::RotatedRect(cv::Point2f(), cut_billon.size(), angle).boundingRect2f();
				rot_mat.at<double>(0, 2) += bbox.width / 2.0 - cut_billon.cols / 2.0;
				rot_mat.at<double>(1, 2) += bbox.height / 2.0 - cut_billon.rows / 2.0;
				warpAffine(cut_billon, rot_billon, rot_mat, bbox.size());
				warpAffine(mask_for_maskedbillon_crop, mask_for_maskedbillon_crop, rot_mat, bbox.size());

				namedWindow("for Masked billon rot", WINDOW_NORMAL);
				imshow("for Masked billon rot", rot_billon);
				namedWindow("for Masked billon", WINDOW_NORMAL);
				imshow("for Masked billon", mask_for_maskedbillon_crop);
				waitKey(0);

				// Decoupage (supression des bords noirs)
				cut_billon;
				cvtColor(rot_billon, cut_billon, COLOR_BGR2GRAY);
				threshold(cut_billon, cut_billon, 1, 255, THRESH_BINARY);
				findNonZero(cut_billon, cut_billon);
				bords = boundingRect(cut_billon);
				rot_billon = rot_billon(bords);
				mask_for_maskedbillon_crop = mask_for_maskedbillon_crop(bords);

				bestMatch_planches[1].image.copyTo(templ);
				double rescaling = (1 / bestMatch_planches[1].scale) * (bestMatch_planches[1].rescale * bestMatch_planches[1].scale);
				resize(templ, templ, Size(0, 0), rescaling, rescaling);

				matchTemplate(rot_billon, templ, result, CV_TM_CCOEFF_NORMED);

				double minVal; double maxVal; Point minLoc; Point maxLoc;
				Point matchLoc;
				mask_for_maskedbillon_crop = mask_for_maskedbillon_crop(Rect(0,0,result.cols,result.rows));
				cvtColor(mask_for_maskedbillon_crop, mask_for_maskedbillon_crop, COLOR_BGR2GRAY);
				normalize(mask_for_maskedbillon_crop, mask_for_maskedbillon_crop, 0, 1, NORM_MINMAX, -1, Mat());
				vector<Mat> mask_planes;
				split(mask_for_maskedbillon_crop, mask_planes);
				//result.mul(mask_planes[0]);

				minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, mask_planes[0]);

				templ.copyTo(rot_billon.rowRange(maxLoc.y, maxLoc.y + templ.rows).colRange(maxLoc.x, maxLoc.x + templ.cols));

				cout << maxVal << endl;

				normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
				//threshold(result, result, 0.8, 1, THRESH_BINARY);
				imshow("result", result);

				namedWindow("Masked billon c", WINDOW_NORMAL);
				threshold(mask_for_maskedbillon_crop, mask_for_maskedbillon_crop, 0.5, 255, THRESH_BINARY);
				imshow("Masked billon c", mask_for_maskedbillon_crop);

				namedWindow("Masked billon", WINDOW_NORMAL);
				imshow("Masked billon", rot_billon);
				waitKey(0);
			}
			*/
			// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

			time(&end);
			cout << "Temps : " << double(end - start) << "s" << endl;

			// Calcul score par rapport a la veritee terrain 
			resize(mask_final, mask_final, Size(0, 0), 1 / scale_resolution, 1 / scale_resolution);
			resize(billon, billon, Size(0, 0), 1 / scale_resolution, 1 / scale_resolution);
			Mat temp_final_mask;
			billon.copyTo(temp_final_mask);
			temp_final_mask.setTo(0);
			int tlx = first_bords.tl().x;
			int tly = first_bords.tl().y;
			int width = first_bords.width;
			int height = first_bords.height;

			mask_final = mask_final(Rect(0,0,min(width,mask_final.cols),min(height,mask_final.rows)));
			mask_final
				.copyTo(
					temp_final_mask
					.rowRange(tly, ((tly + mask_final.rows) > temp_final_mask.rows) ? temp_final_mask.rows : (tly + mask_final.rows))
					.colRange(tlx, ((tlx + mask_final.cols) > temp_final_mask.cols) ? temp_final_mask.cols : (tlx + mask_final.cols)));

			namedWindow("rot billon", WINDOW_NORMAL);
			imshow("rot billon", temp_final_mask);
			temp_final_mask.copyTo(mask_final);

			Mat mask_verite;
			billon.copyTo(mask_verite);
			mask_verite.setTo(0);
			ifstream file("Paquet 3/planks-color/verite_terrain.txt");

			string line;
			string numBillon;
			double numSciage, largeurImage_pix, hauteurImage_pix, largeurImage_mm, hauteurImage_mm, posXMoelle_mm, posYMoelle_mm, larPlanche_mm, epPlanche_mm, angleRotation_deg, Ax_mm, Ay_mm, Bx_mm, By_mm, Cx_mm, Cy_mm, Dx_mm, Dy_mm;
			//while (file >> numBillon >> numSciage >> largeurImage_pix >> hauteurImage_pix >> largeurImage_mm >> hauteurImage_mm >> posXMoelle_mm >> posYMoelle_mm >> larPlanche_mm >> epPlanche_mm >> angleRotation_deg >> Ax_mm >> Ay_mm >> Bx_mm >> By_mm >> Cx_mm >> Cy_mm >> Dx_mm >> Dy_mm) {
			double avg_angle_verite = 0;
			while (getline(file, line)) {
				istringstream iss(line);
				iss >> numBillon >> numSciage >> largeurImage_pix >> hauteurImage_pix >> largeurImage_mm >> hauteurImage_mm >> posXMoelle_mm >> posYMoelle_mm >> larPlanche_mm >> epPlanche_mm >> angleRotation_deg >> Ax_mm >> Ay_mm >> Bx_mm >> By_mm >> Cx_mm >> Cy_mm >> Dx_mm >> Dy_mm;
				numBillon.erase(remove(numBillon.begin(), numBillon.end(), '\"'), numBillon.end());
				//cout << numBillon << endl;
				if (numBillon == billon_name) {
					Point rook_points[1][4];
					rook_points[0][0] = Point(Ax_mm * largeurImage_pix / largeurImage_mm, (hauteurImage_mm - Ay_mm) * hauteurImage_pix / hauteurImage_mm);
					rook_points[0][1] = Point(Bx_mm * largeurImage_pix / largeurImage_mm, (hauteurImage_mm - By_mm) * hauteurImage_pix / hauteurImage_mm);
					rook_points[0][2] = Point(Cx_mm * largeurImage_pix / largeurImage_mm, (hauteurImage_mm - Cy_mm) * hauteurImage_pix / hauteurImage_mm);
					rook_points[0][3] = Point(Dx_mm * largeurImage_pix / largeurImage_mm, (hauteurImage_mm - Dy_mm) * hauteurImage_pix / hauteurImage_mm);
					const Point* ppt[1] = { rook_points[0] };
					int npt[] = { 4 };

					fillPoly(mask_verite, ppt, npt, 1, Scalar(255, 255, 255));
					avg_angle_verite += ((int)angleRotation_deg) % 180;
				}
			}
			avg_angle_verite = 180 - avg_angle_verite / bestMatch_planches.size();
			double avg_angle_matching = 0;
			for (int i = 0; i < bestMatch_planches.size(); i++)
			{
				avg_angle_matching += bestMatch_planches[i].angle % 180;
			}
			avg_angle_matching /= bestMatch_planches.size();
			//namedWindow("verite billon", WINDOW_NORMAL);
			//imshow("verite billon", billon);
			namedWindow("mask billon", WINDOW_NORMAL);
			imshow("mask billon", mask_verite);

			Mat vrai_positif;
			bitwise_and(mask_final, mask_verite, vrai_positif);
			Mat faux_positif;
			bitwise_not(vrai_positif, faux_positif);
			bitwise_and(mask_final, faux_positif, faux_positif);
			Mat faux_negatif;
			bitwise_not(mask_final, faux_negatif);
			bitwise_and(mask_verite, faux_negatif, faux_negatif);

			cvtColor(vrai_positif, vrai_positif, COLOR_BGR2GRAY);
			cvtColor(faux_positif, faux_positif, COLOR_BGR2GRAY);
			cvtColor(faux_negatif, faux_negatif, COLOR_BGR2GRAY);
			double vp = countNonZero(vrai_positif);
			double fp = countNonZero(faux_positif);
			double fn = countNonZero(faux_negatif);
			double precision = vp / (vp + fp);
			double rappel = vp / (vp + fn);
			double f_mesure = 2 * ((precision * rappel) / (precision + rappel));
			cout << "Precision : " << precision << " Rappel : " << rappel << " F-mesure : " << f_mesure << " Difference d'angle : " << abs(avg_angle_verite - avg_angle_matching) << " les angles moyens : " << avg_angle_verite << "|" << avg_angle_matching << endl;


		}
		if (key == ':') {
			Mat billon = imread("Paquet 3/planks-color/plank-color-1/Arbre 1/A05a2.jpeg");
			//Mat billon = imread("Paquet 3/planks-color/plank-color-1/Arbre 5/A01c2.jpeg");
			Mat mask_verite;
			billon.copyTo(mask_verite);
			mask_verite.setTo(0);
			ifstream file("Paquet 3/planks-color/verite_terrain.txt");
			
			string line;
			string numBillon;
			double numSciage, largeurImage_pix, hauteurImage_pix, largeurImage_mm, hauteurImage_mm, posXMoelle_mm, posYMoelle_mm, larPlanche_mm, epPlanche_mm, angleRotation_deg, Ax_mm, Ay_mm, Bx_mm, By_mm, Cx_mm, Cy_mm, Dx_mm, Dy_mm;
			//while (file >> numBillon >> numSciage >> largeurImage_pix >> hauteurImage_pix >> largeurImage_mm >> hauteurImage_mm >> posXMoelle_mm >> posYMoelle_mm >> larPlanche_mm >> epPlanche_mm >> angleRotation_deg >> Ax_mm >> Ay_mm >> Bx_mm >> By_mm >> Cx_mm >> Cy_mm >> Dx_mm >> Dy_mm) {
			while (getline(file,line)) {
				istringstream iss(line);
				iss >> numBillon >> numSciage >> largeurImage_pix >> hauteurImage_pix >> largeurImage_mm >> hauteurImage_mm >> posXMoelle_mm >> posYMoelle_mm >> larPlanche_mm >> epPlanche_mm >> angleRotation_deg >> Ax_mm >> Ay_mm >> Bx_mm >> By_mm >> Cx_mm >> Cy_mm >> Dx_mm >> Dy_mm;
				numBillon.erase(remove(numBillon.begin(), numBillon.end(), '\"'), numBillon.end());
				cout << numBillon << endl;
				if (numBillon == "A05a") {
					Point rook_points[1][4];
					rook_points[0][0] = Point(Ax_mm * largeurImage_pix / largeurImage_mm, (hauteurImage_mm - Ay_mm) * hauteurImage_pix / hauteurImage_mm);
					rook_points[0][1] = Point(Bx_mm * largeurImage_pix / largeurImage_mm, (hauteurImage_mm - By_mm) * hauteurImage_pix / hauteurImage_mm);
					rook_points[0][2] = Point(Cx_mm * largeurImage_pix / largeurImage_mm, (hauteurImage_mm - Cy_mm) * hauteurImage_pix / hauteurImage_mm);
					rook_points[0][3] = Point(Dx_mm * largeurImage_pix / largeurImage_mm, (hauteurImage_mm - Dy_mm) * hauteurImage_pix / hauteurImage_mm);
					const Point* ppt[1] = { rook_points[0] };
					int npt[] = { 4 };

					fillPoly(billon, ppt, npt, 1, Scalar(255, 255, 255));
					fillPoly(mask_verite, ppt, npt, 1, Scalar(255, 255, 255));
				}
			}

			namedWindow("verite billon", WINDOW_NORMAL);
			imshow("verite billon", billon);
			namedWindow("mask billon", WINDOW_NORMAL);
			imshow("mask billon", mask_verite);
			//waitKey(0);
		}
		if (key == '!') {

			Mat billon;
			billon = imread("Billons/A01a.jpeg");
			Mat result;

			// Rotation du billon 
			int angle = (105 + 2 * 90) % 360;
			Mat rot_billon;
			Point2f billon_center(billon.cols / 2.0F, billon.rows / 2.0F);
			Mat rot_mat = getRotationMatrix2D(billon_center, angle, 1.0);
			Rect2f bbox = cv::RotatedRect(cv::Point2f(), billon.size(), angle).boundingRect2f();
			rot_mat.at<double>(0, 2) += bbox.width / 2.0 - billon.cols / 2.0;
			rot_mat.at<double>(1, 2) += bbox.height / 2.0 - billon.rows / 2.0;
			warpAffine(billon, rot_billon, rot_mat, bbox.size());

			// Decoupage (supression des bords noirs)
			Mat cut_billon;
			cvtColor(rot_billon, cut_billon, COLOR_BGR2GRAY);
			threshold(cut_billon, cut_billon, 1, 255, THRESH_BINARY);
			findNonZero(cut_billon, cut_billon);
			Rect bords = boundingRect(cut_billon);
			rot_billon = rot_billon(bords);


			Mat templ = imread("Planche/p013.tif"); //  71 -  80
			resize(templ, templ, Size(0, 0), 2.45, 2.45);

			matchTemplate(rot_billon, templ, result, TM_CCOEFF_NORMED);

			double minVal; double maxVal; Point minLoc; Point maxLoc;
			Point matchLoc;
			minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

			templ.copyTo(rot_billon.rowRange(maxLoc.y, maxLoc.y + templ.rows).colRange(maxLoc.x, maxLoc.x + templ.cols));

			cout << maxVal << endl;

			normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
			threshold(result, result, 0.8, 1, THRESH_BINARY);

			//circle(result, Point(1488, 1818), 15, Scalar(255, 0, 0), -1);
			namedWindow("result", WINDOW_NORMAL);
			imshow("result", result);

			circle(rot_billon, Point(1488, 1818), 15, Scalar(255, 0, 0), -1);

			cout << result.at<float>(1488, 1818) << endl;

			namedWindow("Masked billon", WINDOW_NORMAL);
			imshow("Masked billon", rot_billon);
			waitKey(0);
		}
		if (key == '*') {
			Mat planche1 = imread("Planche/p066.tif");

			// Extraction ab de Lab dans la planche 
			// Conversion BGR a Lab 
			cvtColor(planche1, planche1, COLOR_BGR2Lab);

			// Supression du cannal L de Lab 
			vector<Mat> templ_planes;
			split(planche1, templ_planes);
			templ_planes[0] = 0;
			merge(templ_planes, planche1);

			imwrite("p_lab.png", planche1);
			
		}
		// Quitter
		if (key == 'q') break;
	}

	return 0;
}
