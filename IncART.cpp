#include "IncART.h"

IncART::IncART()
{
	//根据自己的项目修改以下参数
	niter = 500;// 迭代次数
	lambda = 0.5;
	TV_niter = 20;
	TV_lambda = 0.4;
	IsHOTV = false;
	IsRwTV = true;
	delta = 0.3;
	
	//file_prefix = "TV/pcb_TV";
	//file_prefix = "TTV/pcb_TTV";
	file_prefix = "RATV/pcb_RATV1";
	//file_prefix = "TAwTV/pcb_TAwTV";
	
	dd = 0.127;
	detector_width = 512;//探测器宽
	detector_height = 512;//探测器高
	number_of_projections = 360;//旋转总数
	volume_width = 300;//图像宽，x
	volume_height = 300;//图像长，y
	volume_depth = 20;//图像高，z
	projection_multiplier = new float[1];
	projection_multiplier[0] = 2 * pi / number_of_projections;
	volume_spacing_ptr = new float[3];
	stepsize = new float[1];
	//根据自己的项目修改以下参数
	SOD = 600;//焦距
	SDD = 815;
	stepsize[0] = 0.2;	//采样步长
	phi = 45;
	volume_origin_ptr = new float[3];
	volume_origin_ptr[0] = -volume_width / 2 * dd;
	volume_origin_ptr[1] = -volume_height / 2 * dd;
	volume_origin_ptr[2] = -volume_depth / 2 * dd;
	detector_origin_ptr = new float[2];
	detector_origin_ptr[0] = -detector_width / 2 * dd;
	detector_origin_ptr[1] = -detector_height / 2 * dd;
	projection_matrices = new float[number_of_projections * 9];
	//重建图像间隔
	volume_spacing_ptr[0] = dd;
	volume_spacing_ptr[1] = dd;
	volume_spacing_ptr[2] = dd;

	//读取头部文件
	head = cv::Mat(volume_width * volume_height * volume_depth, 1, CV_32FC1);//分配投影内存
	res = cv::Mat(volume_width * volume_height * volume_depth, 1, CV_32FC1);//分配投影内存
	memset(head.ptr<float>(0), 0, volume_width * volume_height * volume_depth*sizeof(float));
	memset(res.ptr<float>(0), 0, volume_width * volume_height * volume_depth*sizeof(float));
	FILE *fp1;
	fp1 = fopen("F:/研究生/代码/CTConstructionCuda/ARTCuda/IncART/Shepp-Logan_3d_256.bin", "rb");
	fseek(fp1, 0, SEEK_SET);
	int prosize = volume_width * volume_height * volume_depth;
	fread(head.ptr<float>(0), sizeof(float), prosize, fp1);
	fclose(fp1);

	/*for (int i = 0; i < volume_depth; i++)
	{
		for (int j = 0; j < volume_height; j++)
		{
			for (int k = 0; k < volume_width; k++)
			{
				if (i / 16 % 2 == 0&sqrt((j-128)*(j-128)+(k-128)*(k-128))<120)
				{
				head.at<float>(i * volume_height * volume_width + j * volume_width + k) = 255;
				}
			}
		}
	}*/
	

	for (int i = 0; i < volume_depth; i++)
	{
		for (int j = 0; j < volume_height; j++)
		{
			for (int k = 0; k < volume_width; k++)
			{
				if (i<4 | i>15)
				{
					head.at<float>(i * volume_height * volume_width + j * volume_width + k) = 0.25;
					if (j<22 | j>300 - 22 | k<22 | k>300 - 22)head.at<float>(i * volume_height * volume_width + j * volume_width + k) = 0;
				}
				else{
					//1号 竖着的长方体 密度=0.75
					if ((j>24 & j<(256 - 25) & k>9 & k<(9 + 13)) | (j>24 & j<(256 - 25) & k>41 & k<(41 + 13)) | (j>24 & j<(256 - 25) & k>73 & k<(73 + 13)))
					{
						j = j + 22; k = k + 22;
						head.at<float>(i * volume_height * volume_width + j * volume_width + k) = 0.75;
						j = j - 22; k = k - 22;
					}
					//2号 横着的长方体 密度=0.75
					else if ((j > 24 & j < (24 + 13) & k>105 & k < (105 + 141)) | (j>61 & j < (61 + 13) & k>105 & k < (105 + 141)) | (j>98 & j < (98 + 13) & k>105 & k < (105 + 141)))
					{
						j = j + 22; k = k + 22;
						head.at<float>(i * volume_height * volume_width + j * volume_width + k) = 0.75;
						j = j - 22; k = k - 22;
					}
					//3号 球体 密度=1
					else if (pow((k - 120), 2) + pow((j - 125), 2) + pow((i - 9.5), 2) <= 25 | pow((k - 160), 2) + pow((j - 125), 2) + pow((i - 9.5), 2) <= 25 | pow((k - 200), 2) + pow((j - 125), 2) + pow((i - 9.5), 2) <= 25
						| pow((k - 120), 2) + pow((j - 165), 2) + pow((i - 9.5), 2) <= 25 | pow((k - 160), 2) + pow((j - 165), 2) + pow((i - 9.5), 2) <= 25 | pow((k - 200), 2) + pow((j - 165), 2) + pow((i - 9.5), 2) <= 25
						| pow((k - 120), 2) + pow((j - 205), 2) + pow((i - 9.5), 2) <= 25 | pow((k - 160), 2) + pow((j - 205), 2) + pow((i - 9.5), 2) <= 25 | pow((k - 200), 2) + pow((j - 205), 2) + pow((i - 9.5), 2) <= 25)
					{
						j = j + 22; k = k + 22;
						head.at<float>(i * volume_height * volume_width + j * volume_width + k) = 1;
						j = j - 22; k = k - 22;
					}
					else{
						j = j + 22; k = k + 22;
						head.at<float>(i * volume_height * volume_width + j * volume_width + k) = 0.25;
						j = j - 22; k = k - 22;
					}
					if (j<22 | j>300 - 22 | k<22 | k>300 - 22)head.at<float>(i * volume_height * volume_width + j * volume_width + k) = 0;
				}
			}
		}
	}

	/*ofstream outFile("simulate_pcb300.bin", ios::out | ios::binary);
	for (int i = 0; i < volume_width; i++)
	{
		for (int j = 0; j < volume_height; j++)
		{
			for (int k = 0; k < volume_depth; k++)
			{
				outFile.write((char*)head.ptr<float>(i * volume_height * volume_depth + j * volume_depth + k), sizeof(float));
			}
		}
	}
	outFile.close();*/

	//投影权重
	W = cv::Mat(detector_width * detector_height * number_of_projections, 1, CV_32FC1);
	V = cv::Mat(volume_height*volume_width*number_of_projections, 1, CV_32FC1);
	memset(V.ptr<float>(0), 0, volume_height*volume_width*number_of_projections*sizeof(float));
	//投影数据
	proj = cv::Mat(detector_width * detector_height * number_of_projections, 1, CV_32FC1);
	Ax = cv::Mat(detector_width * detector_height * 1, 1, CV_32FC1);

	/*****************************************真实投影值***********************************************/
	//投影矩阵，将x、y坐标转换为S
	for (int i = 0; i < 9 * number_of_projections; i = i + 9)
	{
		float delta = i / 9 * 360 / (number_of_projections - 1) * pi / 180;
		projection_matrices[i] = cos(delta);
		projection_matrices[i + 1] = -sin(delta)*cos(phi * pi / 180);
		projection_matrices[i + 2] = SDD * sin(delta)*sin(phi * pi / 180);
		projection_matrices[i + 3] = sin(delta);
		projection_matrices[i + 4] = cos(delta)*cos(phi * pi / 180);
		projection_matrices[i + 5] = -SDD * cos(delta)*sin(phi*pi / 180);
		projection_matrices[i + 6] = 0;
		projection_matrices[i + 7] = -sin(phi*pi / 180);
		projection_matrices[i + 8] = -SDD*cos(phi*pi / 180);
	}

	src_ptr = new float[number_of_projections * 3];
	for (int i = 0; i < 3 * number_of_projections; i = i + 3)
	{
		float delta = i / 3 * 360 / (number_of_projections - 1) * pi / 180;
		src_ptr[i + 0] = -SOD * sin(delta)*sin(phi * pi / 180);
		src_ptr[i + 1] = SOD * cos(delta)*sin(phi * pi / 180);
		src_ptr[i + 2] = SOD * cos(phi * pi / 180);
	}
	CPMalloc(volume_width, volume_height, volume_depth, number_of_projections, detector_width, detector_height);
	CPMemcpy(proj.ptr<float>(0), head.ptr<float>(0), volume_spacing_ptr, stepsize, volume_origin_ptr, detector_origin_ptr, detector_width, detector_height, volume_width, volume_height, volume_depth, number_of_projections);
	CPConeProjection(projection_matrices, src_ptr, number_of_projections, volume_width, volume_height, volume_depth, detector_width, detector_height, dd);
	CPOutput(proj.ptr<float>(0), detector_width, detector_height, number_of_projections);
	CPFree();

	/*FILE *fp2;
	fp2 = fopen("F:/研究生/代码/CTConstructionCuda/ARTCuda/IncART/projections.bin", "rb");
	fseek(fp2, 0, SEEK_SET);
	prosize = detector_width * detector_height * number_of_projections;
	fread(proj.ptr<float>(0), sizeof(float), prosize, fp2);
	fclose(fp2);*/

	/*ofstream outFile("sim_proj.bin", ios::out | ios::binary);
	for (int i = 0; i < number_of_projections; i++)
	{
		for (int j = 0; j < detector_height; j++)
		{
			for (int k = 0; k < detector_width; k++)
			{
				outFile.write((char*)proj.ptr<float>(i * detector_height * detector_width + j * detector_width + k), sizeof(float));
			}
		}
	}
	outFile.close();*/
	/*************************************************************************************************/

	//fstream File("sim_proj.bin", ios::in | ios::binary);
	//if (!File.is_open()) {
	//	cout << "error" << endl;
	//}
	////读取投影数据
	//for (int i = 0; i < number_of_projections; i++)
	//{
	//	for (int j = 0; j < detector_height; j++)
	//	{
	//		for (int k = 0; k < detector_width; k++)
	//		{
	//			File.read((char*)proj.ptr<float>(i * detector_height * detector_width + j * detector_width + k), sizeof (float));
	//		}
	//	}
	//}
	//File.close();

	//fstream File("TAwTV/TAwTV.bin", ios::in | ios::binary);
	//if (!File.is_open()) {
	//	cout << "error" << endl;
	//}
	////读取投影数据
	//for (int i = 0; i < volume_depth; i++)
	//{
	//	for (int j = 0; j < volume_height; j++)
	//	{
	//		for (int k = 0; k < volume_width; k++)
	//		{
	//			File.read((char*)res.ptr<float>(i * volume_height * volume_width + j * volume_width + k), sizeof(float));
	//		}
	//	}
	//}
	//File.close();
}

void IncART::GetAx(int angle)
{
	//投影矩阵，将x、y坐标转换为S
	float theta_rad = angle * 360 / (number_of_projections - 1) * pi / 180;
	float phi_rad = phi * pi / 180;
	projection_matrices[0] = cos(theta_rad);
	projection_matrices[0 + 1] = -sin(theta_rad)*cos(phi_rad);
	projection_matrices[0 + 2] = SDD * sin(theta_rad)*sin(phi_rad);
	projection_matrices[0 + 3] = sin(theta_rad);
	projection_matrices[0 + 4] = cos(theta_rad)*cos(phi_rad);
	projection_matrices[0 + 5] = -SDD * cos(theta_rad)*sin(phi_rad);
	projection_matrices[0 + 6] = 0;
	projection_matrices[0 + 7] = -sin(phi_rad);
	projection_matrices[0 + 8] = -SDD*cos(phi_rad);

	src_ptr[0 + 0] = -SOD * sin(theta_rad)*sin(phi_rad);
	src_ptr[0 + 1] = SOD * cos(theta_rad)*sin(phi_rad);
	src_ptr[0 + 2] = SOD * cos(phi_rad);

	CPMalloc(volume_width, volume_height, volume_depth, 1, detector_width, detector_height);
	CPMemcpy(Ax.ptr<float>(0), res.ptr<float>(0), volume_spacing_ptr, stepsize, volume_origin_ptr, detector_origin_ptr, detector_width, detector_height, volume_width, volume_height, volume_depth, 1);
	CPConeProjection(projection_matrices, src_ptr, 1, volume_width, volume_height, volume_depth, detector_width, detector_height, dd);
	CPOutput(Ax.ptr<float>(0), detector_width, detector_height, 1);
	CPFree();
}

cv::Mat IncART::GetAtb(cv::Mat projDiffer, int angle)
{
	cv::Mat resDiffer = cv::Mat(volume_width * volume_height * volume_depth, 1, CV_32FC1);//分配投影内存
	memset(resDiffer.ptr<float>(0), 0, volume_width * volume_height * volume_depth*sizeof(float));

	float theta_rad = angle * 360 / (number_of_projections - 1) * pi / 180;
	float phi_rad = -phi * pi / 180;
	projection_matrices[0] = -SDD*cos(theta_rad);
	projection_matrices[1] = -SDD*sin(theta_rad);
	projection_matrices[2] = 0;
	projection_matrices[3] = 0;  

	projection_matrices[4] = SDD*sin(theta_rad)*cos(phi_rad);
	projection_matrices[5] = -SDD *cos(theta_rad)*cos(phi_rad);
	projection_matrices[6] = -SDD*sin(phi_rad);
	projection_matrices[7] = 0;

	projection_matrices[8] = sin(theta_rad) * sin(phi_rad);
	projection_matrices[9] = -cos(theta_rad) * sin(phi_rad);
	projection_matrices[10] = cos(phi_rad);
	projection_matrices[11] = -SOD;

	CBMalloc(volume_width, volume_height, volume_depth, 1);
	CBMemcpy(resDiffer.ptr<float>(0), projection_matrices, volume_spacing_ptr, volume_origin_ptr, detector_origin_ptr, projection_multiplier, volume_width, volume_height, volume_depth, 1);
	CBConeBackprojection3D(projDiffer.ptr<float>(0), 1, volume_width, volume_height, volume_depth, detector_width, detector_height, SOD, dd);
	CBOutput(resDiffer.ptr<float>(0), volume_width, volume_height, volume_depth);
	CBFree();
	return resDiffer;
}

cv::Mat IncART::GetW()
{
	cv::Mat head_ones = cv::Mat::ones(volume_width * volume_height * volume_depth, 1, CV_32FC1);//分配投影内存
	cv::Mat W_temp = cv::Mat::zeros(detector_width * detector_height * number_of_projections, 1, CV_32FC1);//一个角度下的投影

	//投影矩阵，将x、y坐标转换为S
	for (int i = 0; i < 9 * number_of_projections; i = i + 9)
	{
		float delta = i / 9 * 360 / (number_of_projections - 1)*pi / 180;
		projection_matrices[i] = cos(delta);
		projection_matrices[i + 1] = -sin(delta)*cos(phi * pi / 180);
		projection_matrices[i + 2] = SDD * sin(delta)*sin(phi * pi / 180);
		projection_matrices[i + 3] = sin(delta);
		projection_matrices[i + 4] = cos(delta)*cos(phi * pi / 180);
		projection_matrices[i + 5] = -SDD * cos(delta)*sin(phi*pi / 180);
		projection_matrices[i + 6] = 0;
		projection_matrices[i + 7] = -sin(phi*pi / 180);
		projection_matrices[i + 8] = -SDD*cos(phi*pi / 180);
	}

	for (int i = 0; i < 3 * number_of_projections; i = i + 3)
	{
		float delta = i / 3 * 360 / (number_of_projections - 1) * pi / 180;
		src_ptr[i + 0] = -SOD * sin(delta)*sin(phi * pi / 180);
		src_ptr[i + 1] = SOD * cos(delta)*sin(phi * pi / 180);
		src_ptr[i + 2] = SOD * cos(phi * pi / 180);
	}

	CPMalloc(volume_width, volume_height, volume_depth, number_of_projections, detector_width, detector_height);
	CPMemcpy(W_temp.ptr<float>(0), head_ones.ptr<float>(0), volume_spacing_ptr, stepsize, volume_origin_ptr, detector_origin_ptr, detector_width, detector_height, volume_width, volume_height, volume_depth, number_of_projections);
	CPConeProjection(projection_matrices, src_ptr, number_of_projections, volume_width, volume_height, volume_depth, detector_width, detector_height, dd);
	CPOutput(W_temp.ptr<float>(0), detector_width, detector_height, number_of_projections);
	CPFree();
	return W_temp;
}

cv::Mat IncART::GetV()
{
	for (int i = 0; i < number_of_projections; i++)
	{
		float beta = i * 360 / (number_of_projections - 1) * pi / 180;
		for (int k = 0; k < volume_width; k++)
		{
			for (int j = 0; j < volume_height; j++)
			{
				V.at<float>(i*volume_width*volume_height + j*volume_width + k) = V.at<float>(i*volume_width*volume_height + j*volume_width + k)
					+ pow(SOD / (SOD + (k + volume_origin_ptr[0]/dd + 0.5) * dd * sin(beta) - (j + volume_origin_ptr[1]/dd + 0.5) * dd * cos(beta)), 2);
			}
		}
	}
	return V;
}

void IncART::ART()
{
	W = GetW();	//反投影权重
	V = GetV();
	W = 1 / W;
	V = 1 / V;
	string psnr_name = file_prefix + "_PSNR_SaveFile.bin";
	string ssim_name = file_prefix + "_SSIM_SaveFile.bin";

	ofstream outFile_PSNR(psnr_name, ios::out | ios::binary);
	ofstream outFile_SSIM(ssim_name, ios::out | ios::binary);

	for (int i = 0; i < niter; i++)
	{
		cv::Mat Temp = res.clone();	//更新前的重建图像
		cv::Mat singleProj = cv::Mat::zeros(detector_width * detector_height, 1, CV_32FC1);//一个角度下的投影
		cv::Mat singleW = cv::Mat::zeros(detector_width * detector_height, 1, CV_32FC1);
		cv::Mat singleV = cv::Mat::zeros(volume_height * volume_width, 1, CV_32FC1);
		cv::Mat singlesV = cv::Mat::zeros(volume_depth * volume_height * volume_width, 1, CV_32FC1);

		for (int j = 0; j < number_of_projections; j++)
		{
			memcpy(singleW.data, W.ptr<float>(j * detector_width * detector_height), detector_width * detector_height * sizeof(float));

			memcpy(singleV.data, V.ptr<float>(j * volume_width * volume_height), volume_width * volume_height * sizeof(float));

			//将一层权重复制volume_height层，类似广播机制
			for (int jj = 0; jj < volume_height; jj++)
			{
				for (int k = 0; k < volume_width; k++)
				{
					for (int z = 0; z < volume_depth; z++)
					{
						singlesV.at<float>(z*volume_width*volume_height + jj*volume_width + k) = singleV.at<float>(jj*volume_width + k);
					}
				}
			}

			memcpy(singleProj.data, proj.ptr<float>(j * detector_width * detector_height), detector_width * detector_height * sizeof(float));

			GetAx(j);  //对重建图像res进行正投影得到Ax

			/*************************************************************************************
			proj_err = singleProj - Ax				(b-Ax)
			weighted_err = singleW.mul(proj_err)	W^-1 * (b-Ax)
			backprj = GetAtb(weighted_err)			At * W^-1 * (b-Ax)
			backprj=V.mul(backprj)					V * At * W^-1 * (b-Ax)
			res=res+lambda*backprj; 				x= x + lambda * V * At * W^-1 * (b-Ax)
			*************************************************************************************/
			res = res + lambda * singlesV.mul(GetAtb(singleW.mul(singleProj - Ax), j));
			res = cv::max(res, 0);
		}
		cv::Mat dp_vec = Temp - res;
		float dp = cv::norm(dp_vec, 2);
		float d_TV_lambda = dp * TV_lambda;
		cv::Mat Temp_TV = res.clone();
		const long imageSize[3] = { volume_width, volume_height, volume_depth };
		pocs_tv(res.ptr<float>(0), res.ptr<float>(0), TV_lambda, imageSize, TV_niter, IsHOTV, IsRwTV, delta);
		cv::Mat dg_vec = Temp_TV - res;
		float dg = cv::norm(dg_vec, 2);
		float c = sum(sum(dg_vec.mul(dp_vec)))[0] / (dg*dp);
		if (c < -0.99)break;
		/***********************计算PSNR和SSIM*********************************************/
		cv::Mat image_ref, image_obj;
		image_ref = cv::Mat(volume_width, volume_height, CV_32FC1);
		memset(image_ref.ptr<float>(0), 0, volume_width * volume_height*sizeof(float));
		memcpy(image_ref.data, res.ptr<float>(volume_depth / 2 * volume_width * volume_height), volume_width * volume_height * sizeof(float));
		normalize(image_ref, image_ref, 0, 255, cv::NORM_MINMAX);

		image_obj = cv::Mat(volume_width, volume_height, CV_32FC1);
		memset(image_obj.ptr<float>(0), 0, volume_width * volume_height*sizeof(float));
		memcpy(image_obj.data, head.ptr<float>(volume_depth / 2 * volume_width * volume_height), volume_width * volume_height * sizeof(float));
		normalize(image_obj, image_obj, 0, 255, cv::NORM_MINMAX);

		double* psnr = mypsnr(image_ref, image_obj);
		double* ssim = myssim(image_ref, image_obj);
		outFile_PSNR.write((char*)psnr, sizeof(double));
		outFile_SSIM.write((char*)ssim, sizeof(double));
		/********************************************************************************/
		char filename[100];
		sprintf_s(filename, "%d.bin", i);

		ofstream outFile(file_prefix + filename, ios::out | ios::binary);
		for (int i = 0; i < volume_width; i++)
		{
			for (int j = 0; j < volume_height; j++)
			{
				for (int k = 0; k < volume_depth; k++)
				{
					outFile.write((char*)res.ptr<float>(i * volume_height * volume_depth + j * volume_depth + k), sizeof(float));
				}
			}
		}
		outFile.close();
	}
	outFile_PSNR.close();
	outFile_SSIM.close();
}

IncART::~IncART()
{
	head.release();	//真实头部模型
	res.release();	//迭代头部模型
	Ax.release();		//正投影结果
	W.release();		//真实投影和计算投影差值对图像各个像素的更新权重，即正投影权重
	V.release();		//反投影权重
	proj.release();	//真实投影值

}
double* IncART::mypsnr(cv::Mat image_ref, cv::Mat image_obj)
 {
	double mse = 0;
	double div = 0;
	int width = image_ref.cols;
	int height = image_ref.rows;
	for (int v = 0; v < height; v++)
	{
		for (int u = 0; u < width; u++)
		{
			div = image_ref.at<float>(v, u) - image_obj.at<float>(v, u);
			mse += div*div;
		}
	}
	mse = mse / (width*height);
	*PSNR = 10 * log10(255 * 255 / mse);
	return PSNR;
}

double* IncART::myssim(cv::Mat image_ref, cv::Mat image_obj)
{
	double C1 = 6.5025, C2 = 58.5225;

	int width = image_ref.cols;
	int height = image_ref.rows;
	double mean_x = 0;
	double mean_y = 0;
	double sigma_x = 0;
	double sigma_y = 0;
	double sigma_xy = 0;
	for (int v = 0; v < height; v++)
	{
		for (int u = 0; u < width; u++)
		{
			mean_x += image_ref.at<float>(v, u);
			mean_y += image_obj.at<float>(v, u);
		}
	}
	mean_x = mean_x / width / height;
	mean_y = mean_y / width / height;
	for (int v = 0; v < height; v++)
	{
		for (int u = 0; u < width; u++)
		{
			sigma_x += (image_ref.at<float>(v, u) - mean_x)* (image_ref.at<float>(v, u) - mean_x);
			sigma_y += (image_obj.at<float>(v, u) - mean_y)* (image_obj.at<float>(v, u) - mean_y);
			sigma_xy += abs((image_ref.at<float>(v, u) - mean_x)* (image_obj.at<float>(v, u) - mean_y));
		}
	}
	sigma_x = sigma_x / (width*height - 1);
	sigma_y = sigma_y / (width*height - 1);
	sigma_xy = sigma_xy / (width*height - 1);
	double fenzi = (2 * mean_x*mean_y + C1) * (2 * sigma_xy + C2);
	double fenmu = (mean_x*mean_x + mean_y * mean_y + C1) * (sigma_x + sigma_y + C2);
	*SSIM = fenzi / fenmu;
	return SSIM;
}
