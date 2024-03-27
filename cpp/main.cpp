#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
// #include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#include "util.h"

using namespace cv;
using namespace std;
using namespace Ort;

class YOWOv2
{
public:
	YOWOv2(string modelpath, const string dataset = "ava_v2.2", const float nms_thresh_ = 0.5, const float conf_thresh_ = 0.1, const bool act_pose_ = false);
	vector<int> detect_multi_hot(vector<Mat> video_clip, vector<Bbox> &boxes, vector<float> &det_conf, vector<vector<float>> &cls_conf);
	vector<int> detect_one_hot(vector<Mat> video_clip, vector<Bbox> &boxes, vector<float> &det_conf, vector<int> &cls_id);
	int len_clip;
	bool multi_hot;

private:
	vector<float> input_tensor;
	void preprocess(vector<Mat> video_clip);
	int inpWidth;
	int inpHeight;
	float nms_thresh;
	float conf_thresh;

	int num_class;
	const int topk = 40;
	const int strides[3] = {8, 16, 32};
	bool act_pose;
	void generate_proposal_multi_hot(const int stride, const float *conf_pred, const float *cls_pred, const float *reg_pred, vector<Bbox> &boxes, vector<float> &det_conf, vector<vector<float>> &cls_conf);
	void generate_proposal_one_hot(const int stride, const float *conf_pred, const float *cls_pred, const float *reg_pred, vector<Bbox> &boxes, vector<float> &det_conf, vector<int> &cls_id);

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "Video Action Detect");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char *> input_names;
	vector<char *> output_names;
	vector<vector<int64_t>> input_node_dims;  // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
	Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
};

YOWOv2::YOWOv2(string modelpath, const string dataset, const float nms_thresh_, const float conf_thresh_, const bool act_pose_)
{

	/// OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);   ///use cuda

	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	/*std::wstring widestr = std::wstring(modelpath.begin(), modelpath.end()); ////windows
	ort_session = new Session(env, widestr.c_str(), sessionOptions); ////windows*/
	ort_session = new Session(env, modelpath.c_str(), sessionOptions); ////linux

	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}

	this->len_clip = this->input_node_dims[0][2];
	this->inpHeight = this->input_node_dims[0][3];
	this->inpWidth = this->input_node_dims[0][4];
	if (dataset == "ava_v2.2" || dataset == "ava")
	{
		this->num_class = 80;
		this->multi_hot = true;
	}
	else
	{
		this->num_class = 24;
		this->multi_hot = false;
	}
	this->conf_thresh = conf_thresh_;
	this->nms_thresh = nms_thresh_;
	this->act_pose = act_pose_;
}

void YOWOv2::preprocess(vector<Mat> video_clip)
{
	const int image_area = this->inpHeight * this->inpWidth;
	this->input_tensor.resize(1 * 3 * this->len_clip * image_area);
	size_t single_chn_size = image_area * sizeof(float);
	const int chn_area = this->len_clip * image_area;
	for (int i = 0; i < this->len_clip; i++)
	{
		Mat resizeimg;
		resize(video_clip[i], resizeimg, cv::Size(this->inpWidth, this->inpHeight));
		resizeimg.convertTo(resizeimg, CV_32FC3);
		vector<cv::Mat> bgrChannels(3);
		split(resizeimg, bgrChannels);

		memcpy(this->input_tensor.data() + i * image_area, (float *)bgrChannels[0].data, single_chn_size);
		memcpy(this->input_tensor.data() + chn_area + i * image_area, (float *)bgrChannels[1].data, single_chn_size);
		memcpy(this->input_tensor.data() + 2 * chn_area + i * image_area, (float *)bgrChannels[2].data, single_chn_size);
	}
}

void YOWOv2::generate_proposal_multi_hot(const int stride, const float *conf_pred, const float *cls_pred, const float *reg_pred, vector<Bbox> &boxes, vector<float> &det_conf, vector<vector<float>> &cls_conf)
{
	const int feat_h = (int)ceil((float)this->inpHeight / stride);
	const int feat_w = (int)ceil((float)this->inpWidth / stride);
	const int area = feat_h * feat_w;
	vector<float> conf_pred_i(area);
	for (int i = 0; i < area; i++)
	{
		conf_pred_i[i] = sigmoid(conf_pred[i]);
	}
	vector<int> topk_inds = TopKIndex(conf_pred_i, this->topk);
	int length = this->num_class;
	if (this->act_pose)
	{
		length = 14;
	}

	for (int i = 0; i < topk_inds.size(); i++)
	{
		const int ind = topk_inds[i];
		if (conf_pred_i[ind] > this->conf_thresh)
		{
			int row = 0, col = 0;
			ind2sub(ind, feat_w, feat_h, row, col);

			float cx = (col + 0.5f + reg_pred[ind * 4]) * stride;
			float cy = (row + 0.5f + reg_pred[ind * 4 + 1]) * stride;
			float w = exp(reg_pred[ind * 4 + 2]) * stride;
			float h = exp(reg_pred[ind * 4 + 3]) * stride;
			boxes.emplace_back(Bbox{int(cx - 0.5 * w), int(cy - 0.5 * h), int(cx + 0.5 * w), int(cy + 0.5 * h)});
			det_conf.emplace_back(conf_pred_i[ind]);

			vector<float> cls_conf_i(length);
			for (int j = 0; j < length; j++)
			{
				cls_conf_i[j] = sigmoid(cls_pred[ind * this->num_class + j]);
			}
			cls_conf.emplace_back(cls_conf_i);
		}
	}
}

void YOWOv2::generate_proposal_one_hot(const int stride, const float *conf_pred, const float *cls_pred, const float *reg_pred, vector<Bbox> &boxes, vector<float> &det_conf, vector<int> &cls_id)
{
	const int feat_h = (int)ceil((float)this->inpHeight / stride);
	const int feat_w = (int)ceil((float)this->inpWidth / stride);
	const int area = feat_h * feat_w;
	vector<float> det_scores_i(area * this->num_class);
	for (int i = 0; i < area; i++)
	{
		for (int j = 0; j < this->num_class; j++)
		{
			det_scores_i[i * this->num_class + j] = sqrt(sigmoid(conf_pred[i]) * sigmoid(cls_pred[i * this->num_class + j]));
		}
	}
	const int num_topk = min(this->topk, area);
	vector<int> topk_inds = TopKIndex(det_scores_i, num_topk);
	for (int i = 0; i < topk_inds.size(); i++)
	{
		const int ind = topk_inds[i];
		if (det_scores_i[ind] > this->conf_thresh)
		{
			det_conf.emplace_back(det_scores_i[ind]);
			const int idx = ind % this->num_class;
			cls_id.emplace_back(idx);

			const int row_ind = ind / this->num_class;
			int row = 0, col = 0;
			ind2sub(row_ind, feat_w, feat_h, row, col);
			float cx = (col + 0.5f + reg_pred[row_ind * 4]) * stride;
			float cy = (row + 0.5f + reg_pred[row_ind * 4 + 1]) * stride;
			float w = exp(reg_pred[row_ind * 4 + 2]) * stride;
			float h = exp(reg_pred[row_ind * 4 + 3]) * stride;
			boxes.emplace_back(Bbox{int(cx - 0.5 * w), int(cy - 0.5 * h), int(cx + 0.5 * w), int(cy + 0.5 * h)});
		}
	}
}

vector<int> YOWOv2::detect_multi_hot(vector<Mat> video_clip, vector<Bbox> &boxes, vector<float> &det_conf, vector<vector<float>> &cls_conf)
{
	if (int(video_clip.size()) != this->len_clip)
	{
		cout << "input frame number is not " << this->len_clip << endl;
		exit(-1);
	}
	const int origin_h = video_clip[0].rows;
	const int origin_w = video_clip[0].cols;
	this->preprocess(video_clip);

	std::vector<int64_t> input_img_shape = {1, 3, this->len_clip, this->inpHeight, this->inpWidth};
	Value input_tensor_ = Value::CreateTensor<float>(memory_info_handler, this->input_tensor.data(), this->input_tensor.size(), input_img_shape.data(), input_img_shape.size());

	Ort::RunOptions runOptions;
	vector<Value> ort_outputs = this->ort_session->Run(runOptions, this->input_names.data(), &input_tensor_, this->input_names.size(), this->output_names.data(), this->output_names.size());
	const float *conf_preds0 = ort_outputs[0].GetTensorMutableData<float>();
	const float *conf_preds1 = ort_outputs[1].GetTensorMutableData<float>();
	const float *conf_preds2 = ort_outputs[2].GetTensorMutableData<float>();
	const float *cls_preds0 = ort_outputs[3].GetTensorMutableData<float>();
	const float *cls_preds1 = ort_outputs[4].GetTensorMutableData<float>();
	const float *cls_preds2 = ort_outputs[5].GetTensorMutableData<float>();
	const float *reg_preds0 = ort_outputs[6].GetTensorMutableData<float>();
	const float *reg_preds1 = ort_outputs[7].GetTensorMutableData<float>();
	const float *reg_preds2 = ort_outputs[8].GetTensorMutableData<float>();

	this->generate_proposal_multi_hot(this->strides[0], conf_preds0, cls_preds0, reg_preds0, boxes, det_conf, cls_conf);
	this->generate_proposal_multi_hot(this->strides[1], conf_preds1, cls_preds1, reg_preds1, boxes, det_conf, cls_conf);
	this->generate_proposal_multi_hot(this->strides[2], conf_preds2, cls_preds2, reg_preds2, boxes, det_conf, cls_conf);

	vector<int> keep_inds = multiclass_nms_class_agnostic(boxes, det_conf, this->nms_thresh);

	const int max_hw = max(this->inpHeight, this->inpWidth);
	const float ratio_h = float(origin_h) / max_hw;
	const float ratio_w = float(origin_w) / max_hw;
	for (int i = 0; i < keep_inds.size(); i++)
	{
		const int ind = keep_inds[i];
		boxes[ind].xmin = int((float)boxes[ind].xmin * ratio_w);
		boxes[ind].ymin = int((float)boxes[ind].ymin * ratio_h);
		boxes[ind].xmax = int((float)boxes[ind].xmax * ratio_w);
		boxes[ind].ymax = int((float)boxes[ind].ymax * ratio_h);
	}
	return keep_inds;
}

vector<int> YOWOv2::detect_one_hot(vector<Mat> video_clip, vector<Bbox> &boxes, vector<float> &det_conf, vector<int> &cls_id)
{
	if (int(video_clip.size()) != this->len_clip)
	{
		cout << "input frame number is not " << this->len_clip << endl;
		exit(-1);
	}
	const int origin_h = video_clip[0].rows;
	const int origin_w = video_clip[0].cols;
	this->preprocess(video_clip);

	std::vector<int64_t> input_img_shape = {1, 3, this->len_clip, this->inpHeight, this->inpWidth};
	Value input_tensor_ = Value::CreateTensor<float>(memory_info_handler, this->input_tensor.data(), this->input_tensor.size(), input_img_shape.data(), input_img_shape.size());

	Ort::RunOptions runOptions;
	vector<Value> ort_outputs = this->ort_session->Run(runOptions, this->input_names.data(), &input_tensor_, this->input_names.size(), this->output_names.data(), this->output_names.size());
	const float *conf_preds0 = ort_outputs[0].GetTensorMutableData<float>();
	const float *conf_preds1 = ort_outputs[1].GetTensorMutableData<float>();
	const float *conf_preds2 = ort_outputs[2].GetTensorMutableData<float>();
	const float *cls_preds0 = ort_outputs[3].GetTensorMutableData<float>();
	const float *cls_preds1 = ort_outputs[4].GetTensorMutableData<float>();
	const float *cls_preds2 = ort_outputs[5].GetTensorMutableData<float>();
	const float *reg_preds0 = ort_outputs[6].GetTensorMutableData<float>();
	const float *reg_preds1 = ort_outputs[7].GetTensorMutableData<float>();
	const float *reg_preds2 = ort_outputs[8].GetTensorMutableData<float>();

	this->generate_proposal_one_hot(this->strides[0], conf_preds0, cls_preds0, reg_preds0, boxes, det_conf, cls_id);
	this->generate_proposal_one_hot(this->strides[1], conf_preds1, cls_preds1, reg_preds1, boxes, det_conf, cls_id);
	this->generate_proposal_one_hot(this->strides[2], conf_preds2, cls_preds2, reg_preds2, boxes, det_conf, cls_id);
	
	vector<int> keep_inds = multiclass_nms_class_aware(boxes, det_conf, cls_id, this->nms_thresh, this->num_class);

	const int max_hw = max(this->inpHeight, this->inpWidth);
	const float ratio_h = float(origin_h) / max_hw;
	const float ratio_w = float(origin_w) / max_hw;
	for (int i = 0; i < keep_inds.size(); i++)
	{
		const int ind = keep_inds[i];
		boxes[ind].xmin = int((float)boxes[ind].xmin * ratio_w);
		boxes[ind].ymin = int((float)boxes[ind].ymin * ratio_h);
		boxes[ind].xmax = int((float)boxes[ind].xmax * ratio_w);
		boxes[ind].ymax = int((float)boxes[ind].ymax * ratio_h);
	}
	return keep_inds;
}

Mat vis_multi_hot(Mat frame, const vector<Bbox> boxes, const vector<float> det_conf, const vector<vector<float>> cls_conf, const vector<int> keep_inds, const float vis_thresh)
{
	Mat dstimg = frame.clone();
	for (int i = 0; i < keep_inds.size(); i++)
	{
		const int ind = keep_inds[i];
		vector<int> indices;
		vector<float> scores;
		for (int j = 0; j < cls_conf[ind].size(); j++)
		{
			const float det_cls_conf = sqrt(det_conf[ind] * cls_conf[ind][j]);
			if (det_cls_conf > vis_thresh)
			{
				scores.emplace_back(det_cls_conf);
				indices.emplace_back(j);
			}
		}

		if (scores.size() > 0)
		{
			int xmin = boxes[ind].xmin;
			int ymin = boxes[ind].ymin;
			int xmax = boxes[ind].xmax;
			int ymax = boxes[ind].ymax;
			rectangle(dstimg, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 255, 0), 2);

			Mat blk = Mat::zeros(frame.rows, frame.cols, CV_8UC3);
			for (int j = 0; j < indices.size(); j++)
			{
				string text = format("[%.2f] ", scores[j]) + ava_labels[indices[j]];
				int baseline = 0;
				Size text_size = getTextSize(text, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
				const int coord_x = xmin + 3;
				const int coord_y = ymin + 14 + 20 * j;
				rectangle(blk, Point(coord_x - 1, coord_y - 12), Point(coord_x + text_size.width + 1, coord_y + text_size.height - 4), Scalar(0, 255, 0), -1);
				putText(blk, text, Point(coord_x, coord_y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
			}
			addWeighted(dstimg, 1.0, blk, 0.5, 1, dstimg);
		}
	}
	return dstimg;
}

Mat vis_one_hot(Mat frame, const vector<Bbox> boxes, const vector<float> det_conf, const vector<int> cls_id, const vector<int> keep_inds, const float vis_thresh, const float text_scale)
{
	Mat dstimg = frame.clone();
	for (int i = 0; i < keep_inds.size(); i++)
	{
		const int ind = keep_inds[i];
		if (det_conf[ind] > vis_thresh)
		{
			int xmin = boxes[ind].xmin;
			int ymin = boxes[ind].ymin;
			int xmax = boxes[ind].xmax;
			int ymax = boxes[ind].ymax;
			
			string text = format("%s: %.2f ", ucf24_labels[cls_id[ind]], det_conf[ind]);
			rectangle(dstimg, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 0, 255), 2);
			int baseline = 0;
			Size text_size = getTextSize(text, FONT_HERSHEY_SIMPLEX, 1, 2, &baseline);
			rectangle(dstimg, Point(xmin, ymin - text_size.height), Point(int(xmin+text_size.width*text_scale), ymin), Scalar(0, 255, 0), -1);
			putText(dstimg, text, Point(xmin, ymin-5), FONT_HERSHEY_SIMPLEX, text_scale, Scalar(0, 0, 0), 1);
		}
	}
	return dstimg;
}

int main()
{
	YOWOv2 mynet("weights/yowo_v2_nano_ava.onnx", "ava");
	const int len_clip = mynet.len_clip;
	const float vis_thresh = 0.3;

	const string videopath = "dataset/ucf24_demo/v_Basketball_g01_c02.mp4";
	const string savepath = "result.mp4";
	cv::VideoCapture vcapture(videopath);
	if (!vcapture.isOpened())
	{
		cout << "VideoCapture,open video file failed, " << videopath << endl;
		return -1;
	}
	int height = vcapture.get(cv::CAP_PROP_FRAME_HEIGHT);
	int width = vcapture.get(cv::CAP_PROP_FRAME_WIDTH);
	int fps = vcapture.get(cv::CAP_PROP_FPS);
	int video_length = vcapture.get(cv::CAP_PROP_FRAME_COUNT);
	VideoWriter vwriter;
	vwriter.open(savepath,
				 cv::VideoWriter::fourcc('X', '2', '6', '4'),
				 fps,
				 Size(width, height));

	Mat frame;
	vector<Mat> video_clip;
	while (vcapture.read(frame))
	{
		if (frame.empty())
		{
			cout << "cv::imread source file failed, " << videopath;
			return -1;
		}

		if (video_clip.size() <= 0)
		{
			for (int i = 0; i < len_clip; i++)
			{
				video_clip.emplace_back(frame);
			}
		}
		video_clip.emplace_back(frame);
		video_clip.erase(video_clip.begin());

		if(mynet.multi_hot)
		{
			vector<Bbox> boxes;
			vector<float> det_conf;
			vector<vector<float>> cls_conf;
			vector<int> keep_inds = mynet.detect_multi_hot(video_clip, boxes, det_conf, cls_conf); ////keep_inds记录vector里面的有效检测框的序号
			Mat dstimg = vis_multi_hot(frame, boxes, det_conf, cls_conf, keep_inds, vis_thresh);
			vwriter.write(dstimg);
		}
		else
		{
			vector<Bbox> boxes;
			vector<float> det_conf;
			vector<int> cls_id;
			vector<int> keep_inds = mynet.detect_one_hot(video_clip, boxes, det_conf, cls_id); ////keep_inds记录vector里面的有效检测框的序号
			Mat dstimg = vis_one_hot(frame, boxes, det_conf, cls_id, keep_inds, vis_thresh, 0.4);
			vwriter.write(dstimg);
		}
	}
	
	vwriter.release();
	vcapture.release();
	destroyAllWindows();
}
