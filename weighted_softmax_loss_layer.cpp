#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/weighted_softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void WeightedSoftmaxWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter weighted_softmax_param(this->layer_param_);
  weighted_softmax_param.set_type("Softmax");
  weighted_softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(weighted_softmax_param);
  weighted_softmax_bottom_vec_.clear();
  weighted_softmax_bottom_vec_.push_back(bottom[0]);
  weighted_softmax_top_vec_.clear();
  weighted_softmax_top_vec_.push_back(&prob_);
  weighted_softmax_layer_->SetUp(weighted_softmax_bottom_vec_, weighted_softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }
}

template <typename Dtype>
void WeightedSoftmaxWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  weighted_softmax_layer_->Reshape(weighted_softmax_bottom_vec_, weighted_softmax_top_vec_);
  weighted_softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, weighted_softmax_axis_);
  inner_num_ = bottom[0]->count(weighted_softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if weighted_softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  //CHECK_EQ(bottom[0].shape(weighted_softmax_axis_), 2) << " WeightedSoftmaxWithLossLayer only support 2 class nowadays ! "; 
  if (top.size() >= 2) {
    // weighted_softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
Dtype WeightedSoftmaxWithLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
        normalizer = Dtype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void WeightedSoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the weighted_softmax prob values.
  weighted_softmax_layer_->Forward(weighted_softmax_bottom_vec_, weighted_softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int dim = prob_.count() / outer_num_;
  Dtype loss_pos = 0;
  Dtype loss_neg = 0;
  count_pos_ = 0;
  count_neg_ = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      //if (has_ignore_label_ && label_value == ignore_label_) {
      //  continue;
      //}
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.shape(weighted_softmax_axis_));

	  Dtype tmp_loss = log(std::max(prob_data[i * dim + label_value * inner_num_ + j], Dtype(FLT_MIN)));
	  if (label_value > 0) {
		count_pos_ += 1; 
		loss_pos -= tmp_loss;
	  } else {
		count_neg_ += 1;
		loss_neg -= tmp_loss;
	  }
    }	
  }

  Dtype count_all = (count_pos_ + count_neg_);
  if (count_pos_ > 0) {
	loss_pos = loss_pos * (count_all / 2.0) / count_pos_; 
  }
  if (count_neg_ > 0) {
	loss_neg = loss_neg * (count_all / 2.0) / count_neg_; 
  }

  //top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);
  top[0]->mutable_cpu_data()[0] = (loss_pos + loss_neg) / count_all;
  
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void WeightedSoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int dim = prob_.count() / outer_num_;
	int channel = bottom[0]->shape(1);
	Dtype w = 0;
    Dtype count_all = count_pos_ + count_neg_;
	CHECK_EQ(count_all, outer_num_ * inner_num_) << " weighted_softmax_loss : count_all should equals to outer_num_ * inner_num_";
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        //if (has_ignore_label_ && label_value == ignore_label_) {
        //  for (int c = 0; c < bottom[0]->shape(weighted_softmax_axis_); ++c) {
        //    bottom_diff[i * dim + c * inner_num_ + j] = 0;
        //  }
        //} else {
		
		  bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
		  
		  if (label_value > 0) {
			w = count_all / 2.0 / count_pos_;
		  } else {
			w = count_all / 2.0 / count_neg_;
		  }
		
		  for (int k = 0; k < channel; ++k) {
			bottom_diff[i * dim + k * inner_num_ + j] *= w;
	      }
          
		  //++count;
        //}
      }
    }

    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / count_all;
    caffe_scal(prob_.count(), loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(WeightedSoftmaxWithLossLayer);
#endif

INSTANTIATE_CLASS(WeightedSoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(WeightedSoftmaxWithLoss);

}  // namespace caffe
