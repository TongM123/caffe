#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void PoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  PoolingParameter pool_param = this->layer_param_.pooling_param();
  if (pool_param.global_pooling()) {
    CHECK(!(pool_param.has_kernel_size() ||
      pool_param.has_kernel_h() || pool_param.has_kernel_w()))
      << "With Global_pooling: true Filter size cannot specified";
  } else {
    CHECK(!pool_param.has_kernel_size() !=
      !(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
    CHECK(pool_param.has_kernel_size() ||
      (pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  }
  CHECK((!pool_param.has_pad() && pool_param.has_pad_h()
      && pool_param.has_pad_w())
      || (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!pool_param.has_stride() && pool_param.has_stride_h()
      && pool_param.has_stride_w())
      || (!pool_param.has_stride_h() && !pool_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  global_pooling_ = pool_param.global_pooling();
  if (global_pooling_) {
    kernel_h_ = bottom[0]->height();
    kernel_w_ = bottom[0]->width();
  } else {
    if (pool_param.has_kernel_size()) {
      kernel_h_ = kernel_w_ = pool_param.kernel_size();
    } else {
      kernel_h_ = pool_param.kernel_h();
      kernel_w_ = pool_param.kernel_w();
    }
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!pool_param.has_pad_h()) {
    pad_h_ = pad_w_ = pool_param.pad();
  } else {
    pad_h_ = pool_param.pad_h();
    pad_w_ = pool_param.pad_w();
  }
  if (!pool_param.has_stride_h()) {
    stride_h_ = stride_w_ = pool_param.stride();
  } else {
    stride_h_ = pool_param.stride_h();
    stride_w_ = pool_param.stride_w();
  }
  if (global_pooling_) {
    CHECK(pad_h_ == 0 && pad_w_ == 0 && stride_h_ == 1 && stride_w_ == 1)
      << "With Global_pooling: true; only pad = 0 and stride = 1";
  }
  if (pad_h_ != 0 || pad_w_ != 0) {
    CHECK(this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_AVE
        || this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_MAX)
        << "Padding implemented only for average and max pooling.";
    CHECK_LT(pad_h_, kernel_h_);
    CHECK_LT(pad_w_, kernel_w_);
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  if (global_pooling_) {
    kernel_h_ = bottom[0]->height();
    kernel_w_ = bottom[0]->width();
  }
  pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
  pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
  if (pad_h_ || pad_w_) {
    // If we have padding, ensure that the last pooling starts strictly
    // inside the image (instead of at the padding); otherwise clip the last.
    if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_) {
      --pooled_height_;
    }
    if ((pooled_width_ - 1) * stride_w_ >= width_ + pad_w_) {
      --pooled_width_;
    }
    CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_);
    CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_);
  }
  top[0]->Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }
  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_MAX && top.size() == 1) {
    max_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
        pooled_width_);
  }
  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  }
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void PoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;  // suppress warnings about uninitalized variables
  Dtype* top_mask = NULL;
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:{
    // Initialize
    if (use_top_mask) {
      top_mask = top[1]->mutable_cpu_data();
      caffe_set(top_count, Dtype(-1), top_mask);
    } else {
      mask = max_idx_.mutable_cpu_data();
      caffe_set(top_count, -1, mask);
    }
    caffe_set(top_count, Dtype(-FLT_MAX), top_data);
    // The main loop
	const int num_channels = bottom[0]->channels();
	int ph, pw, hstart, wstart0, hend, wend, pool_index, index0, index, h, w, index_tmp;
	float max_tmp;
	
	const int pooled_width_const = this->pooled_width_;
	const int pooled_height_const = this->pooled_height_;
	const int width_const = this->width_;
	const int height_const = this->height_;
	
	const int pooled_fm_size = this->pooled_height_ * this->pooled_width_;
    const int fm_size = this->height_ * this->width_;

    const int pooled_batch_size = pooled_fm_size * this->channels_;
    const int batch_size = fm_size * this->channels_;

    const int height_kernel_h_ = this->height_ - this->kernel_h_;
    const int width_kernel_w_ = this->width_ - this->kernel_w_;

    const int height_kernel_h_pad_h_ = height_kernel_h_ + this->pad_h_;
    const int width_kernel_w_pad_w_ = width_kernel_w_ + this->pad_w_;
	
	// Prepare default accumulator. (and bypass memory aliasing warning...)
    //float min_float = -FLT_MAX;
    //void* min_float_ptr = &min_float;
    //uint32_t min_float_cast = *reinterpret_cast<uint32_t*>(min_float_ptr);
    //mov(stack_min_float, min_float_cast);
    //vbroadcastss(xmm15, stack_min_float);
    
    //之后可以把这两个循环拆成汇编，或者使用OpenMP优化
    for (int image = 0; image < bottom[0]->num(); ++image)
    {
        for (int channel = 0; channel < num_channels; ++channel)
        {
           /* 
            __asm volatile
            {
                ;//input: 
                ;//stack_bottom_data_ptr = bottom_data + fm_size*(layer->channels_*batch_start + channel_start)*4;
                MUL R4, batch_size, image
                MUL R5, fm_size, channel
                ADD R6, R4, R5
                MOV stack_bottom_data_ptr, R6, LSL #2
                ADD stack_bottom_data_ptr, stack_bottom_data_ptr, bottom_data
                
                ;//output: 
                ;//stack_top_data_ptr = top_data + pooled_fm_size*(layer->channels_*batch_start + channel_start)*4;
                MUL R4, pooled_batch_size, image
                MUL R5, pooled_fm_size, channel
                ADD R6, R4, R5
                MOV R6, R6, LSL #2                    ;//R6=(pooled_batch_size*batch+pooled_fm_size*fm)*4
                MOV stack_top_data_ptr, R6
                ADD stack_top_data_ptr, stack_top_data_ptr, top_data

                ;//float mask, or, 32bit integer mask
                CMP use_top_mask, #0
                ADDEQ stack_top_mask_ptr, mask, R6
                ADDNE stack_top_mask_ptr, top_mask, R6
                
                
                
                MOV ph, 0
                
                ;//Iterate through output height.
                ;//循环条件：ph < pooled_height_const
                loopph:        
                    
                    ;//hstart                        ;hstart = ph * stride_h_ - pad_h_;
                    MUL R4, ph, stride_h_        
                    SUB hstart, R4, pad_h_

                    ;//hend                        ;hend = min(hstart + kernel_h_, height_)
                    ADD R6, hstart, kernel_h_
                    CMP R6, height_const
                    MOVLT hend, R6                
                    MOVGE hend, height_const
                    
                    ;//hstart
                    CMP hstart, #0                ;//hstart = max(hstart, 0);
                    MOVLT hstart, #0
                    
                    
                    ;//wstart0 = -pad_w_
                    SUB wstart0, #0, pad_w_
                    
                    
                    ;//pool_index = ph * pooled_width_const * 4;
                    MUL pool_index, ph, pooled_width_const
                    MOV pool_index, pool_index, LSL #2
                    
                    ;//index0 = hstart * width_;
                    MUL index0, hstart, width_const
                    
                    ;//Iterate through output width.
                    
                    MOV pw, 0    
                
                    // Iterate through output width.
                    //循环条件：pw < pooled_width_const
                    looppw:
                        
                        ;//wend = min(wstart0+kernel_w_, width_);
                        ADD R4, wstart0, kernel_w_
                        CMP R4, width_const
                        MOVLT wend, R4
                        MOVGE wend, width_const
                        
                        ;//wstart = max(wstart0, 0); 
                        CMP wstart0, #0
                        MOVLT wstart, #0
                        MOVGE wstart, wstart0
                        
                        ;//wstart0 += stride_w_
                        ADD wstart0, wstart0, stride_w_
                        
                        ;//num_elements_to_do = wend - wstart; 
                        SUB num_elements_to_do, wend, wstart
                        
                        ;//const int index = index0 + wstart; 
                        ADD index, index0, wstart
                        
                        ;//const int effective_ptr = (index)*4 + input_ptr;
                        ADD effective_ptr, stack_bottom_data_ptr, index, LSL #2
                        
                        ;//Prepare accumulators.使用移动对齐的压缩单精度浮点值
                        ;//scalar模式
                        ;//如果是PoolingParameter_PoolMethod_MAX，初始化
                        ;//这里先不使用SIMD，只是普通的计算
                        SUB max_tmp, 0, FLT_MAX
                        MOV index_tmp, -1
                        
                        
                        
                            ;// Iterate through kernel height.
                            ;//循环条件：h < hend
                            MOV h, hstart
                        looph:
                            ;//align??
                                                
                            ADD R4, index, num_elements_to_do        ;//R4=index+num_elements_to_do
                            MOV R5, effective_ptr                    ;//R5=effective_ptr
                            MOV R6, index                            ;//R6=index
                                
                                
                                ;// Iterate through kernel width.
                                ;//循环条件：R6 < R4
                            loopw:
                                ;//align??
                                ;//计算，使用压缩。。。
                                
                                ;//这里先不使用SIMD，只是普通的计算
                                LDR R7, [R5]
                                CMP R7, max_tmp
                                MOVGT max_tmp, R7
                                MOVGT index_tmp, R6
                                
                                
                                ADD R5, R5, #4        ;//R5+=sizeof(float)
                                ADD R6, R6, #1        ;//index++
                                
                                CMP R6, R4
                                JLT loopw
                                ;//kernel_w loop end
                                
                            ;//向上一行
                            ADD effective_ptr, effective_ptr, width_const, LSL #2    
                            ADD index, index, width_const
                            ADD h, h, #1
                            
                            CMP h, hend
                            JLT looph
                            ;//kernel_h loop end
                        
                        
                        ;// Save accumulators.
                        ;//................
                        
                        
                        ;//保存结果到下一层
                        ;//stack_top_data_ptr + pool_index  <<2??
                        ADD R7, stack_top_data_ptr, pool_index, LSL #2
                        STR max_tmp, [R7]
                        
                        ;//保存结果到mask
                        ;//stack_top_mask_ptr + pool_index  <<2??
                        ADD R7, stack_top_mask_ptr, pool_index, LSL #2
                        STR index_tmp, [R7]
                        
                        
                        ;//Update pool_index, pool_index += sizeof(float)
                        ADD pool_index, pool_index, #4
                        
                        ADD pw, pw, #1
                        CMP pw, pooled_width_const
                        JLT looppw
                        //output width loop end
                    
                    ADD ph, ph, #1
                    CMP ph, pooled_height_const
                    JLT loopph
                    ;//output height loop end
                
                
            }*/
			float x = 12, var, y=3;
			__asm__(
				"MOV var, x\n\t"         //把x的值给R0  
				"ADD y, var, x/y\n\t"    //计算x/y时R0的值会被修改
				:[y]"+r"(y), [var]"+r"(var), [x]"+r"(x)
				:
				:
				)
				LOG(INFO)<<"y="<<y

        }
    }
    break;
									   }
  case PoolingParameter_PoolMethod_AVE:{
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = 0;
    }
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                top_data[ph * pooled_width_ + pw] +=
                    bottom_data[h * width_ + w];
              }
            }
            top_data[ph * pooled_width_ + pw] /= pool_size;
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
      }
    }
    break;
                                        }
  case PoolingParameter_PoolMethod_STOCHASTIC:{
    NOT_IMPLEMENTED;
    break;
											 }
  default:{
    LOG(FATAL) << "Unknown pooling method.";
		  }
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;  // suppress warnings about uninitialized variables
  const Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // The main loop
    if (use_top_mask) {
      top_mask = top[1]->cpu_data();
    } else {
      mask = max_idx_.cpu_data();
    }
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            const int index = ph * pooled_width_ + pw;
            const int bottom_index =
                use_top_mask ? top_mask[index] : mask[index];
            bottom_diff[bottom_index] += top_diff[index];
          }
        }
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    // The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                bottom_diff[h * width_ + w] +=
                  top_diff[ph * pooled_width_ + pw] / pool_size;
              }
            }
          }
        }
        // offset
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}


#ifdef CPU_ONLY
STUB_GPU(PoolingLayer);
#endif

INSTANTIATE_CLASS(PoolingLayer);

}  // namespace caffe
