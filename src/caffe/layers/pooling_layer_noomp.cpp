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
  /*
  LOG(INFO)<<"kernel_h_="<<kernel_h_;
  LOG(INFO)<<"kernel_w_="<<kernel_w_;
  LOG(INFO)<<"stride_h_="<<stride_h_;
  LOG(INFO)<<"stride_w_="<<stride_w_;
  LOG(INFO)<<"pad_h_="<<pad_h_;
  LOG(INFO)<<"pad_w_="<<pad_w_;
  LOG(INFO)<<"channels_="<<channels_;
  LOG(INFO)<<"height_="<<height_;
  LOG(INFO)<<"width_="<<width_;
  LOG(INFO)<<"pooled_height_="<<pooled_height_;
  LOG(INFO)<<"pooled_width_="<<pooled_width_;
  LOG(INFO)<<"global_pooling_="<<global_pooling_;
  LOG(INFO)<<"bottom[0]->num()="<<bottom[0]->num();
  LOG(INFO)<<"bottom[0]->channels()="<<bottom[0]->channels();
  LOG(INFO)<<"pooled_fm_size="<<this->pooled_height_ * this->pooled_width_;
  LOG(INFO)<<"fm_size="<<this->height_ * this->width_;
  LOG(INFO)<<"pooled_batch_size="<<this->pooled_height_ * this->pooled_width_ * this->channels_;
  LOG(INFO)<<"batch_size="<<this->height_ * this->width_ * this->channels_;
  LOG(INFO)<<"height_kernel_h_="<<this->height_ - this->kernel_h_;
  LOG(INFO)<<"width_kernel_w_="<<this->width_ - this->kernel_w_;
  LOG(INFO)<<"height_kernel_h_pad_h_="<<this->height_ - this->kernel_h_ + this->pad_h_;
  LOG(INFO)<<"width_kernel_w_pad_w_="<<this->width_ - this->kernel_w_ + this->pad_w_;
  LOG(INFO)<<"bottom_data="<<bottom[0]->cpu_data();
  LOG(INFO)<<"top_data="<<top[0]->mutable_cpu_data();
  
#if defined (__LP64__) || defined (__64BIT__) || defined (_LP64) || (__WORDSIZE == 64)
LOG(INFO)<<"I am LP64\n";
#else
LOG(INFO)<<"I am ILP32 \n";
#endif
*/
}
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
  case PoolingParameter_PoolMethod_MAX:
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
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_);
            int wend = min(wstart + kernel_w_, width_);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            const int pool_index = ph * pooled_width_ + pw;
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int index = h * width_ + w;
                if (bottom_data[index] > top_data[pool_index]) {
                  top_data[pool_index] = bottom_data[index];
                  if (use_top_mask) {
                    top_mask[pool_index] = static_cast<Dtype>(index);
                  } else {
                    mask[pool_index] = index;
                  }
                }
              }
            }
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
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
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}


template <>
void PoolingLayer<float>::Forward_cpu(const vector<Blob<float>*>& bottom,
      const vector<Blob<float>*>& top) {
  //const float* bottom_data = bottom[0]->cpu_data();
  float* bottom_data = (float*)bottom[0]->cpu_data();
  float* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;  // suppress warnings about uninitalized variables
  float* top_mask = NULL;
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:{
    // Initialize
    if (use_top_mask) {
      top_mask = top[1]->mutable_cpu_data();
      caffe_set(top_count, float(-1), top_mask);
    } else {
      mask = max_idx_.mutable_cpu_data();
      caffe_set(top_count, -1, mask);
    }
    caffe_set(top_count, float(-FLT_MAX), top_data);
    // The main loop
	const int num_channels = bottom[0]->channels();
	int ph, pw, hstart, wstart0, hend, wend, pool_index, index0, index, h, w, index_tmp;
	float max_tmp;
	
	int optimal_version = 0;

    if (this->pad_h_ == 0 &&
        this->pad_w_ == 0 &&
        (this->pooled_height_-1) * this->stride_h_ + this->kernel_h_ == this->height_ &&
        (this->pooled_width_-1) * this->stride_w_ + this->kernel_w_ == this->width_)
      optimal_version = 1;
	
	
	//存放参数，方便传递给inline assembly
	int param[30];
	param[0] = pooled_width_;//0
	param[1] = pooled_height_;//4
	param[2] = width_;//8
	param[3] = height_;//12
	
	param[4] = param[1] * param[0];//pooled_fm_size 16  = this->pooled_height_ * this->pooled_width_;
    param[5] = param[3] * param[2];//fm_size 20 = this->height_ * this->width_;

    param[6] = param[4] * channels_;//pooled_batch_size 24 = pooled_fm_size * this->channels_;
    param[7] = param[5] * channels_;//batch_size 28 = fm_size * this->channels_;

    param[8] = height_ - kernel_h_;//height_kernel_h_ 32 = this->height_ - this->kernel_h_;
    param[9] = width_ - kernel_w_;//width_kernel_w_ 36 = this->width_ - this->kernel_w_;

    param[10] = param[8] + pad_h_;//height_kernel_h_pad_h_ 40 = height_kernel_h_ + this->pad_h_;
    param[11] = param[9] + pad_w_;//width_kernel_w_pad_w_ 44 width_kernel_w_ + this->pad_w_;
	
	param[12] = stride_h_; //48
	param[13] = stride_w_; //52
	param[14] = pad_h_; //56
	param[15] = pad_w_; //60
	param[16] = kernel_h_; //64
	param[17] = kernel_w_; //68
	param[18] = optimal_version; // 72
	param[19] = (int)use_top_mask; //76
	
	
	// Prepare default accumulator. (and bypass memory aliasing warning...)
    //float min_float = -FLT_MAX;
    //void* min_float_ptr = &min_float;
    //uint32_t min_float_cast = *reinterpret_cast<uint32_t*>(min_float_ptr);
    //mov(stack_min_float, min_float_cast);
    //vbroadcastss(xmm15, stack_min_float);
    
	float *stack_bottom_data_ptr, *stack_top_data_ptr, *stack_top_mask_ptr;
	//int debug_int1, debug_int2;
	//long long debug_long1;
	//float debug_float1;
	float NEG_MIN = -FLT_MAX;
	
	/*
	//debug
	LOG(INFO)<<"NEG_MIN="<<NEG_MIN;
	float tmp = 0.1;
	printf("bottom_data:\n");
	for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < num_channels; ++c) {
        for (int oh = 0; oh < height_; ++oh) {
          for (int ow = 0; ow < width_; ++ow) {
			index = n*param[7] + c*param[5] + oh * width_ + ow;
			tmp += 0.1;
			bottom_data[index] = tmp;
			if(n==0 && c ==0) printf("%f ",tmp);
		  }
		  if(n==0 && c ==0) printf("\n");
		}
      }
    }
	printf("bottom_data_after init:\n");
	for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < num_channels; ++c) {
        for (int oh = 0; oh < height_; ++oh) {
          for (int ow = 0; ow < width_; ++ow) {
			index = n*param[7] + c*param[5] + oh * width_ + ow;
			if(n==0 && c ==0) printf("%f ",bottom_data[index]);
		  }
		  if(n==0 && c ==0) printf("\n");
		}
      }
    }
	float tmp_max[4000];
	*/
    //之后可以把这两个循环拆成汇编，或者使用OpenMP优化
    for (int image = 0; image < bottom[0]->num(); ++image)
    {
        for (int channel = 0; channel < num_channels; ++channel)
        {            
            __asm__ __volatile__(
            
                //input: 
                //stack_bottom_data_ptr = bottom_data + fm_size*(layer->channels_*batch_start + channel_start)*4;//*4?*8?
				"LDR w3, [%[param], #28]  \n\t" //batch_size=param[7]-x3
				"LDR w4, [%[param], #20] \n\t"//fm_size=param[5]-x4
                "MUL x5, x3, %[image]\n\t"
                "MUL x6, x4, %[channel]\n\t"
                "ADD x7, x5, x6\n\t"
                "ADD %[stack_bottom_data_ptr], %[bottom_data], x7, LSL #2\n\t"
                
                //output: 
                //stack_top_data_ptr = top_data + pooled_fm_size*(layer->channels_*batch_start + channel_start)*4;
				"LDR w3, [%[param], #24]  \n\t" //pooled_batch_size=param[6]-x3
				"LDR w4, [%[param], #16] \n\t"//pooled_fm_size=param[4]-x4
                "MUL x5, x3, %[image]\n\t"
                "MUL x6, x4, %[channel]\n\t"
                "ADD x7, x5, x6\n\t"
				//"MOV x5, #0 \n\t"
				//"ADD x5, XZR, x7, LSL #2 \n\t"        //x5=(pooled_batch_size*batch+pooled_fm_size*fm)*4
				"LSL x5, x7, #2 \n\t"
				"ADD %[stack_top_data_ptr], %[top_data], x5\n\t"

				
                //float mask, or, 32bit integer mask
				"LDR w3, [%[param], #76]  \n\t" //use_top_mask=param[19]-x3
                "CMP x3, #0\n\t"
				"ADD x6, %[mask], x5 \n\t"
				"ADD x7, %[top_mask], x5 \n\t"
				"CSEL %[stack_top_mask_ptr], x6, x7, EQ \n\t"
                
                
				//ph: x3
                "MOV x3, #0\n\t"
                
                //Iterate through output height.
                //循环条件：ph < pooled_height_const
                "loopph:\n\t"        
                    
					//optimal_version
				//	"LDR w8, [%[param], #72]  \n\t" //optimal_version=param[18]-x3
				//	"CMP x8, #0\n\t"
				//	"BEQ Not_Optimal\n\t"
					
				//"Optimal:\n\t"
					


					
					
					
					//B Endpw\n\t"
					
				"Not_Optimal:\n\t"
                    //hstart                        ;hstart = ph * stride_h_ - pad_h_;
					//此处可以更换param顺序一次读入64bit优化
					"LDR w8, [%[param], #48] \n\t"     //stride_h_=param[12]
                    "MUL x5, x3, x8\n\t"    
					"LDR w8, [%[param], #56] \n\t"     //pad_h_=param[14]
                    "SUB %[hstart], x5, x8\n\t"

                    //hend                        ;hend = min(hstart + kernel_h_, height_)
					"LDR w8, [%[param], #64] \n\t"    //kernel_h_=param[16]
                    "ADD x6, %[hstart], x8 \n\t"        
					"LDR w8, [%[param], #12] \n\t"       //height_const=param[3]
                    "CMP x6, x8\n\t"                       
					"CSEL %[hend], x6, x8, LT \n\t"

                    
                    //hstart
                    "CMP %[hstart], #0\n\t"                //hstart = max(hstart, 0);
                    "CSEL %[hstart], XZR, %[hstart], LT \n\t"
                    
                    //wstart0 = -pad_w_
					"LDR w8, [%[param], #60] \n\t" //pad_w_=param[15]
                    "SUB %[wstart0], XZR, x8\n\t"
                    
                    //pool_index = ph * pooled_width_const * 4;
					"LDR w8, [%[param], #0] \n\t" //pooled_width_const=param[0]
                    "MUL %[pool_index], x3, x8\n\t"
					"LSL %[pool_index], %[pool_index], #2 \n\t"
                    
                    //index0 = hstart * width_;
					"LDR w8, [%[param], #8] \n\t" //width_=param[2]
                    "MUL %[index0], %[hstart], x8\n\t"
                    
                    //Iterate through output width.
					//pw: x4
                    "MOV x4, #0\n\t"    
                
                    // Iterate through output width.
                    //循环条件：pw < pooled_width_const
                    "looppw:\n\t"
                        
                        //wend = min(wstart0+kernel_w_, width_); x6
						"LDR w8, [%[param], #68] \n\t" //kernel_w_=param[17]
                        "ADD x5, %[wstart0], x8\n\t"
						"LDR w8, [%[param], #8] \n\t" //width_=param[2]
                        "CMP x5, x8\n\t"
						"CSEL x6, x5, x8, LT \n\t"

                        
                        //wstart = max(wstart0, 0); x5
                        "CMP %[wstart0], #0\n\t"
						"CSEL x5, XZR, %[wstart0], LT \n\t"
                        
                        //wstart0 += stride_w_
						"LDR w8, [%[param], #52] \n\t" //stride_w_=param[13]
                        "ADD %[wstart0], %[wstart0], x8\n\t"
                        
                        //num_elements_to_do = wend - wstart; x9
                        "SUB x9, x6, x5\n\t"
                        
                        //const int index = index0 + wstart; 
                        "ADD %[index], %[index0], x5\n\t"
                        
                        //const long long effective_ptr = (index)*4 + input_ptr; x10 //*4?*8 
                        "ADD x10, %[stack_bottom_data_ptr], %[index], LSL #2\n\t"
                        
                        //Prepare accumulators.使用移动对齐的压缩单精度浮点值
                        //scalar模式
                        //如果是PoolingParameter_PoolMethod_MAX，初始化
                        //这里先不使用SIMD，只是普通的计算
						"FMOV %s[max_tmp], %s[NEG_MIN]\n\t"
                        "MOV %[index_tmp], #-1\n\t"
						
                            // Iterate through kernel height.
                            //循环条件：h < hend
                            "MOV %[h], %[hstart]\n\t"
                        "looph:\n\t"
                            //align??
                                  
							//这里可以精简一下寄存器的使用
                            "ADD x5, %[index], x9\n\t"        //x5=index+num_elements_to_do
                            "MOV x6, x10\n\t"                    //x6=effective_ptr
                            "MOV x7, %[index]\n\t"                            //x7=index
                            
                                // Iterate through kernel width.
                                //循环条件：x7 < x5（index<num_elements_to_do）
                            "loopw:\n\t"
                                //align??
                                //计算，使用压缩。。。
                                
                                //这里先不使用SIMD，只是普通的计算
                                "LDR s0, [x6]\n\t"
                                "FCMP s0, %s[max_tmp]\n\t"
								"FCSEL %s[max_tmp], s0, %s[max_tmp], GT \n\t"
								"CSEL %[index_tmp], x7, %[index_tmp], GT \n\t"
																
                                "ADD x6, x6, #4\n\t"        //x6+=sizeof(float)
								"ADD x7, x7, #1 \n\t"
								"CMP x7, x5 \n\t"
								
                                "BLT loopw\n\t"
                                //kernel_w loop end
                                
                            //向上一行
							//effective_ptr += width_const * 4
							"LDR w8, [%[param], #8] \n\t" //width_const=param[2]
                            "ADD x10, x10, x8, LSL #2\n\t"
                            "ADD %[index], %[index], x8\n\t"
                            "ADD %[h], %[h], #1\n\t"
                            
                            "CMP %[h], %[hend]\n\t"
                            "BLT looph\n\t"
                            //kernel_h loop end
                        
                        
                        // Save accumulators.
                        //................
                        
                        
                        //保存结果到下一层
                        //stack_top_data_ptr + pool_index(已经乘4了)  
                        "ADD x8, %[stack_top_data_ptr], %[pool_index]\n\t"
                        "STR %s[max_tmp], [x8]\n\t"
						
						//"ADD x8, %[tmp_max], %[pool_index] \n\t"
						//"STR s0, [x8] \n\t"
                        
                        //保存结果到mask
                        //stack_top_mask_ptr + pool_index(已经乘4了) 
                        "ADD x8, %[stack_top_mask_ptr], %[pool_index]\n\t"
                        "STR %w[index_tmp], [x8]\n\t"
                        
                        
                        //Update pool_index, pool_index += sizeof(float)?
                        "ADD %[pool_index], %[pool_index], #4\n\t"
                        
                        "ADD x4, x4, #1\n\t"
						"LDR w8, [%[param], #0] \n\t" //pooled_width_const=param[0]
                        "CMP x4, x8\n\t"
                        "BLT looppw\n\t"
                        //output width loop end
						
                    "Endpw:\n\t"
					
                    "ADD x3, x3, #1\n\t"
					"LDR w8, [%[param], #4] \n\t" //pooled_height_const=param[4]
                    "CMP x3, x8\n\t"
                    "BLT loopph\n\t"
                    //output height loop end
				
                
				://output
				[stack_bottom_data_ptr]"=&r"(stack_bottom_data_ptr), [stack_top_data_ptr]"=&r"(stack_top_data_ptr), [stack_top_mask_ptr]"=&r"(stack_top_mask_ptr)
				//, [debug_int1]"=&r"(debug_int1), [debug_int2]"=&r"(debug_int2)
				//, [debug_long1]"=&r"(debug_long1) 
				//,[debug_float1]"=&w"(debug_float1)
				//[result]"=r"(result)
                ://input
				[param]"r"(param),
				[image]"r"(image), [channel]"r"(channel), 
				[mask]"r"(mask), [top_mask]"r"(top_mask), 
				[bottom_data]"r"(bottom_data), [top_data]"r"(top_data), 
				[max_tmp]"w"(max_tmp),
				[hstart]"r"(hstart), [wstart0]"r"(wstart0), [hend]"r"(hend),  
				[pool_index]"r"(pool_index), [index0]"r"(index0), [index]"r"(index), [index_tmp]"r"(index_tmp),
				[NEG_MIN]"w"(NEG_MIN),
				//[tmp_max]"r"(tmp_max),
				//[input]"m"(input),
				//[pooled_width_const]"r"(pooled_width_const), [pooled_height_const]"r"(pooled_height_const), 
				//[width_const]"r"(width_const), [height_const]"r"(height_const), 
				//[pooled_fm_size]"r"(pooled_fm_size), [fm_size]"r"(fm_size), 
				//[pooled_batch_size]"r"(pooled_batch_size), [batch_size]"r"(batch_size), 
				//[pad_h_]"r"(pad_h_), [pad_w_]"r"(pad_w_), 
				//[kernel_h_]"r"(kernel_h_), [kernel_w_]"r"(kernel_w_), 
				//[stride_h_]"r"(stride_h_), [stride_w_]"r"(stride_w_), 
				//[optimal_version]"r"(optimal_version), [use_top_mask]"r"(use_top_mask),
				//[height_kernel_h_pad_h_]"r"(height_kernel_h_pad_h_), [width_kernel_w_pad_w_]"r"(width_kernel_w_pad_w_), 
				[h]"r"(h), [w]"r"(w)
				
				:"x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "s0", "memory", "cc",
				"s1", "w11", "w12"
            );//不要忘记分号
		//LOG(INFO)<<"debug_int1="<<debug_int1;
		//LOG(INFO)<<"debug_int2="<<debug_int2;
		//LOG(INFO)<<"debug_float1="<<debug_float1;
		//LOG(INFO)<<"debug_long1="<<debug_long1;
		//LOG(INFO)<<"bottom_data="<<bottom_data;
		//LOG(INFO)<<"stack_bottom_data_ptr="<<stack_bottom_data_ptr;
		//LOG(INFO)<<"top_data="<<top_data;
		//LOG(INFO)<<"stack_top_data_ptr="<<stack_top_data_ptr;
		//LOG(INFO)<<"use_top_mask="<<use_top_mask;
		//if(use_top_mask) LOG(INFO)<<"top_mask="<<top_mask;
		//else LOG(INFO)<<"mask="<<mask;
		//LOG(INFO)<<"stack_top_mask_ptr="<<stack_top_mask_ptr;
		/*if(image==0 && channel==0) {
			printf("tmp_max:\n");
			for (int plh = 0; plh < pooled_height_; ++plh) {
			  for (int plw = 0; plw < pooled_width_; ++plw) {
				pool_index = plh * pooled_width_ + plw;
				printf("%f ",tmp_max[pool_index]);
			  }
			  printf("\n");
			}

			printf("topdata:\n");
			for (int plh = 0; plh < pooled_height_; ++plh) {
			  for (int plw = 0; plw < pooled_width_; ++plw) {
				pool_index = plh * pooled_width_ + plw;
				printf("%f ",top_data[pool_index]);
			  }
			  printf("\n");
			}

			printf("mask:\n");
			for (int plh = 0; plh < pooled_height_; ++plh) {
			  for (int plw = 0; plw < pooled_width_; ++plw) {
				pool_index = plh * pooled_width_ + plw;
				printf("%d ",mask[pool_index]);
			  }
			  printf("\n");
			}
		}
		*/
		}
	}
    break;
  }
  case PoolingParameter_PoolMethod_AVE: {
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
  case PoolingParameter_PoolMethod_STOCHASTIC: {
    NOT_IMPLEMENTED;
    break;
  }
  default: {
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
