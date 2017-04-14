// ------------------------------------------------------------------
// R-FCN
// Copyright (c) 2016 Microsoft
// Licensed under The MIT License [see r-fcn/LICENSE for details]
// Written by Yi Li
// ------------------------------------------------------------------

#include <cfloat>

# include "caffe/layers/psroi_pooling_layer.hpp"

using std::max;
using std::min;

namespace caffe
{
	template <typename Dtype>
		void PSROIPoolingLayer<Dtype>::LayerSetUp
		(
			const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top
		)
		{
    		PSROIPoolingParameter psroi_pooling_param =
      		this->layer_param_.psroi_pooling_param();
    		spatial_scale_ = psroi_pooling_param.spatial_scale();
  			LOG(INFO) << "Spatial scale: " << spatial_scale_;

	    	CHECK_GT(psroi_pooling_param.output_dim(), 0)
   			<< "output_dim must be > 0";
    		CHECK_GT(psroi_pooling_param.group_size(), 0)
    	  	<< "group_size must be > 0";

    		output_dim_ = psroi_pooling_param.output_dim();
    		group_size_ = psroi_pooling_param.group_size();
    		pooled_height_ = group_size_;
    		pooled_width_ = group_size_;
  		}

	template <typename Dtype>
		void PSROIPoolingLayer<Dtype>::Reshape
		(
			const vector<Blob<Dtype>*>& bottom,
    		const vector<Blob<Dtype>*>& top
		)
	{
    	channels_ = bottom[0]->channels();
    	CHECK_EQ(channels_, output_dim_*group_size_*group_size_) << "input channel number does not match layer parameters";
    	height_ = bottom[0]->height();
    	width_ = bottom[0]->width();
    	top[0]->Reshape(bottom[1]->num(), output_dim_, pooled_height_, pooled_width_);
    	mapping_channel_.Reshape(bottom[1]->num(), output_dim_, pooled_height_, pooled_width_);
	}

    template <typename Dtype>
        void PSROIPoolingForward
        (
            const int nthreads,
            const Dtype* bottom_data,
            const Dtype spatial_scale,
            const int channels,
            const int height,
            const int width,
            const int pooled_height,
            const int pooled_width,
            const Dtype* bottom_rois,
            const int output_dim,
            const int group_size,
            Dtype* top_data,
            int* mapping_channel
		)
    {
        for (int threadIndex = 0; threadIndex != nthreads; ++threadIndex)
        {
            //The output is in order (n, ctop, ph, pw)
            int pw = threadIndex % pooled_width;
            int ph = (threadIndex / pooled_width) % pooled_height;
            int ctop = (threadIndex / pooled_width / pooled_height) % output_dim;
            int n = threadIndex / pooled_width / pooled_height / output_dim;

            bottom_rois += n * 5;
            int roi_batch_ind = bottom_rois[0];
            Dtype roi_start_w = static_cast<Dtype>(round(bottom_rois[1])) * spatial_scale;
            Dtype roi_start_h = static_cast<Dtype>(round(bottom_rois[2])) * spatial_scale;
            Dtype roi_end_w = static_cast<Dtype>(round(bottom_rois[3]) + 1.) * spatial_scale;
            Dtype roi_end_h = static_cast<Dtype>(round(bottom_rois[4] + 1.) * spatial_scale);

            Dtype roi_width = max(roi_end_w - roi_start_w, static_cast<Dtype>(0.1));
            Dtype roi_height = max(roi_end_h - roi_start_h, static_cast<Dtype>(0.1f));

            Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
            Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

            int hstart = floor(static_cast<Dtype>(ph) * bin_size_h + roi_start_h);
            int wstart = floor(static_cast<Dtype>(pw) * bin_size_w + roi_start_w);
            int hend = ceil(static_cast<Dtype>(ph + 1) * bin_size_h + roi_start_h);
            int wend = ceil(static_cast<Dtype>(pw + 1) * bin_size_w + roi_start_w);

            hstart = min(max(hstart, 0), height);
            hend = min(max(hend, 0), height);
            wstart = min(max(wstart, 0), width);
            wend = min(max(wend, 0), width);
            bool is_empty = (hend <= hstart ) || (wend <= wstart);

            int gw = pw;
            int gh = ph;
            int c = (ctop*group_size + gh)*group_size + gw;

            const Dtype* shifted_bottom_data = bottom_data + (roi_batch_ind*channels + c) * height * width;
            //bottom_data += (roi_batch_ind *channels + c) * height * width;
            //modified: bottom_data is no longer incremented per thread for its now sequentially executed
            Dtype out_sum = 0;
            for (int h = hstart; h < hend; ++h)
            {
                for(int w = wstart; w < wend; ++w)
                {
                    int bottom_index = h*width + w;
                    out_sum += shifted_bottom_data[bottom_index];
                }
            }

            Dtype bin_area = (hend - hstart) * (wend - wstart);
            top_data[threadIndex] = is_empty? 0. : out_sum/bin_area;
            mapping_channel[threadIndex] = c;
        }
    }

    template <typename Dtype>
    void PSROIPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top)
    {
        const Dtype* bottom_data = bottom[0]->cpu_data();
        const Dtype* bottom_rois = bottom[1]->cpu_data();
        Dtype* top_data = top[0]->mutable_cpu_data();
        int* mapping_channel_ptr = mapping_channel_.mutable_cpu_data();
        int count = top[0]->count();

    	// NOLINT_NEXT_LINE(whitespace/operators)
        PSROIPoolingForward<Dtype>
		(
            count, bottom_data, spatial_scale_, channels_, height_, width_, pooled_height_,
            pooled_width_, bottom_rois, output_dim_, group_size_, top_data, mapping_channel_ptr
		);
    }

    template <typename Dtype>
        void PSROIPoolingBackwardAtomic
        (
            const int nthreads,
            const Dtype* top_diff,
            const int* mapping_channel,
            const int num_rois,
            const Dtype spatial_scale,
            const int channels,
            const int height, const int width,
            const int pooled_height,
            const int pooled_width,
            const int output_dim,
            Dtype* bottom_diff,
            const Dtype* bottom_rois
        )
    {
        for (int threadIndex = 0; threadIndex != nthreads; ++threadIndex)
        {
        	// The output is in order (n, ctop, ph, pw)
        	int pw = threadIndex % pooled_width;
        	int ph = (threadIndex / pooled_width) % pooled_height;
        	int n = threadIndex / pooled_width / pooled_height / output_dim;

        	// [start, end) interval for spatial sampling
        	bottom_rois += n * 5;
        	int roi_batch_ind = bottom_rois[0];
        	Dtype roi_start_w = static_cast<Dtype>(round(bottom_rois[1])) * spatial_scale;
        	Dtype roi_start_h = static_cast<Dtype>(round(bottom_rois[2])) * spatial_scale;
        	Dtype roi_end_w = static_cast<Dtype>(round(bottom_rois[3]) + 1.) * spatial_scale;
        	Dtype roi_end_h = static_cast<Dtype>(round(bottom_rois[4]) + 1.) * spatial_scale;

        	// Force too small ROIs to be 1x1
        	Dtype roi_width = max(roi_end_w - roi_start_w, static_cast<Dtype>(0.1)); //avoid 0
        	Dtype roi_height = max(roi_end_h - roi_start_h, static_cast<Dtype>(0.1));

      		// Compute w and h at bottom
        	Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
        	Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

        	int hstart = floor(static_cast<Dtype>(ph)* bin_size_h + roi_start_h);
        	int wstart = floor(static_cast<Dtype>(pw)* bin_size_w + roi_start_w);
        	int hend = ceil(static_cast<Dtype>(ph + 1) * bin_size_h + roi_start_h);
	        int wend = ceil(static_cast<Dtype>(pw + 1) * bin_size_w + roi_start_w);
		    // Add roi offsets and clip to input boundaries
	        hstart = min(max(hstart, 0), height);
    	    hend = min(max(hend, 0), height);
    	    wstart = min(max(wstart, 0), width);
    	    wend = min(max(wend, 0), width);
    	    bool is_empty = (hend <= hstart) || (wend <= wstart);

      		// Compute c at bottom

	        int c = mapping_channel[threadIndex];
    	    Dtype* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;
    	    Dtype bin_area = (hend - hstart)*(wend - wstart);
    	    Dtype diff_val = is_empty ? 0. : top_diff[threadIndex] / bin_area;
    	    for (int h = hstart; h < hend; ++h)
    	    {
    	        for (int w = wstart; w < wend; ++w)
    	        {
    	            int bottom_index = h*width + w;
                    *(offset_bottom_diff + bottom_index) += diff_val;
    	        }
    	    }
        }
    }

  	template <typename Dtype>
  	void PSROIPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
	    if (!propagate_down[0])
		{
    		return;
		}

	    const Dtype* bottom_rois = bottom[1]->cpu_data();
	    const Dtype* top_diff = top[0]->cpu_diff();
	    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	    const int bottom_count = bottom[0]->count();
	    const int* mapping_channel_ptr = mapping_channel_.cpu_data();
        for (int i = 0; i != bottom[1]->count(); ++i)
        {
            bottom[1]->mutable_cpu_diff()[i] = static_cast<Dtype>(0);
        }
        for (int i = 0; i != bottom_count; ++i)
        {
            bottom_diff[i] = static_cast<Dtype>(0);
        }
	    const int count = top[0]->count();
	    PSROIPoolingBackwardAtomic<Dtype>
		(
            count, top_diff, mapping_channel_ptr, top[0]->num(), spatial_scale_,
            channels_, height_, width_, pooled_height_, pooled_width_, output_dim_,
            bottom_diff, bottom_rois
		);
	}

    INSTANTIATE_CLASS(PSROIPoolingLayer);
    REGISTER_LAYER_CLASS(PSROIPooling);
}  // namespace caffe
