#ifndef UVIAD_HPP__
#define UVIAD_HPP__

#include "common/object.hpp"
#include "trt/ad/ad.hpp"

namespace AD
{
    class UviadModelImpl :public ADModelImpl
    {
        public:
            tensor::Memory<float> segment_predict_;
            // mask框的仿射矩阵
            tensor::Memory<float> mask_affine_matrix_;
            std::vector<std::shared_ptr<tensor::Memory<float>>> inverse_affine_matrixs_;
            // 整图的segment缓存
            tensor::Memory<unsigned char> segment_cache_;
            // 整图的segment缓存
            tensor::Memory<unsigned char> original_segment_cache_;

            // 框的segment缓存
            tensor::Memory<float> box_segment_cache_;
            // 框的segment缓存
            tensor::Memory<unsigned char> original_box_segment_cache_;

            std::vector<int> segment_head_dims_;
        public:
        void adjust_memory(int batch_size);

        public:
            virtual InferResult forwards(const std::vector<cv::Mat> &inputs, void *stream = nullptr) override;
            virtual bool load(const std::string &engine_file,
                        float confidence_threshold,
                        int gpu_id,
                        int max_batch_size) override;
                
            void preprocess(int ibatch,
                const tensor::Image &image,
                std::shared_ptr<tensor::Memory<unsigned char>> preprocess_buffer,
                affine::ResizeMatrix &affine,
                void *stream = nullptr);
        private:
            void decode_segment(int ib, float *parray, const cv::Mat &input, std::vector<object::SegmentationInstance> &output, cudaStream_t stream);
    };

    std::shared_ptr<InferBase> load_uviad(const std::string &engine_file,
                                                // const std::vector<std::string> &names,
                                                int gpu_id,
                                                float confidence_threshold,
                                                int max_batch_size);
} // end namespace AD

#endif