#ifndef UVIAD_HPP__
#define UVIAD_HPP__

#include "trt/ad/ad.hpp"

namespace AD
{
    class UviadModelImpl :public ADModelImpl
    {
        public:
            tensor::Memory<float> segment_predict_;

            // mask框的仿射矩阵


    };
}



#endif