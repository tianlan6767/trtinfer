#include "kernels/decode.cuh"
#include <stdio.h>
#include <cfloat>


namespace cuda
{

static __device__ void
affine_project_obb(float *matrix, float x, float y, float w, float h, float *ox, float *oy, float *ow, float *oh)
{
    *ox = matrix[0] * x + matrix[1] * y + matrix[2];
    *oy = matrix[3] * x + matrix[4] * y + matrix[5];
    *ow = matrix[0] * w; // 在均匀缩放时, matrix[0] = s_inv
    *oh = matrix[4] * h;
}

static __device__ void affine_project(float *matrix, float x, float y, float *ox, float *oy)
{
    *ox = matrix[0] * x + matrix[1] * y + matrix[2];
    *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}

static __device__ float
box_iou(float aleft, float atop, float aright, float abottom, float bleft, float btop, float bright, float bbottom)
{
    float cleft   = max(aleft, bleft);
    float ctop    = max(atop, btop);
    float cright  = min(aright, bright);
    float cbottom = min(abottom, bbottom);

    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if (c_area == 0.0f)
        return 0.0f;

    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

static __device__ void convariance_matrix(float w, float h, float r, float& a, float& b, float& c){
    float a_val = w * w / 12.0f;
    float b_val = h * h / 12.0f;
    float cos_r = cosf(r); 
    float sin_r = sinf(r);

    a = a_val * cos_r * cos_r + b_val * sin_r * sin_r;
    b = a_val * sin_r * sin_r + b_val * cos_r * cos_r;
    c = (a_val - b_val) * sin_r * cos_r;
}

static __device__ float box_probiou(
    float cx1, float cy1, float w1, float h1, float r1,
    float cx2, float cy2, float w2, float h2, float r2,
    float eps = 1e-7
){

    // Calculate the prob iou between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.
    float a1, b1, c1, a2, b2, c2;
    convariance_matrix(w1, h1, r1, a1, b1, c1);
    convariance_matrix(w2, h2, r2, a2, b2, c2);

    float t1 = ((a1 + a2) * powf(cy1 - cy2, 2) + (b1 + b2) * powf(cx1 - cx2, 2)) / ((a1 + a2) * (b1 + b2) - powf(c1 + c2, 2) + eps);
    float t2 = ((c1 + c2) * (cx2 - cx1) * (cy1 - cy2)) / ((a1 + a2) * (b1 + b2) - powf(c1 + c2, 2) + eps);
    float t3 = logf(((a1 + a2) * (b1 + b2) - powf(c1 + c2, 2)) / (4 * sqrtf(fmaxf(a1 * b1 - c1 * c1, 0.0f)) * sqrtf(fmaxf(a2 * b2 - c2 * c2, 0.0f)) + eps) + eps); 
    float bd = 0.25f * t1 + 0.5f * t2 + 0.5f * t3;
    bd = fmaxf(fminf(bd, 100.0f), eps);
    float hd = sqrtf(1.0f - expf(-bd) + eps);
    return 1 - hd;    
}

// yolo v5 目标检测后处理kernel
__global__ void decode_kernel_v5(float *predict,
                                 int num_bboxes,
                                 int num_classes,
                                 int output_cdim,
                                 float confidence_threshold,
                                 float *invert_affine_matrix,
                                 float *parray,
                                 int *box_count,
                                 int max_image_boxes,
                                 int num_box_element,
                                 int start_x,
                                 int start_y,
                                 int batch_index)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes)
        return;

    float *pitem     = predict + output_cdim * position;
    float objectness = pitem[4];
    if (objectness < confidence_threshold)
        return;

    float *class_confidence = pitem + 5;

    float confidence = *class_confidence++;
    int label        = 0;
    for (int i = 1; i < num_classes; ++i, ++class_confidence)
    {
        if (*class_confidence > confidence)
        {
            confidence = *class_confidence;
            label      = i;
        }
    }
    confidence *= objectness;

    if (confidence < confidence_threshold)
        return;

    int index = atomicAdd(box_count, 1);
    if (index >= max_image_boxes)
        return;

    float cx     = *pitem++;
    float cy     = *pitem++;
    float width  = *pitem++;
    float height = *pitem++;
    float left   = cx - width * 0.5f;
    float top    = cy - height * 0.5f;
    float right  = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;
    affine_project(invert_affine_matrix, left, top, &left, &top);
    affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

    float *pout_item = parray + index * num_box_element;
    *pout_item++     = left + start_x;
    *pout_item++     = top + start_y;
    *pout_item++     = right + start_x;
    *pout_item++     = bottom + start_y;
    *pout_item++     = confidence;
    *pout_item++     = label;
    *pout_item++     = 1; // 1 = keep, 0 = ignore
    *pout_item++     = position;
    *pout_item++     = batch_index; // batch_index
    // 这里的batch_index是为了在后续的mask weights中使用，方便找到当前batch的图片位置
}

// yolo v8 v11 目标检测后处理kernel
__global__ void decode_kernel_v11(float *predict,
                                  int num_bboxes,
                                  int num_classes,
                                  int output_cdim,
                                  float confidence_threshold,
                                  float *invert_affine_matrix,
                                  float *parray,
                                  int *box_count,
                                  int max_image_boxes,
                                  int num_box_element,
                                  int start_x,
                                  int start_y,
                                  int batch_index)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes)
        return;

    float *pitem            = predict + output_cdim * position;
    float *class_confidence = pitem + 4;
    float confidence        = *class_confidence++;
    int label               = 0;
    for (int i = 1; i < num_classes; ++i, ++class_confidence)
    {
        if (*class_confidence > confidence)
        {
            confidence = *class_confidence;
            label      = i;
        }
    }
    if (confidence < confidence_threshold)
        return;

    int index = atomicAdd(box_count, 1);
    if (index >= max_image_boxes)
        return;

    float cx     = *pitem++;
    float cy     = *pitem++;
    float width  = *pitem++;
    float height = *pitem++;
    float left   = cx - width * 0.5f;
    float top    = cy - height * 0.5f;
    float right  = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;
    affine_project(invert_affine_matrix, left, top, &left, &top);
    affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

    float *pout_item = parray + index * num_box_element;
    *pout_item++     = left + start_x;
    *pout_item++     = top + start_y;
    *pout_item++     = right + start_x;
    *pout_item++     = bottom + start_y;
    *pout_item++     = confidence;
    *pout_item++     = label;
    *pout_item++     = 1; // 1 = keep, 0 = ignore
    *pout_item++     = position;
    *pout_item++     = batch_index; // batch_index
    // 这里的batch_index是为了在后续的mask weights中使用，方便找到当前batch的图片位置
}

// yolo v8 v11 姿态估计后处理kernel
__global__ void decode_kernel_v11_pose(float *predict,
                                       int num_bboxes,
                                       int num_classes,
                                       int output_cdim,
                                       float confidence_threshold,
                                       float *invert_affine_matrix,
                                       float *parray,
                                       int *box_count,
                                       int max_image_boxes,
                                       int num_box_element,
                                       int num_key_point,
                                       int start_x,
                                       int start_y,
                                       int batch_index)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes)
        return;

    float *pitem            = predict + output_cdim * position;
    float *class_confidence = pitem + 4;
    float *key_points       = pitem + 4 + num_classes;
    float confidence        = *class_confidence++;

    int label = 0;
    for (int i = 1; i < num_classes; ++i, ++class_confidence)
    {
        if (*class_confidence > confidence)
        {
            confidence = *class_confidence;
            label      = i;
        }
    }
    if (confidence < confidence_threshold)
        return;

    int index = atomicAdd(box_count, 1);
    if (index >= max_image_boxes)
        return;

    float cx     = *pitem++;
    float cy     = *pitem++;
    float width  = *pitem++;
    float height = *pitem++;
    float left   = cx - width * 0.5f;
    float top    = cy - height * 0.5f;
    float right  = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;
    affine_project(invert_affine_matrix, left, top, &left, &top);
    affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

    float *pout_item = parray + index * (num_box_element + num_key_point * 3);
    *pout_item++     = left + start_x;
    *pout_item++     = top + start_y;
    *pout_item++     = right + start_x;
    *pout_item++     = bottom + start_y;
    *pout_item++     = confidence;
    *pout_item++     = label;
    *pout_item++     = 1; // 1 = keep, 0 = ignore
    *pout_item++     = position;
    *pout_item++     = batch_index; // batch_index
    for (int i = 0; i < num_key_point; i++)
    {
        float x = *key_points++;
        float y = *key_points++;
        affine_project(invert_affine_matrix, x, y, &x, &y);
        float score  = *key_points++;
        *pout_item++ = x + start_x;
        *pout_item++ = y + start_y;
        *pout_item++ = score;
    }
}

// yolo v8 v11 旋转框后处理kernel
__global__ void decode_kernel_v11_obb(float *predict,
                                      int num_bboxes,
                                      int num_classes,
                                      int output_cdim,
                                      float confidence_threshold,
                                      float *invert_affine_matrix,
                                      float *parray,
                                      int *box_count,
                                      int max_image_boxes,
                                      int num_box_element,
                                      int start_x,
                                      int start_y,
                                      int batch_index)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes)
        return;

    float *pitem            = predict + output_cdim * position;
    float *class_confidence = pitem + 4;
    float confidence        = *class_confidence++;
    int label               = 0;
    for (int i = 1; i < num_classes; ++i, ++class_confidence)
    {
        if (*class_confidence > confidence)
        {
            confidence = *class_confidence;
            label      = i;
        }
    }
    if (confidence < confidence_threshold)
        return;

    int index = atomicAdd(box_count, 1);
    if (index >= max_image_boxes)
        return;

    float cx     = *pitem++;
    float cy     = *pitem++;
    float width  = *pitem++;
    float height = *pitem++;
    float angle  = *(pitem + num_classes);

    affine_project_obb(invert_affine_matrix, cx, cy, width, height, &cx, &cy, &width, &height);

    float *pout_item = parray + index * num_box_element;
    *pout_item++     = cx + start_x;
    *pout_item++     = cy + start_y;
    *pout_item++     = width;
    *pout_item++     = height;
    *pout_item++     = angle;
    *pout_item++     = confidence;
    *pout_item++     = label;
    *pout_item++     = 1; // 1 = keep, 0 = ignore
    *pout_item++     = position;
    *pout_item++     = batch_index; // batch_index
}

__global__ void
fast_nms_kernel(float *bboxes, int *box_count, int max_image_boxes, float threshold, int num_box_element)
{
    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    int count    = min((int)*box_count, max_image_boxes);
    if (position >= count)
        return;

    // left, top, right, bottom, confidence, class, keepflag, batch_index
    float *pcurrent = bboxes + position * num_box_element;
    for (int i = 0; i < count; ++i)
    {
        float *pitem = bboxes + i * num_box_element;
        if (i == position || pcurrent[5] != pitem[5])
            continue;

        if (pitem[4] >= pcurrent[4])
        {
            if (pitem[4] == pcurrent[4] && i < position)
                continue;

            float iou =
                box_iou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3], pitem[0], pitem[1], pitem[2], pitem[3]);

            if (iou > threshold)
            {
                pcurrent[6] = 0; // 1=keep, 0=ignore
                return;
            }
        }
    }
}

__global__ void fast_nms_pose_kernel(
    float *bboxes, int *box_count, int max_image_boxes, float threshold, int num_box_element, int num_key_point)
{
    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    int count    = min((int)*box_count, max_image_boxes);
    if (position >= count)
        return;

    float *pcurrent = bboxes + position * (num_box_element + num_key_point * 3);
    for (int i = 0; i < count; ++i)
    {
        float *pitem = bboxes + i * (num_box_element + num_key_point * 3);
        if (i == position || pcurrent[6] != pitem[6])
            continue;

        if (pitem[4] >= pcurrent[4])
        {
            if (pitem[4] == pcurrent[4] && i < position)
                continue;

            float iou =
                box_iou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3], pitem[0], pitem[1], pitem[2], pitem[3]);

            if (iou > threshold)
            {
                pcurrent[6] = 0; // 1=keep, 0=ignore
                return;
            }
        }
    }
}

// yolo obb nms kernel
__global__ void
fast_nms_obb_kernel(float *bboxes, int *box_count, int max_image_boxes, float threshold, int num_box_element)
{
    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    int count    = min((int)*box_count, max_image_boxes);
    if (position >= count)
        return;

    float *pcurrent = bboxes + position * num_box_element;
    for (int i = 0; i < count; ++i)
    {
        float *pitem = bboxes + i * (num_box_element);
        if (i == position || pcurrent[6] != pitem[6])
            continue;

        if (pitem[5] >= pcurrent[5])
        {
            if (pitem[5] == pcurrent[5] && i < position)
                continue;

            float iou = box_probiou(pcurrent[0],
                                    pcurrent[1],
                                    pcurrent[2],
                                    pcurrent[3],
                                    pcurrent[4],
                                    pitem[0],
                                    pitem[1],
                                    pitem[2],
                                    pitem[3],
                                    pitem[4]);

            if (iou > threshold)
            {
                pcurrent[7] = 0; // 1=keep, 0=ignore
                return;
            }
        }
    }
}

__global__ void decode_dfine_kernel(
    int64_t* labels, 
    float* scores, 
    float* boxes, 
    int num_bboxes, 
    float confidence_threshold, 
    int *box_count, 
    int start_x,
    int start_y,
    float* result, 
    int max_image_boxes, 
    int num_box_element
)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes)
        return;

    int label = (int)*(labels + position);
    float confidence = *(scores + position);

    float *box = boxes + position * 4;

    if (confidence < confidence_threshold)
        return;
    int index = atomicAdd(box_count, 1);
    if (index >= max_image_boxes)
        return;

    float *pout_item = result + num_box_element * index;

    // *pout_item++     = box[0];
    // *pout_item++     = box[1];
    // *pout_item++     = box[2];
    // *pout_item++     = box[3];
    
    *pout_item++     = box[0] + start_x;
    *pout_item++     = box[1] + start_y;
    *pout_item++     = box[2] + start_x;
    *pout_item++     = box[3] + start_y;
    *pout_item++     = confidence;
    *pout_item++     = label;
    *pout_item++     = 1; // 1 = keep, 0 = ignore
    *pout_item++     = position;
    *pout_item++     = 0; // batch_index
}

__global__ void decode_single_mask_kernel(int left,
                                          int top,
                                          float *mask_weights,
                                          float *mask_predict,
                                          int mask_width,
                                          int mask_height,
                                          float *mask_out,
                                          int mask_dim,
                                          int out_width,
                                          int out_height)
{
    // mask_predict to mask_out
    // mask_weights @ mask_predict
    int dx = blockDim.x * blockIdx.x + threadIdx.x;
    int dy = blockDim.y * blockIdx.y + threadIdx.y;
    if (dx >= out_width || dy >= out_height)
        return;

    int sx = left + dx;
    int sy = top + dy;
    if (sx < 0 || sx >= mask_width || sy < 0 || sy >= mask_height)
    {
        mask_out[dy * out_width + dx] = 0;
        return;
    }

    float cumprod = 0;
    for (int ic = 0; ic < mask_dim; ++ic)
    {
        float cval = mask_predict[(ic * mask_height + sy) * mask_width + sx];
        float wval = mask_weights[ic];
        cumprod += cval * wval;
    }

    float alpha = 1.0f / (1.0f + exp(-cumprod));
    // 在这里先返回float值，再将mask采样回原图后才x255
    mask_out[dy * out_width + dx] = alpha;
};

__global__ void softmax_kernel(float* predict, int length, int *max_index)
{
    extern __shared__ float shared_data[];
    float *shared_max_vals = shared_data;
    int *shared_max_indices = (int*)&shared_max_vals[blockDim.x];

    int tid = threadIdx.x;

    // 1. 找出最大值和最大值的下标，并存储在共享内存中
    float max_val = -FLT_MAX;
    int max_idx = -1;
    for(int i=tid; i < length; i +=blockDim.x)
    {
        if (predict[i] > max_val)
        {
            max_val = predict[i];
            max_idx = i;
        }
    }
    shared_max_vals[tid] = max_val;
    shared_max_indices[tid] = max_idx;
    __syncthreads();

    // 2. 归约操作，找出全局最大值和对应的下标
    if (tid ==0)
    {
        for (int i=1; i< blockDim.x; i++)
        {
            if (shared_max_vals[i] > shared_max_vals[0])
            {
                shared_max_indices[0] = shared_max_indices[i];
                shared_max_vals[0] = shared_max_vals[i];
            }
        }
        *max_index = shared_max_indices[0];
    }
    __syncthreads();

    max_val = shared_max_vals[0];

    // 3. 计算指数并求和
    float sum_exp = 0.0f;
    for(int i=tid; i < length; i +=blockDim.x)
    {
        predict[i] = expf(predict[i] - max_val);
        sum_exp += predict[i];
    }
    shared_max_vals[tid] = sum_exp;
    __syncthreads();

    // 4. 汇总所有线程的指数和
    if (tid == 0)
    {
        for (int i=1; i < blockDim.x; i++)
        {
            shared_max_vals[0] += shared_max_vals[i];
        }
    }
    __syncthreads();
    float total_sum = shared_max_vals[0];

    // 5. 每个元素除以总和, 得到softmax值
    for (int i=tid; i < length; i += blockDim.x)
    {
        predict[i] /= total_sum;
    }

}


__global__ void max_kernel(float* predict, int length, int *max_index)
{
    extern __shared__ float shared_data[];
    float *shared_max_vals = shared_data;
    int *shared_max_indices = (int*)&shared_max_vals[blockDim.x];

    int tid = threadIdx.x;

    // 1. 找出最大值和最大值的下标，并存储在共享内存中
    float max_val = -FLT_MAX;
    int max_idx = -1;
    for(int i=tid; i < length; i +=blockDim.x)
    {
        if (predict[i] > max_val)
        {
            max_val = predict[i];
            max_idx = i;
        }
    }
    shared_max_vals[tid] = max_val;
    shared_max_indices[tid] = max_idx;
    __syncthreads();

    // 2. 归约操作，找出全局最大值和对应的下标
    if (tid ==0)
    {
        for (int i=1; i< blockDim.x; i++)
        {
            if (shared_max_vals[i] > shared_max_vals[0])
            {
                shared_max_indices[0] = shared_max_indices[i];
                shared_max_vals[0] = shared_max_vals[i];
            }
        }
        *max_index = shared_max_indices[0];
    }
}

__global__ void normalize_and_thres_mask_kernel(float* mask, unsigned char* mask_out, int numel, float confidence_threshold)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numel)
        return;

    float val = mask[idx] / 255.0f; // 归一化到0-1之间
    // 这里的confidence_threshold是为了过滤掉一些无效的mask值，避免后续处理时的干扰
    mask_out[idx] = val > confidence_threshold ? 255 : 0;
}

// logits: NCHW，每个空间位置做 argmax，scores 为对应类别的 softmax
__global__ void semantic_argmax_kernel(const float *logits, unsigned char *class_ids, float *scores, int batch,
                                       int num_classes, int spatial)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = batch * spatial;
    if (idx >= total)
        return;
    const int b = idx / spatial;
    const int s = idx - b * spatial;
    const float *base = logits + (size_t)b * num_classes * spatial + s;

    int best = 0;
    float maxv = base[0];
    for (int c = 1; c < num_classes; ++c)
    {
        float v = base[(size_t)c * spatial];
        if (v > maxv)
        {
            maxv = v;
            best = c;
        }
    }

    float sum = 0.f;
    for (int c = 0; c < num_classes; ++c)
        sum += expf(base[(size_t)c * spatial] - maxv);
    class_ids[idx] = (unsigned char)best;
    scores[idx]    = (sum > 0.f) ? (1.f / sum) : 0.f;
}

} // namespace cuda