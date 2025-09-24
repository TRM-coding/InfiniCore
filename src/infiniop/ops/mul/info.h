#ifndef __RMS_NORM_INFO_H__
#define __RMS_NORM_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::mul {

class MulInfo {
    // MulInfo() = default;

public:
    infiniDtype_t atype;
    infiniDtype_t btype;
    std::vector<size_t> shape;
    std::vector<ptrdiff_t> y_strides;
    std::vector<std::vector<ptrdiff_t> > x_strides;

    size_t ndim() const { return shape.size(); }
    size_t dim() const { return shape[ndim() - 1]; }

    static utils::Result<MulInfo> create(
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x1_desc,
        infiniopTensorDescriptor_t x2_desc
    ) {

        //TODO:补充数据检查
        
        return utils::Result<MulInfo>(MulInfo{
            x1_desc->dtype(),
            x2_desc->dtype(),
            y_desc->shape(),
            y_desc->strides(),
            {x1_desc->strides(),x2_desc->strides(),}
        });
    }
};

} // namespace op::rms_norm

#endif // __RMS_NORM_INFO_H__
