#ifndef __INFINIOP_HANDLE_API_H__
#define __INFINIOP_HANDLE_API_H__

#include "../infinicore.h"

struct InfiniopHandle;

typedef struct InfiniopHandle *infiniopHandle_t;

INFINI_EXTERN_C __export infiniStatus_t infiniopCreateHandle(infiniopHandle_t *handle_ptr);

INFINI_EXTERN_C __export infiniStatus_t infiniopDestroyHandle(infiniopHandle_t handle);

#endif
